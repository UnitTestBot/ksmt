package io.ksmt.solver.neurosmt.runtime.standalone

import com.github.ajalt.clikt.core.NoSuchParameter
import com.github.ajalt.clikt.core.PrintHelpMessage
import com.github.ajalt.clikt.core.UsageError
import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.deserialize
import io.ksmt.solver.neurosmt.runtime.NeuroSMTModelRunner
import java.io.File
import java.io.FileInputStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isRegularFile
import kotlin.io.path.name
import kotlin.io.path.pathString

fun printStats(
    sat: Int, unsat: Int, skipped: Int,
    ok: Int, fail: Int,
    confusionMatrix: Map<Pair<KSolverStatus, KSolverStatus>, Int>
) {
    println("$sat sat; $unsat unsat; $skipped skipped")
    println("$ok ok; $fail failed")
    println()

    val satToSat = confusionMatrix.getOrDefault(KSolverStatus.SAT to KSolverStatus.SAT, 0)
    val unsatToUnsat = confusionMatrix.getOrDefault(KSolverStatus.UNSAT to KSolverStatus.UNSAT, 0)
    val unsatToSat = confusionMatrix.getOrDefault(KSolverStatus.UNSAT to KSolverStatus.SAT, 0)
    val satToUnsat = confusionMatrix.getOrDefault(KSolverStatus.SAT to KSolverStatus.UNSAT, 0)

    println("target vs output")
    println("  sat  vs   sat : $satToSat")
    println("unsat  vs unsat : $unsatToUnsat")
    println("unsat  vs   sat : $unsatToSat")
    println("  sat  vs unsat : $satToUnsat")
    println()
}

fun main(args: Array<String>) {
    val arguments = CLArgs()
    try {
        arguments.parse(args)
    } catch (e: PrintHelpMessage) {
        println(arguments.getFormattedHelp())
        return
    } catch (e: NoSuchParameter) {
        println(e.message)
        return
    } catch (e: UsageError) {
        println(e.message)
        return
    } catch (e: Exception) {
        println("Error!\n$e")
        return
    }

    val ctx = KContext(
        astManagementMode = KContext.AstManagementMode.NO_GC,
        simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY
    )

    val files = Files.walk(Path.of(arguments.datasetPath)).filter { it.isRegularFile() }.toList()

    val runner = NeuroSMTModelRunner(
        ctx,
        ordinalsPath = arguments.ordinalsPath,
        embeddingPath = arguments.embeddingsPath,
        convPath = arguments.convPath,
        decoderPath = arguments.decoderPath
    )

    var sat = 0; var unsat = 0; var skipped = 0
    var ok = 0; var fail = 0

    val confusionMatrix = mutableMapOf<Pair<KSolverStatus, KSolverStatus>, Int>()

    files.forEachIndexed sample@{ ind, it ->
        if (ind % 100 == 0) {
            println("\n#$ind:")
            printStats(sat, unsat, skipped, ok, fail, confusionMatrix)
        }

        val sampleFile = File(it.pathString)

        val assertList = try {
            deserialize(ctx, FileInputStream(sampleFile))
        } catch (e: Exception) {
            skipped++
            return@sample
        }

        val answer = when {
            it.name.endsWith("-sat") -> KSolverStatus.SAT
            it.name.endsWith("-unsat") -> KSolverStatus.UNSAT
            else -> KSolverStatus.UNKNOWN
        }

        if (answer == KSolverStatus.UNKNOWN) {
            skipped++
            return@sample
        }

        val prob = with(ctx) {
            val formula = when (assertList.size) {
                0 -> {
                    skipped++
                    return@sample
                }
                1 -> {
                    assertList[0]
                }
                else -> {
                    mkAnd(assertList)
                }
            }

            runner.run(formula)
        }

        val output = if (prob < arguments.threshold) {
            KSolverStatus.UNSAT
        } else {
            KSolverStatus.SAT
        }

        when (answer) {
            KSolverStatus.SAT -> sat++
            KSolverStatus.UNSAT -> unsat++
            else -> { /* can't happen */ }
        }

        if (output == answer) {
            ok++
        } else {
            fail++
        }

        confusionMatrix.compute(answer to output) { _, v ->
            if (v == null) {
                1
            } else {
                v + 1
            }
        }
    }

    println()
    printStats(sat, unsat, skipped, ok, fail, confusionMatrix)
}