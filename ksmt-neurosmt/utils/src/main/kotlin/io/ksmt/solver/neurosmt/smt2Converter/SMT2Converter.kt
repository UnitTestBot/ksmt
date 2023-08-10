package io.ksmt.solver.neurosmt.smt2Converter

import io.ksmt.KContext
import io.ksmt.parser.KSMTLibParseException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.neurosmt.FormulaGraphExtractor
import io.ksmt.solver.neurosmt.getAnswerForTest
import io.ksmt.solver.z3.KZ3SMTLibParser
import me.tongfei.progressbar.ProgressBar
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.isRegularFile
import kotlin.io.path.name

fun main(args: Array<String>) {
    val inputRoot = args[0]
    val outputRoot = args[1]

    val files = Files.walk(Path.of(inputRoot)).filter { it.isRegularFile() }

    var ok = 0; var fail = 0
    var sat = 0; var unsat = 0; var skipped = 0

    val ctx = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)

    var curIdx = 0
    ProgressBar.wrap(files, "converting smt2 files").forEach {
        if (!it.name.endsWith(".smt2")) {
            return@forEach
        }

        val answer = getAnswerForTest(it)

        if (answer == KSolverStatus.UNKNOWN) {
            skipped++
            return@forEach
        }

        with(ctx) {
            val formula = try {
                val assertList = KZ3SMTLibParser(ctx).parse(it)
                when (assertList.size) {
                    0 -> {
                        skipped++
                        return@forEach
                    }
                    1 -> {
                        ok++
                        assertList[0]
                    }
                    else -> {
                        ok++
                        mkAnd(assertList)
                    }
                }
            } catch (e: KSMTLibParseException) {
                fail++
                e.printStackTrace()
                return@forEach
            }

            val outputFile = File("$outputRoot/$curIdx-${answer.toString().lowercase()}")
            val outputStream = FileOutputStream(outputFile)
            outputStream.write("; $it\n".encodeToByteArray())

            val extractor = FormulaGraphExtractor(ctx, formula, outputStream)
            extractor.extractGraph()
        }

        when (answer) {
            KSolverStatus.SAT -> sat++
            KSolverStatus.UNSAT -> unsat++
            else -> { /* can't happen */ }
        }

        curIdx++
    }

    println()
    println("processed: $ok; failed: $fail")
    println("sat: $sat; unsat: $unsat; skipped: $skipped")
}