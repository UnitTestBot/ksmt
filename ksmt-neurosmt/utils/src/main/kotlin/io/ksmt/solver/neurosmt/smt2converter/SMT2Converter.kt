package io.ksmt.solver.neurosmt.smt2converter

import io.ksmt.KContext
import io.ksmt.parser.KSMTLibParseException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.*
import me.tongfei.progressbar.ProgressBar
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
    var sat = 0; var unsat = 0; var unknown = 0

    val ctx = KContext()

    var curIdx = 0
    ProgressBar.wrap(files, "converting smt2 files").forEach {
        if (!it.name.endsWith(".smt2")) {
            return@forEach
        }

        val answer = getAnswerForTest(it)

        when (answer) {
            KSolverStatus.SAT -> sat++
            KSolverStatus.UNSAT -> unsat++
            KSolverStatus.UNKNOWN -> {
                unknown++
                return@forEach
            }
        }

        with(ctx) {
            val formula = try {
                ok++
                mkAnd(KZ3SMTLibParser(ctx).parse(it))
            } catch (e: KSMTLibParseException) {
                fail++
                return@forEach
            }

            val extractor = FormulaGraphExtractor(ctx, formula, FileOutputStream("$outputRoot/$answer/$curIdx"))
            extractor.extractGraph()
        }

        curIdx++
    }

    println()
    println("parsed: $ok; failed: $fail")
    println("sat: $sat; unsat: $unsat; unknown: $unknown")
}