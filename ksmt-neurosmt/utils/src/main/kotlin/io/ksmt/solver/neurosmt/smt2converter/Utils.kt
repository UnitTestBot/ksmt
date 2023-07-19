package io.ksmt.solver.neurosmt.smt2converter

import io.ksmt.solver.KSolverStatus
import java.io.File
import java.nio.file.Path

fun getAnswerForTest(path: Path): KSolverStatus {
    File(path.toUri()).useLines { lines ->
        for (line in lines) {
            when (line) {
                "(set-info :status sat)" -> return KSolverStatus.SAT
                "(set-info :status unsat)" -> return KSolverStatus.UNSAT
                "(set-info :status unknown)" -> return KSolverStatus.UNKNOWN
            }
        }
    }

    return KSolverStatus.UNKNOWN
}