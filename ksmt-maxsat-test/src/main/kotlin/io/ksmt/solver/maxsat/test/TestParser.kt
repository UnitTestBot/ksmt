package io.ksmt.solver.maxsat.test

import io.ksmt.KContext
import io.ksmt.solver.maxsat.constraints.Constraint
import io.ksmt.solver.maxsat.constraints.HardConstraint
import io.ksmt.solver.maxsat.constraints.SoftConstraint
import io.ksmt.utils.mkConst
import java.io.File
import java.nio.file.Path
import kotlin.io.path.extension
import kotlin.io.path.notExists

fun parseTest(path: Path, ctx: KContext): List<Constraint> {
    var currentState: CurrentLineState

    val constraints = mutableListOf<Constraint>()

    if (path.notExists()) {
        error("Path [$path] does not exist")
    }

    require(path.extension == "wcnf") {
        "File extension cannot be '${path.extension}' as it must be 'wcnf'"
    }

    File(path.toUri()).forEachLine { wcnfStr ->
        currentState = when {
            wcnfStr.startsWith("c") -> CurrentLineState.COMMENT
            wcnfStr.startsWith("h ") -> CurrentLineState.HARD_CONSTRAINT
            wcnfStr.substringBefore(" ").toUIntOrNull() != null -> CurrentLineState.SOFT_CONSTRAINT
            else -> CurrentLineState.ERROR
        }

        if (currentState == CurrentLineState.ERROR) {
            error("Unexpected string:\n\"$wcnfStr\"")
        }

        if (currentState == CurrentLineState.HARD_CONSTRAINT || currentState == CurrentLineState.SOFT_CONSTRAINT) {
            val constraintBeginIndex = getConstraintBeginIndex(currentState, wcnfStr)
            val constraintEndIndex = wcnfStr.lastIndex - 1

            // We do not take the last element in the string as this is a zero indicating the constraint end.
            val constraintStr = wcnfStr.substring(constraintBeginIndex, constraintEndIndex)
            val constraintStrSplit = constraintStr.split(" ").filterNot { it.isEmpty() }
            val constraintIntVars = constraintStrSplit.map { x -> x.toInt() }

            with(ctx) {
                val constraintVars = constraintIntVars.map {
                    if (it < 0) {
                        !ctx.boolSort.mkConst("${-it}")
                    } else if (it > 0) {
                        ctx.boolSort.mkConst("$it")
                    } else {
                        error("Map element should not be 0!")
                    }
                }

                val constraint = constraintVars.reduce { x, y -> x or y }

                if (currentState == CurrentLineState.HARD_CONSTRAINT) {
                    constraints.add(HardConstraint(constraint))
                } else {
                    val weight = wcnfStr.substringBefore(" ").toUInt()
                    constraints.add(SoftConstraint(constraint, weight))
                }
            }
        }
    }

    return constraints
}

private fun getConstraintBeginIndex(currentState: CurrentLineState, str: String): Int = when (currentState) {
    CurrentLineState.HARD_CONSTRAINT -> 2
    CurrentLineState.SOFT_CONSTRAINT -> str.substringBefore(" ").length + 1
    else -> error("Unexpected value: [${currentState}]")
}
