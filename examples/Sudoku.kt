package org.ksmt.solver.z3

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import kotlin.time.Duration.Companion.seconds

const val BLOCK_SIZE = 3
const val SIZE = 9

val sudokuTask = listOf(
    listOf(2, 0, 0, 9, 0, 6, 0, 0, 1),
    listOf(0, 0, 6, 0, 4, 0, 0, 0, 9),
    listOf(0, 0, 0, 5, 2, 0, 4, 0, 0),
    listOf(3, 0, 2, 0, 0, 7, 0, 5, 0),
    listOf(0, 0, 0, 2, 0, 0, 1, 0, 0),
    listOf(0, 9, 0, 3, 0, 0, 7, 0, 0),
    listOf(0, 8, 7, 0, 5, 0, 3, 1, 0),
    listOf(6, 0, 3, 0, 1, 0, 8, 0, 0),
    listOf(4, 0, 0, 0, 0, 9, 0, 0, 0)
)

val sudokuIndices = 0 until SIZE

fun main() = KContext().useWith {
    val symbols = sudokuIndices.map { row ->
        sudokuIndices.map { col ->
            intSort.mkConst("x_${row}_${col}")
        }
    }
    val rules = sudokuRules(symbols)
    val symbolAssignments = assignSymbols(symbols)
    val solution = KZ3Solver(this).useWith {
        rules.forEach { assert(it) }
        symbolAssignments.forEach { assert(it) }
        val status = check(timeout = 1.seconds)
        check(status == KSolverStatus.SAT) { "Sudoku is unsolvable" }
        buildSolution(symbols, model())
    }
    println(solution)
}

private inline fun <T : AutoCloseable?, R> T.useWith(block: T.() -> R): R = use { it.block() }

private fun KContext.sudokuRules(symbols: List<List<KExpr<KIntSort>>>): List<KExpr<KBoolSort>> {
    val symbolConstraints = symbols.flatten().map { (it ge 1.intExpr) and (it le 9.intExpr) }
    val rowDistinctConstraints = symbols.map { row -> mkDistinct(row) }
    val colDistinctConstraints = sudokuIndices.map { col ->
        val colSymbols = sudokuIndices.map { row -> symbols[row][col] }
        mkDistinct(colSymbols)
    }
    val blockDistinctConstraints = (0 until SIZE step BLOCK_SIZE).flatMap { blockRow ->
        (0 until SIZE step BLOCK_SIZE).map { blockCol ->
            val block = (blockRow until blockRow + BLOCK_SIZE).flatMap { row ->
                (blockCol until blockCol + BLOCK_SIZE).map { col ->
                    symbols[row][col]
                }
            }
            mkDistinct(block)
        }
    }
    return symbolConstraints + rowDistinctConstraints + colDistinctConstraints + blockDistinctConstraints
}

private fun KContext.assignSymbols(symbols: List<List<KExpr<KIntSort>>>): List<KExpr<KBoolSort>> =
    sudokuIndices.flatMap { row ->
        sudokuIndices.mapNotNull { col ->
            val value = sudokuTask[row][col]
            if (value != 0) symbols[row][col] eq value.intExpr else null
        }
    }

private fun KContext.buildSolution(symbols: List<List<KExpr<KIntSort>>>, model: KModel): List<List<Int>> =
    symbols.map { row ->
        row.map { symbol ->
            val value = model.eval(symbol)
            value as KInt32NumExpr
            value.value
        }
    }
