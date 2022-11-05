import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import kotlin.system.measureTimeMillis
import kotlin.time.Duration.Companion.seconds

const val BLOCK_SIZE = 3
const val SIZE = 9
const val EMPTY_CELL_VALUE = 0

val sudokuIndices = 0 until SIZE

val sudokuTask = """
    9 * 6 | * 7 * | 4 * 3
    * * * | 4 * * | 2 * *
    * 7 * | * 2 3 | * 1 *
    ------ ------- ------
    5 * * | * * * | 1 * *
    * 4 * | 2 * 8 | * 6 *
    * * 3 | * * * | * * 5
    ------ ------- ------
    * 3 * | 7 * * | * 5 *
    * * 7 | * * 5 | * * *
    4 * 5 | * 1 * | 7 * 8
""".trimIndent()

fun main() = KContext().useWith {

    // Parse and display a given Sudoku grid.
    val initialGrid = parseSudoku(sudokuTask)
    println("Task:")
    println(printSudokuGrid(initialGrid))

    // Create symbolic variables for each cell of the Sudoku grid.
    val symbols = sudokuIndices.map { row ->
        sudokuIndices.map { col ->
            intSort.mkConst("x_${row}_${col}")
        }
    }

    // Create symbolic variables constraints according to the Sudoku rules.
    val rules = sudokuRules(symbols)

    // Create variable constraints from the filled cells of the Sudoku grid.
    val symbolAssignments = assignSymbols(symbols, initialGrid)

    // Create Z3 SMT solver instance.
    KZ3Solver(this).useWith {

        // Assert all constraints.
        rules.forEach { assert(it) }
        symbolAssignments.forEach { assert(it) }

        while (true) {
            val solution: List<List<Int>>
            val timeMs = measureTimeMillis {

                // Solve Sudoku.
                val status = check(timeout = 1.seconds)
                if (status != KSolverStatus.SAT) {
                    println("No more solutions")
                    return
                }

                // Get the SMT model and convert it to a Sudoku solution.
                solution = buildSolution(symbols, model())
            }

            println("Solution (in $timeMs ms):")
            println(printSudokuGrid(solution))

            assert(mkAnd(assignSymbols(symbols, solution)).not())
        }

    }


}

private fun KContext.sudokuRules(symbols: List<List<KExpr<KIntSort>>>): List<KExpr<KBoolSort>> {

    // Each cell has a value from 1 to 9.
    val symbolConstraints = symbols.flatten().map { (it ge 1.intExpr) and (it le 9.intExpr) }

    // Each row contains distinct numbers.
    val rowDistinctConstraints = symbols.map { row -> mkDistinct(row) }

    // Each column contains distinct numbers.
    val colDistinctConstraints = sudokuIndices.map { col ->
        val colSymbols = sudokuIndices.map { row -> symbols[row][col] }
        mkDistinct(colSymbols)
    }

    // Each 3x3 block contains distinct numbers.
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

private fun KContext.assignSymbols(
    symbols: List<List<KExpr<KIntSort>>>,
    grid: List<List<Int>>
): List<KExpr<KBoolSort>> =
    sudokuIndices.flatMap { row ->
        sudokuIndices.mapNotNull { col ->
            val value = grid[row][col]
            if (value != EMPTY_CELL_VALUE) symbols[row][col] eq value.intExpr else null
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

private fun parseSudoku(task: String): List<List<Int>> =
    task.lines()
        .map { row -> row.mapNotNull { it.cellValueOrNull() } }
        .filterNot { it.isEmpty() }

private fun Char.cellValueOrNull(): Int? = when {
    isDigit() -> digitToInt()
    this == '*' -> EMPTY_CELL_VALUE
    else -> null
}

private fun printSudokuGrid(grid: List<List<Int>>) = buildString {
    for ((rowIdx, row) in grid.withIndex()) {
        for ((colIdx, cell) in row.withIndex()) {
            append(if (cell == EMPTY_CELL_VALUE) "*" else cell)
            append(" ")
            if ((colIdx + 1) % BLOCK_SIZE == 0 && (colIdx + 1) != SIZE) {
                append("| ")
            }
        }
        appendLine()
        if ((rowIdx + 1) % BLOCK_SIZE == 0 && (rowIdx + 1) != SIZE) {
            appendLine("-".repeat((SIZE + 2) * 2))
        }
    }
}

private inline fun <T : AutoCloseable?, R> T.useWith(block: T.() -> R): R = use { it.block() }
