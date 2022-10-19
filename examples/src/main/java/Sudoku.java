import org.ksmt.KContext;
import org.ksmt.expr.KExpr;
import org.ksmt.expr.KInt32NumExpr;
import org.ksmt.solver.KModel;
import org.ksmt.solver.KSolverStatus;
import org.ksmt.solver.z3.KZ3Solver;
import org.ksmt.sort.KBoolSort;
import org.ksmt.sort.KIntSort;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

public class Sudoku {
    private static final int BLOCK_SIZE = 3;
    private static final int SIZE = 9;
    private static final int EMPTY_CELL_VALUE = 0;

    private static final String sudokuTask = """
            2 * * | 9 * 6 | * * 1
            * * 6 | * 4 * | * * 9
            * * * | 5 2 * | 4 * *
            ------ ------- ------
            3 * 2 | * * 7 | * 5 *
            * * * | 2 * * | 1 * *
            * 9 * | 3 * * | 7 * *
            ------ ------- ------
            * 8 7 | * 5 * | 3 1 *
            6 * 3 | * 1 * | 8 * *
            4 * * | * * 9 | * * *""".stripIndent();

    public static void main(String[] args) {
        // Parse and display a given Sudoku grid.
        final List<List<Integer>> initialGrid = parseSudoku(sudokuTask);
        System.out.println("Task:");
        System.out.println(printSudokuGrid(initialGrid));

        try (final KContext ctx = new KContext()) {
            // Create symbolic variables for each cell of the Sudoku grid.
            final List<List<KExpr<KIntSort>>> symbols = createSymbolicGrid(ctx);

            // Create symbolic variables constraints according to the Sudoku rules.
            final List<KExpr<KBoolSort>> rules = sudokuRules(ctx, symbols);

            // Create variable constraints from the filled cells of the Sudoku grid.
            final List<KExpr<KBoolSort>> symbolAssignments = assignSymbols(ctx, symbols, initialGrid);

            // Create Z3 SMT solver instance.
            try (final KZ3Solver solver = new KZ3Solver(ctx)) {
                // Assert all constraints.
                rules.forEach(expr -> JavaCompatibilityUtils.solverAssert(solver, expr));
                symbolAssignments.forEach(expr -> JavaCompatibilityUtils.solverAssert(solver, expr));

                // Solve Sudoku.
                final KSolverStatus status = JavaCompatibilityUtils.check(solver);
                if (status != KSolverStatus.SAT) {
                    throw new IllegalArgumentException("Sudoku is unsolvable");
                }

                // Get the SMT model and convert it to a Sudoku solution.
                final List<List<Integer>> solution = buildSolution(symbols, solver.model());

                System.out.println("Solution:");
                System.out.println(printSudokuGrid(solution));
            }
        }
    }

    private static List<List<KExpr<KIntSort>>> createSymbolicGrid(final KContext ctx) {
        final List<List<KExpr<KIntSort>>> grid = new ArrayList<>();
        for (int row = 0; row < SIZE; row++) {
            final List<KExpr<KIntSort>> gridRow = new ArrayList<>();
            for (int col = 0; col < SIZE; col++) {
                gridRow.add(ctx.mkConst(ctx.getIntSort(), "x_" + row + "_" + col));
            }
            grid.add(gridRow);
        }
        return grid;
    }

    private static List<KExpr<KBoolSort>> sudokuRules(final KContext ctx,
                                                      final List<List<KExpr<KIntSort>>> symbols) {

        // Each cell has a value from 1 to 9.
        final Stream<KExpr<KBoolSort>> symbolConstraints = symbols
                .stream().flatMap(Collection::stream)
                .map(cell -> ctx.and(
                        ctx.ge(cell, ctx.getIntExpr(1)),
                        ctx.le(cell, ctx.getIntExpr(9))));

        // Each row contains distinct numbers.
        final Stream<KExpr<KBoolSort>> rowDistinctConstraints = symbols.stream()
                .map(row -> ctx.mkDistinct(row));

        // Each column contains distinct numbers.
        final List<KExpr<KBoolSort>> colDistinctConstraints = new ArrayList<>();
        for (int col = 0; col < SIZE; col++) {
            final List<KExpr<KIntSort>> colSymbols = new ArrayList<>();
            for (int row = 0; row < SIZE; row++) {
                colSymbols.add(symbols.get(row).get(col));
            }
            colDistinctConstraints.add(ctx.mkDistinct(colSymbols));
        }

        // Each 3x3 block contains distinct numbers.
        final List<KExpr<KBoolSort>> blockDistinctConstraints = new ArrayList<>();
        for (int blockRow = 0; blockRow < SIZE; blockRow += BLOCK_SIZE) {
            for (int blockCol = 0; blockCol < SIZE; blockCol += BLOCK_SIZE) {
                final List<KExpr<KIntSort>> blockSymbols = new ArrayList<>();
                for (int row = blockRow; row < blockRow + BLOCK_SIZE; row++) {
                    for (int col = blockCol; col < blockCol + BLOCK_SIZE; col++) {
                        blockSymbols.add(symbols.get(row).get(col));
                    }
                }
                blockDistinctConstraints.add(ctx.mkDistinct(blockSymbols));
            }
        }

        return Stream.concat(
                Stream.concat(symbolConstraints, rowDistinctConstraints),
                Stream.concat(colDistinctConstraints.stream(), blockDistinctConstraints.stream())
        ).toList();
    }

    private static List<KExpr<KBoolSort>> assignSymbols(final KContext ctx,
                                                        final List<List<KExpr<KIntSort>>> symbols,
                                                        final List<List<Integer>> grid) {
        final List<KExpr<KBoolSort>> assignments = new ArrayList<>();
        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                int cell = grid.get(row).get(col);
                if (cell != EMPTY_CELL_VALUE) {
                    assignments.add(ctx.eq(symbols.get(row).get(col), ctx.getIntExpr(cell)));
                }
            }
        }
        return assignments;
    }

    private static List<List<Integer>> buildSolution(final List<List<KExpr<KIntSort>>> symbols,
                                                     final KModel model) {
        final List<List<Integer>> solution = new ArrayList<>();
        for (int row = 0; row < SIZE; row++) {
            final List<Integer> solutionRow = new ArrayList<>();
            for (int col = 0; col < SIZE; col++) {
                final KExpr<KIntSort> symbol = symbols.get(row).get(col);
                final KExpr<KIntSort> value = model.eval(symbol, false);
                solutionRow.add(((KInt32NumExpr) value).getValue());
            }
            solution.add(solutionRow);
        }
        return solution;
    }

    private static List<List<Integer>> parseSudoku(final String task) {
        return task
                .lines()
                .map(row -> row
                        .chars()
                        .mapToObj(Sudoku::cellValueOrNull)
                        .filter(Objects::nonNull))
                .map(Stream::toList)
                .filter(it -> !it.isEmpty())
                .toList();
    }

    private static Integer cellValueOrNull(int symbol) {
        if (Character.isDigit(symbol)) {
            return Character.digit(symbol, 10);
        }
        if (symbol == '*') {
            return EMPTY_CELL_VALUE;
        }
        return null;
    }

    private static String printSudokuGrid(final List<List<Integer>> grid) {
        final StringBuilder str = new StringBuilder();

        for (int rowIdx = 0; rowIdx < grid.size(); rowIdx++) {
            final List<Integer> row = grid.get(rowIdx);
            for (int colIdx = 0; colIdx < row.size(); colIdx++) {
                final Integer cell = row.get(colIdx);
                str.append(EMPTY_CELL_VALUE == cell ? "*" : cell);
                str.append(" ");
                if ((colIdx + 1) % BLOCK_SIZE == 0 && (colIdx + 1) != SIZE) {
                    str.append("| ");
                }
            }
            str.append('\n');
            if ((rowIdx + 1) % BLOCK_SIZE == 0 && (rowIdx + 1) != SIZE) {
                str.append("-".repeat((SIZE + 2) * 2));
                str.append('\n');
            }
        }

        return str.toString();
    }
}
