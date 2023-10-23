package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.KSolverStatus.UNSAT
import io.ksmt.solver.maxsat.constraints.SoftConstraint
import io.ksmt.solver.maxsat.solvers.KMaxSATSolver
import io.ksmt.solver.z3.KZ3SolverConfiguration
import io.ksmt.utils.getValue
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

abstract class KMaxSATSolverTest {
    abstract fun getSolver(): KMaxSATSolver<KZ3SolverConfiguration>

    protected val ctx: KContext = KContext()
    private lateinit var maxSATSolver: KMaxSATSolver<KZ3SolverConfiguration>

    @BeforeEach
    fun initSolver() {
        maxSATSolver = getSolver()
    }

    @AfterEach
    fun closeSolver() = maxSATSolver.close()

    @Test
    fun hardConstraintsUnsatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        maxSATSolver.assert(a)
        maxSATSolver.assert(b)
        maxSATSolver.assert(!a)

        val maxSATResult = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult.hardConstraintsSATStatus == UNSAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun hardConstraintsUnsatNoSoftTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        maxSATSolver.assert(a)
        maxSATSolver.assert(b)
        maxSATSolver.assert(!a)

        maxSATSolver.assertSoft(a and !b, 3u)
        maxSATSolver.assertSoft(!a, 5u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == UNSAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun noSoftConstraintsTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        maxSATSolver.assert(a)
        maxSATSolver.assert(b)
        maxSATSolver.assert(c)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun noSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        maxSATSolver.assert(a)
        maxSATSolver.assert(b)
        maxSATSolver.assert(c)
        maxSATSolver.assertSoft(a and !c, 3u)
        maxSATSolver.assertSoft(!a, 5u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun oneOfTwoSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)
        maxSATSolver.assert(!a or b)

        maxSATSolver.assertSoft(a or !b, 2u)
        maxSATSolver.assertSoft(!a or !b, 3u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 1)
        assertSoftConstraintsSat(listOf(SoftConstraint(!a or !b, 3u)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun twoOfFourSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        val d by boolSort

        maxSATSolver.assert(a or b)
        maxSATSolver.assert(c or d)
        maxSATSolver.assert(!a or b)
        maxSATSolver.assert(!c or d)

        maxSATSolver.assertSoft(a or !b, 2u)
        maxSATSolver.assertSoft(c or !d, 2u)
        maxSATSolver.assertSoft(!a or !b, 3u)
        maxSATSolver.assertSoft(!c or !d, 3u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        assertSoftConstraintsSat(
            listOf(SoftConstraint(!a or !b, 3u), SoftConstraint(!c or !d, 3u)),
            maxSATResult.satSoftConstraints,
        )
    }

    @Test
    fun sixOfEightSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        val d by boolSort

        maxSATSolver.assertSoft(a or b, 9u)
        maxSATSolver.assertSoft(c or d, 9u)
        maxSATSolver.assertSoft(!a or b, 9u)
        maxSATSolver.assertSoft(!c or d, 9u)
        maxSATSolver.assertSoft(a or !b, 2u)
        maxSATSolver.assertSoft(c or !d, 2u)
        maxSATSolver.assertSoft(!a or !b, 3u)
        maxSATSolver.assertSoft(!c or !d, 3u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 6)
        assertEquals(42u, maxSATResult.satSoftConstraints.sumOf { it.weight })
    }

    @Test
    fun twoOfThreeSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)

        maxSATSolver.assertSoft(!a or b, 4u)
        maxSATSolver.assertSoft(a or !b, 6u)
        maxSATSolver.assertSoft(!a or !b, 2u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        val softConstraintsToAssertSAT =
            listOf(SoftConstraint(!a or b, 4u), SoftConstraint(a or !b, 6u))
        assertSoftConstraintsSat(softConstraintsToAssertSAT, maxSATResult.satSoftConstraints)
    }

    @Test
    fun sameExpressionSoftConstraintsSatTest() = with(ctx) {
        val x by boolSort
        val y by boolSort

        maxSATSolver.assert(x or y)

        maxSATSolver.assertSoft(!x or y, 7u)
        maxSATSolver.assertSoft(x or !y, 6u)
        maxSATSolver.assertSoft(!x or !y, 3u)
        maxSATSolver.assertSoft(!x or !y, 4u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 3)
        assertSoftConstraintsSat(
            listOf(
                SoftConstraint(!x or y, 7u),
                SoftConstraint(!x or !y, 3u),
                SoftConstraint(!x or !y, 4u),
            ),
            maxSATResult.satSoftConstraints,
        )
    }

    @Test
    fun sameExpressionSoftConstraintsUnsatTest() = with(ctx) {
        val x by boolSort
        val y by boolSort

        maxSATSolver.assert(x or y)

        maxSATSolver.assertSoft(!x or y, 6u)
        maxSATSolver.assertSoft(x or !y, 6u)
        maxSATSolver.assertSoft(!x or !y, 3u)
        maxSATSolver.assertSoft(!x or !y, 2u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        assertSoftConstraintsSat(
            listOf(SoftConstraint(!x or y, 6u), SoftConstraint(x or !y, 6u)),
            maxSATResult.satSoftConstraints,
        )
    }

    @Test
    fun chooseOneConstraintByWeightTest() = with(ctx) {
        val z by boolSort
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(z)
        maxSATSolver.assertSoft(a and b, 1u)
        maxSATSolver.assertSoft(!a and !b, 5u)
        maxSATSolver.assertSoft(a and b and z, 2u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 1)
        assertSoftConstraintsSat(listOf(SoftConstraint(!a and !b, 5u)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun equalWeightsTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)
        maxSATSolver.assert(!a or b)

        maxSATSolver.assertSoft(a or !b, 2u)
        maxSATSolver.assertSoft(!a or !b, 2u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.satSoftConstraints.size == 1)
    }

    @Test
    fun inequalitiesTest() = with(ctx) {
        val x by intSort
        val y by intSort

        val a1 = x gt 0.expr
        val a2 = x lt y
        val a3 = x + y le 0.expr

        maxSATSolver.assert(a3 eq a1)
        maxSATSolver.assert(a3 or a2)

        maxSATSolver.assertSoft(a3, 3u)
        maxSATSolver.assertSoft(!a3, 5u)
        maxSATSolver.assertSoft(!a1, 10u)
        maxSATSolver.assertSoft(!a2, 3u)

        val maxSATResult = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResult.satSoftConstraints.size == 2)
        assertSoftConstraintsSat(
            listOf(SoftConstraint(!a3, 5u), SoftConstraint(!a1, 10u)),
            maxSATResult.satSoftConstraints,
        )
    }

    @Test
    fun oneScopePushPopTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSATSolver.assert(a or b)

        maxSATSolver.assertSoft(!a or b, 1u)

        val maxSATResult = maxSATSolver.checkMaxSAT()
        assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1u)), maxSATResult.satSoftConstraints)

        maxSATSolver.push()

        maxSATSolver.assertSoft(a or !b, 1u)
        maxSATSolver.assertSoft(!a or !b, 1u)

        val maxSATResultScoped = maxSATSolver.checkMaxSAT()

        assertTrue(maxSATResult.hardConstraintsSATStatus == SAT)
        assertTrue(maxSATResult.maxSATSucceeded)
        assertTrue(maxSATResultScoped.satSoftConstraints.size == 2)

        maxSATSolver.pop()

        assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1u)), maxSATResult.satSoftConstraints)
    }

    @Test
    fun threeScopesPushPopTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort

        maxSATSolver.assert(a)

        maxSATSolver.push()
        maxSATSolver.assertSoft(a or b, 1u)
        maxSATSolver.assertSoft(c or b, 1u)

        val maxSATResult = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult.satSoftConstraints.size == 2)

        maxSATSolver.push()
        maxSATSolver.assertSoft(!b and !c, 2u)
        val maxSATResult2 = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult2.satSoftConstraints.size == 2)

        maxSATSolver.push()
        maxSATSolver.assertSoft(a or c, 1u)
        val maxSATResult3 = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult3.satSoftConstraints.size == 3)

        maxSATSolver.pop(2u)
        val maxSATResult4 = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult4.satSoftConstraints.size == 2)

        maxSATSolver.pop()
        val maxSATResult5 = maxSATSolver.checkMaxSAT()
        assertTrue(maxSATResult5.satSoftConstraints.isEmpty())
    }

    @Test
    fun similarExpressionsTest(): Unit = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSATSolver.assertSoft(a or b, 1u)
        maxSATSolver.assertSoft(!a or b, 1u)
        maxSATSolver.assertSoft(a or !b, 1u)
        maxSATSolver.assertSoft(!a or !b, 1u)
        maxSATSolver.assertSoft(!a or !b or !b, 1u)

        maxSATSolver.checkMaxSAT()
    }

    private fun assertSoftConstraintsSat(
        constraintsToAssert: List<SoftConstraint>,
        satConstraints: List<SoftConstraint>,
    ) {
        for (constraint in constraintsToAssert) {
            assertTrue(
                satConstraints.any {
                    constraint.expression.internEquals(it.expression) && constraint.weight == it.weight
                },
            )
        }
    }
}
