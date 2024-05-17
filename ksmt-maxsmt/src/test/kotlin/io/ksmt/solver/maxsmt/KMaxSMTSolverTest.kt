package io.ksmt.solver.maxsmt

import io.ksmt.KContext
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.KSolverStatus.UNSAT
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverBase
import io.ksmt.utils.getValue
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

abstract class KMaxSMTSolverTest {
    abstract fun getSolver(): KMaxSMTSolverBase<out KSolverConfiguration>

    protected val ctx: KContext = KContext()
    private lateinit var maxSMTSolver: KMaxSMTSolverBase<out KSolverConfiguration>

    @BeforeEach
    fun initSolver() {
        maxSMTSolver = getSolver()
    }

    @AfterEach
    fun closeSolver() = maxSMTSolver.close()

    @Test
    fun hardConstraintsUnsatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        maxSMTSolver.assert(a)
        maxSMTSolver.assert(b)
        maxSMTSolver.assert(!a)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()
        assertTrue(maxSMTResult.hardConstraintsSatStatus == UNSAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun hardConstraintsUnsatNoSoftTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        maxSMTSolver.assert(a)
        maxSMTSolver.assert(b)
        maxSMTSolver.assert(!a)

        maxSMTSolver.assertSoft(a and !b, 3u)
        maxSMTSolver.assertSoft(!a, 5u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == UNSAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun noSoftConstraintsTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        maxSMTSolver.assert(a)
        maxSMTSolver.assert(b)
        maxSMTSolver.assert(c)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun noSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        maxSMTSolver.assert(a)
        maxSMTSolver.assert(b)
        maxSMTSolver.assert(c)
        maxSMTSolver.assertSoft(a and !c, 3u)
        maxSMTSolver.assertSoft(!a, 5u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.isEmpty())
    }

    @Test
    fun oneOfTwoSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSMTSolver.assert(a or b)
        maxSMTSolver.assert(!a or b)

        maxSMTSolver.assertSoft(a or !b, 2u)
        maxSMTSolver.assertSoft(!a or !b, 3u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 1)
        assertSoftConstraintsSat(listOf(SoftConstraint(!a or !b, 3u)), maxSMTResult.satSoftConstraints)
    }

    @Test
    fun twoOfFourSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        val d by boolSort

        maxSMTSolver.assert(a or b)
        maxSMTSolver.assert(c or d)
        maxSMTSolver.assert(!a or b)
        maxSMTSolver.assert(!c or d)

        maxSMTSolver.assertSoft(a or !b, 2u)
        maxSMTSolver.assertSoft(c or !d, 2u)
        maxSMTSolver.assertSoft(!a or !b, 3u)
        maxSMTSolver.assertSoft(!c or !d, 3u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 2)
        assertSoftConstraintsSat(
            listOf(SoftConstraint(!a or !b, 3u), SoftConstraint(!c or !d, 3u)),
            maxSMTResult.satSoftConstraints,
        )
    }

    @Test
    fun sixOfEightSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        val d by boolSort

        maxSMTSolver.assertSoft(a or b, 9u)
        maxSMTSolver.assertSoft(c or d, 9u)
        maxSMTSolver.assertSoft(!a or b, 9u)
        maxSMTSolver.assertSoft(!c or d, 9u)
        maxSMTSolver.assertSoft(a or !b, 2u)
        maxSMTSolver.assertSoft(c or !d, 2u)
        maxSMTSolver.assertSoft(!a or !b, 3u)
        maxSMTSolver.assertSoft(!c or !d, 3u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 6)
        assertEquals(42u, maxSMTResult.satSoftConstraints.sumOf { it.weight })
    }

    @Test
    fun twoOfThreeSoftConstraintsSatTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSMTSolver.assert(a or b)

        maxSMTSolver.assertSoft(!a or b, 4u)
        maxSMTSolver.assertSoft(a or !b, 6u)
        maxSMTSolver.assertSoft(!a or !b, 2u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 2)
        val softConstraintsToAssertSAT =
            listOf(SoftConstraint(!a or b, 4u), SoftConstraint(a or !b, 6u))
        assertSoftConstraintsSat(softConstraintsToAssertSAT, maxSMTResult.satSoftConstraints)
    }

    @Test
    fun sameExpressionSoftConstraintsSatTest() = with(ctx) {
        val x by boolSort
        val y by boolSort

        maxSMTSolver.assert(x or y)

        maxSMTSolver.assertSoft(!x or y, 7u)
        maxSMTSolver.assertSoft(x or !y, 6u)
        maxSMTSolver.assertSoft(!x or !y, 3u)
        maxSMTSolver.assertSoft(!x or !y, 4u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 3)
        assertSoftConstraintsSat(
            listOf(
                SoftConstraint(!x or y, 7u),
                SoftConstraint(!x or !y, 3u),
                SoftConstraint(!x or !y, 4u),
            ),
            maxSMTResult.satSoftConstraints,
        )
    }

    @Test
    fun sameExpressionSoftConstraintsUnsatTest() = with(ctx) {
        val x by boolSort
        val y by boolSort

        maxSMTSolver.assert(x or y)

        maxSMTSolver.assertSoft(!x or y, 6u)
        maxSMTSolver.assertSoft(x or !y, 6u)
        maxSMTSolver.assertSoft(!x or !y, 3u)
        maxSMTSolver.assertSoft(!x or !y, 2u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 2)
        assertSoftConstraintsSat(
            listOf(SoftConstraint(!x or y, 6u), SoftConstraint(x or !y, 6u)),
            maxSMTResult.satSoftConstraints,
        )
    }

    @Test
    fun chooseOneConstraintByWeightTest() = with(ctx) {
        val z by boolSort
        val a by boolSort
        val b by boolSort

        maxSMTSolver.assert(z)
        maxSMTSolver.assertSoft(a and b, 1u)
        maxSMTSolver.assertSoft(!a and !b, 5u)
        maxSMTSolver.assertSoft(a and b and z, 2u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 1)
        assertSoftConstraintsSat(listOf(SoftConstraint(!a and !b, 5u)), maxSMTResult.satSoftConstraints)
    }

    @Test
    fun equalWeightsTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSMTSolver.assert(a or b)
        maxSMTSolver.assert(!a or b)

        maxSMTSolver.assertSoft(a or !b, 2u)
        maxSMTSolver.assertSoft(!a or !b, 2u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.satSoftConstraints.size == 1)
    }

    @Test
    fun similarExpressionsTest(): Unit = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSMTSolver.assertSoft(a or b, 1u)
        maxSMTSolver.assertSoft(!a or b, 1u)
        maxSMTSolver.assertSoft(a or !b, 1u)
        maxSMTSolver.assertSoft(!a or !b, 1u)
        maxSMTSolver.assertSoft(!a or !b or !b, 1u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.satSoftConstraints.size == 4)
        assertTrue(maxSMTResult.satSoftConstraints.any { it.expression == !a or !b })
        assertTrue(maxSMTResult.satSoftConstraints.any { it.expression == !a or !b or !b })
    }

    @Test
    fun inequalitiesTest() = with(ctx) {
        val x by intSort
        val y by intSort

        val a1 = x gt 0.expr
        val a2 = x lt y
        val a3 = x + y le 0.expr

        maxSMTSolver.assert(a3 eq a1)
        maxSMTSolver.assert(a3 or a2)

        maxSMTSolver.assertSoft(a3, 3u)
        maxSMTSolver.assertSoft(!a3, 5u)
        maxSMTSolver.assertSoft(!a1, 10u)
        maxSMTSolver.assertSoft(!a2, 3u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResult.satSoftConstraints.size == 2)
        assertSoftConstraintsSat(
            listOf(SoftConstraint(!a3, 5u), SoftConstraint(!a1, 10u)),
            maxSMTResult.satSoftConstraints,
        )
    }

    @Test
    fun threeExpressionsAreInconsistentTest() = with(ctx) {
        val x by boolSort
        val y by boolSort

        maxSMTSolver.assertSoft(x, 671u)
        maxSMTSolver.assertSoft(y, 783u)
        maxSMTSolver.assertSoft(!x and !y or !x or !y, 859u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertSoftConstraintsSat(
            listOf(SoftConstraint(y, 783u), SoftConstraint(!x and !y or !x or !y, 859u)),
            maxSMTResult.satSoftConstraints,
        )
    }

    @Test
    fun oneScopePushPopTest() = with(ctx) {
        val a by boolSort
        val b by boolSort

        maxSMTSolver.assert(a or b)

        maxSMTSolver.assertSoft(!a or b, 1u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()
        assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1u)), maxSMTResult.satSoftConstraints)

        maxSMTSolver.push()

        maxSMTSolver.assertSoft(a or !b, 1u)
        maxSMTSolver.assertSoft(!a or !b, 1u)

        val maxSMTResultScoped = maxSMTSolver.checkMaxSMT()

        assertTrue(maxSMTResult.hardConstraintsSatStatus == SAT)
        assertTrue(maxSMTResult.maxSMTSucceeded)
        assertTrue(maxSMTResultScoped.satSoftConstraints.size == 2)

        maxSMTSolver.pop()

        assertSoftConstraintsSat(listOf(SoftConstraint(!a or b, 1u)), maxSMTResult.satSoftConstraints)
    }

    @Test
    fun threeScopesPushPopTest() = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort

        maxSMTSolver.assert(a)

        maxSMTSolver.push()
        maxSMTSolver.assertSoft(a or b, 1u)
        maxSMTSolver.assertSoft(c or b, 1u)

        val maxSMTResult = maxSMTSolver.checkMaxSMT()
        assertTrue(maxSMTResult.satSoftConstraints.size == 2)

        maxSMTSolver.push()
        maxSMTSolver.assertSoft(!b and !c, 2u)
        val maxSMTResult2 = maxSMTSolver.checkMaxSMT()
        assertTrue(maxSMTResult2.satSoftConstraints.size == 2)

        maxSMTSolver.push()
        maxSMTSolver.assertSoft(a or c, 1u)
        val maxSMTResult3 = maxSMTSolver.checkMaxSMT()
        assertTrue(maxSMTResult3.satSoftConstraints.size == 3)

        maxSMTSolver.pop(2u)
        val maxSMTResult4 = maxSMTSolver.checkMaxSMT()
        assertTrue(maxSMTResult4.satSoftConstraints.size == 2)

        maxSMTSolver.pop()
        val maxSMTResult5 = maxSMTSolver.checkMaxSMT()
        assertTrue(maxSMTResult5.satSoftConstraints.isEmpty())
    }

    private fun assertSoftConstraintsSat(
        constraintsToAssert: List<SoftConstraint>,
        satConstraints: List<SoftConstraint>,
    ) {
        for (constraint in constraintsToAssert) {
            assertTrue(
                satConstraints.any {
                    constraint.expression == it.expression && constraint.weight == it.weight
                },
            )
        }
    }
}
