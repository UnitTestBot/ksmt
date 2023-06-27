package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndNaryExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrNaryExpr
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.utils.getValue
import io.ksmt.utils.mkConst
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class KMaxSATSolverTest {
    @Test
    fun noSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort
        val c by boolSort
        maxSATSolver.assert(a)
        maxSATSolver.assert(b)
        maxSATSolver.assert(c)
        maxSATSolver.assertSoft(mkAnd(a, mkNot(c)), 3)
        maxSATSolver.assertSoft(mkNot(a), 5)

        val (status, satSoftConstraints) = maxSATSolver.checkMaxSMT()
        assertTrue(satSoftConstraints.isEmpty())
    }

    @Test
    fun oneOfTwoSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        val notA = KNotExpr(this, a)
        val notB =  KNotExpr(this, b)

        maxSATSolver.assert(KOrBinaryExpr(this, a, b))
        maxSATSolver.assert(KOrBinaryExpr(this, notA, b))
        maxSATSolver.assertSoft(KOrBinaryExpr(this, a, notB), 2)

        val notAOrNotBExpr = KOrBinaryExpr(this, notA, notB)
        maxSATSolver.assertSoft(notAOrNotBExpr, 3)

        val (status, satSoftConstraints) = maxSATSolver.checkMaxSMT()
        assertTrue(satSoftConstraints.size == 1 && satSoftConstraints[0].weight == 3 &&
                satSoftConstraints[0].constraint == notAOrNotBExpr)
    }

    @Test
    fun twoOfThreeSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        val notA = KNotExpr(this, a)
        val notB =  KNotExpr(this, b)

        maxSATSolver.assert(KOrBinaryExpr(this, a, b))

        val notAOrBExpr = KOrBinaryExpr(this, notA, b)
        maxSATSolver.assertSoft(notAOrBExpr, 4)

        val aOrNotBExpr = KOrBinaryExpr(this, a, notB)
        maxSATSolver.assertSoft(aOrNotBExpr, 6)

        maxSATSolver.assertSoft(KOrBinaryExpr(this, notA, notB), 2)

        val (status, satSoftConstraints) = maxSATSolver.checkMaxSMT()

        assertTrue(satSoftConstraints.size == 2)
    }

    @Test
    fun smokeTest4() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val a by boolSort
        val b by boolSort

        val notA = KNotExpr(this, a)
        val notB =  KNotExpr(this, b)

        maxSATSolver.assert(KOrBinaryExpr(this, a, b))
        maxSATSolver.assertSoft(KOrNaryExpr(this, listOf(notA, notA, b)), 1)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notA, b), 1)
        maxSATSolver.assertSoft(KOrNaryExpr(this, listOf(notA, notA, b)), 1)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, a, notB), 1)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notA,notB), 1)
        maxSATSolver.assertSoft(KOrNaryExpr(this, listOf(notA, notA, notB)), 1)

        val (status, satSoftConstraints) = maxSATSolver.checkMaxSMT()
    }

    @Test
    fun smokeTest5() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)
        val z = boolSort.mkConst("z")
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")
        val c = boolSort.mkConst("c")

        maxSATSolver.assert(z)
        maxSATSolver.assertSoft(KAndBinaryExpr(this, a, b), 1)
        val constr = KAndBinaryExpr(this, KNotExpr(this, a), KNotExpr(this, b))
        maxSATSolver.assertSoft(constr, 5)
        maxSATSolver.assertSoft(KAndNaryExpr(this, listOf(a, b, z)), 2)

        val (status, satSoftConstraints) = maxSATSolver.checkMaxSMT()

        assertTrue(satSoftConstraints.size == 1 && satSoftConstraints[0].weight == 5 &&
                satSoftConstraints[0].constraint == constr)
    }

    @Test
    fun sameExpressionSoftConstraintsSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        val notX = KNotExpr(this, x)
        val notY =  KNotExpr(this, y)

        maxSATSolver.assert(KOrBinaryExpr(this, x , y))
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, y), 6)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, x, notY), 6)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, notY), 3)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, notY), 4)

        val (status, satSoftConstraints) = maxSATSolver.checkMaxSMT()

        satSoftConstraints.forEach { println("constr: ${it.constraint};  weight: ${it.weight}") }
    }

    @Test
    fun sameExpressionSoftConstraintsUNSATTest() = with (KContext()) {
        val z3Solver = KZ3Solver(this)
        val maxSATSolver = KMaxSATSolver(this, z3Solver)

        val x by boolSort
        val y by boolSort

        val notX = KNotExpr(this, x)
        val notY =  KNotExpr(this, y)

        maxSATSolver.assert(KOrBinaryExpr(this, x , y))
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, y), 6)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, x, notY), 6)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, notY), 3)
        maxSATSolver.assertSoft(KOrBinaryExpr(this, notX, notY), 2)

        val (status, satSoftConstraints) = maxSATSolver.checkMaxSMT()

        satSoftConstraints.forEach { println("constr: ${it.constraint};  weight: ${it.weight}") }
    }
}
