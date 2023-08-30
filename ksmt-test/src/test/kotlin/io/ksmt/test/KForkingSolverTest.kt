package io.ksmt.test

import io.ksmt.KContext
import io.ksmt.solver.KForkingSolverManager
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.bitwuzla.KBitwuzlaForkingSolverManager
import io.ksmt.solver.cvc5.KCvc5ForkingSolverManager
import io.ksmt.solver.yices.KYicesForkingSolverManager
import io.ksmt.solver.z3.KZ3ForkingSolverManager
import io.ksmt.utils.getValue
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class KForkingSolverTest {
    @Nested
    inner class KForkingSolverTestBitwuzla {
        @Test
        fun testCheckSat() = testCheckSat(::mkBitwuzlaForkingSolverManager)

        @Test
        fun testModel() = testModel(::mkBitwuzlaForkingSolverManager)

        @Test
        fun testUnsatCore() = testUnsatCore(::mkBitwuzlaForkingSolverManager)

        @Test
        fun testUninterpretedSort() = testUninterpretedSort(::mkBitwuzlaForkingSolverManager)

        @Test
        fun testScopedAssertions() = testScopedAssertions(::mkBitwuzlaForkingSolverManager)

        @Test
        fun testLifeTime() = testLifeTime(::mkBitwuzlaForkingSolverManager)

        private fun mkBitwuzlaForkingSolverManager(ctx: KContext) = KBitwuzlaForkingSolverManager(ctx)
    }

    @Nested
    inner class KForkingSolverTestCvc5 {
        @Test
        fun testCheckSat() = testCheckSat(::mkCvc5ForkingSolverManager)

        @Test
        fun testModel() = testModel(::mkCvc5ForkingSolverManager)

        @Test
        fun testUnsatCore() = testUnsatCore(::mkCvc5ForkingSolverManager)

        @Test
        fun testUninterpretedSort() = testUninterpretedSort(::mkCvc5ForkingSolverManager)

        @Test
        fun testScopedAssertions() = testScopedAssertions(::mkCvc5ForkingSolverManager)

        @Test
        fun testLifeTime() = testLifeTime(::mkCvc5ForkingSolverManager)

        private fun mkCvc5ForkingSolverManager(ctx: KContext) = KCvc5ForkingSolverManager(ctx)
    }

    @Nested
    inner class KForkingSolverTestYices {
        @Test
        fun testCheckSat() = testCheckSat(::mkYicesForkingSolverManager)

        @Test
        fun testModel() = testModel(::mkYicesForkingSolverManager)

        @Test
        fun testUnsatCore() = testUnsatCore(::mkYicesForkingSolverManager)

        @Test
        fun testUninterpretedSort() = testUninterpretedSort(::mkYicesForkingSolverManager)

        @Test
        fun testScopedAssertions() = testScopedAssertions(::mkYicesForkingSolverManager)

        @Test
        fun testLifeTime() = testLifeTime(::mkYicesForkingSolverManager)

        private fun mkYicesForkingSolverManager(ctx: KContext) = KYicesForkingSolverManager(ctx)
    }

    @Nested
    inner class KForkingSolverTestZ3 {
        @Test
        fun testCheckSat() = testCheckSat(::mkZ3ForkingSolverManager)

        @Test
        fun testModel() = testModel(::mkZ3ForkingSolverManager)

        @Test
        fun testUnsatCore() = testUnsatCore(::mkZ3ForkingSolverManager)

        @Test
        fun testUninterpretedSort() = testUninterpretedSort(::mkZ3ForkingSolverManager)

        @Test
        fun testScopedAssertions() = testScopedAssertions(::mkZ3ForkingSolverManager)

        @Test
        fun testLifeTime() = testLifeTime(::mkZ3ForkingSolverManager)

        private fun mkZ3ForkingSolverManager(ctx: KContext) = KZ3ForkingSolverManager(ctx)
    }

    private fun testCheckSat(createForkingSolverManager: (KContext) -> KForkingSolverManager<*>) =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            createForkingSolverManager(ctx).use { man ->
                man.createForkingSolver().use { parentSolver ->
                    with(ctx) {
                        val a by boolSort
                        val b by boolSort
                        val f = a and b
                        val neg = !a

                        parentSolver.push()

                        // * check children's assertions do not change parent's state
                        parentSolver.assert(f)
                        require(parentSolver.check() == KSolverStatus.SAT)
                        require(parentSolver.checkWithAssumptions(listOf(neg)) == KSolverStatus.UNSAT)

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.SAT, fork.check())
                            fork.assert(neg)
                            assertEquals(KSolverStatus.UNSAT, fork.check())
                        }

                        assertEquals(KSolverStatus.SAT, parentSolver.check())
                        // *

                        // * check parent's assertions translated into child solver
                        parentSolver.push()
                        assertEquals(KSolverStatus.UNSAT, parentSolver.fork().checkWithAssumptions(listOf(neg)))
                        parentSolver.assert(neg)
                        require(parentSolver.check() == KSolverStatus.UNSAT)

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.UNSAT, fork.check())
                        }
                        parentSolver.pop()
                        // *

                        // * check children independence
                        assertEquals(KSolverStatus.SAT, parentSolver.check())
                        parentSolver.fork().also { fork1 ->
                            val fork2 = parentSolver.fork()
                            fork2.assert(neg)
                            assertEquals(KSolverStatus.UNSAT, fork2.check())
                            assertEquals(KSolverStatus.SAT, fork1.check())

                            fork1.assert(neg)
                            assertEquals(KSolverStatus.UNSAT, fork1.check())
                            assertEquals(KSolverStatus.SAT, parentSolver.fork().check())
                        }
                        assertEquals(KSolverStatus.SAT, parentSolver.check())
                        // *
                    }
                }
            }
        }

    private fun testUnsatCore(createForkingSolverManager: (KContext) -> KForkingSolverManager<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            createForkingSolverManager(ctx).use { man ->
                man.createForkingSolver().use { parentSolver ->
                    with(ctx) {
                        val a by boolSort
                        val b by boolSort
                        val f = a and b
                        val neg = !a

                        // * check that unsat core is empty (non-tracked assertions)
                        parentSolver.push()
                        parentSolver.assert(f)

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.SAT, fork.check())
                            fork.assert(neg)
                            assertEquals(KSolverStatus.UNSAT, fork.check())
                            assertTrue { fork.unsatCore().isEmpty() }
                            assertEquals(KSolverStatus.SAT, parentSolver.check()) // parent's state hasn't changed
                        }
                        parentSolver.pop()
                        // *

                        // check tracked exprs are in unsat core
                        parentSolver.push()
                        parentSolver.assertAndTrack(f)

                        parentSolver.fork().also { fork ->
                            fork.assertAndTrack(neg)
                            assertEquals(KSolverStatus.UNSAT, fork.check())
                            assertContains(fork.unsatCore(), neg)
                            assertContains(fork.unsatCore(), f)
                            assertEquals(KSolverStatus.SAT, parentSolver.check()) // parent's state hasn't changed
                        }
                        // *

                        // * check unsat core saves from parent to child
                        parentSolver.assert(neg)
                        require(parentSolver.check() == KSolverStatus.UNSAT)
                        require(neg !in parentSolver.unsatCore())
                        require(f in parentSolver.unsatCore()) // only tracked f is in unsat core

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.UNSAT, fork.check())
                            assertContains(fork.unsatCore(), f)
                            assertTrue { neg !in fork.unsatCore() }
                        }
                    }
                }
            }
        }

    private fun testModel(createForkingSolverManager: (KContext) -> KForkingSolverManager<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            createForkingSolverManager(ctx).use { man ->
                man.createForkingSolver().use { parentSolver ->
                    with(ctx) {
                        val a by boolSort
                        val b by boolSort
                        val f = a and !b

                        parentSolver.assert(f)

                        require(parentSolver.check() == KSolverStatus.SAT)
                        require(parentSolver.model().eval(a) == true.expr)
                        require(parentSolver.model().eval(b) == false.expr)

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.SAT, fork.check())
                            assertEquals(true.expr, fork.model().eval(a))
                            assertEquals(false.expr, fork.model().eval(b))
                        }
                    }
                }
            }
        }

    private fun testScopedAssertions(createForkingSolverManager: (KContext) -> KForkingSolverManager<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            createForkingSolverManager(ctx).use { man ->
                man.createForkingSolver().use { parent ->
                    with(ctx) {
                        val a by boolSort
                        val b by boolSort
                        val f = a and b
                        val neg = !a

                        parent.push()

                        parent.assertAndTrack(f)
                        require(parent.check() == KSolverStatus.SAT)
                        parent.push()
                        parent.assertAndTrack(neg)

                        require(parent.check() == KSolverStatus.UNSAT)

                        parent.fork().also { fork ->
                            assertEquals(KSolverStatus.UNSAT, fork.check())
                            assertContains(fork.unsatCore(), f)
                            assertContains(fork.unsatCore(), neg)

                            fork.pop()
                            assertEquals(KSolverStatus.SAT, fork.check())
                            assertEquals(true.expr, fork.model().eval(a))
                            assertEquals(true.expr, fork.model().eval(b))
                            assertEquals(KSolverStatus.UNSAT, fork.checkWithAssumptions(listOf(neg)))
                            assertEquals(KSolverStatus.UNSAT, parent.check()) // check parent's state hasn't changed

                            fork.fork().also { ffork ->
                                assertEquals(KSolverStatus.SAT, ffork.check())
                                assertEquals(KSolverStatus.UNSAT, ffork.checkWithAssumptions(listOf(neg)))

                                ffork.push()
                                ffork.assertAndTrack(neg)
                                assertEquals(KSolverStatus.UNSAT, ffork.check())
                                assertContains(ffork.unsatCore(), f)
                                assertContains(ffork.unsatCore(), neg)

                                assertEquals(KSolverStatus.SAT, fork.check()) // check parent's state hasn't changed
                                assertEquals(KSolverStatus.UNSAT, parent.check()) // check parent's state hasn't changed

                                ffork.pop()
                                assertEquals(KSolverStatus.SAT, ffork.check())
                                assertEquals(KSolverStatus.UNSAT, ffork.checkWithAssumptions(listOf(neg)))
                            }
                        }

                        // check child's state is detached
                        val fork = parent.fork()
                        assertEquals(KSolverStatus.UNSAT, fork.check())
                        parent.pop()

                        assertEquals(KSolverStatus.SAT, parent.check())
                        assertEquals(KSolverStatus.UNSAT, fork.check())

                        parent.pop()

                        fork.pop()
                        fork.pop()

                        fork.assert(neg)
                        assertEquals(KSolverStatus.SAT, fork.check())
                    }
                }
            }
        }

    @Suppress("LongMethod")
    private fun testUninterpretedSort(createForkingSolverManager: (KContext) -> KForkingSolverManager<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            createForkingSolverManager(ctx).use { man ->
                man.createForkingSolver().use { parentSolver ->
                    with(ctx) {
                        val uSort = mkUninterpretedSort("u")
                        val u1 by uSort
                        val u2 by uSort

                        val eq12 = u1 eq u2

                        parentSolver.push()

                        parentSolver.fork().also { fork ->
                            assertDoesNotThrow { fork.pop() } // check assertion levels saved
                            fork.assert(u1 neq u2)
                            assertEquals(KSolverStatus.SAT, fork.check())
                        }

                        parentSolver.assert(eq12)

                        require(parentSolver.check() == KSolverStatus.SAT)
                        val pu1v = parentSolver.model().eval(u1)

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.SAT, fork.check())
                            fork.assert(u1 eq pu1v)
                            assertEquals(KSolverStatus.SAT, fork.check())
                            assertEquals(pu1v, fork.model().eval(u1))
                        }

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.SAT, fork.check())
                            fork.assert(u1 neq pu1v)
                            assertEquals(KSolverStatus.SAT, fork.check())
                            assertNotEquals(pu1v, fork.model().eval(u1))
                        }

                        parentSolver.push().also {
                            val u5 by uSort
                            val pu5v = mkUninterpretedSortValue(uSort, 5)
                            parentSolver.assert(u5 eq pu5v)

                            parentSolver.assert(u1 eq pu1v)

                            parentSolver.check()
                            parentSolver.model().uninterpretedSortUniverse(uSort)?.also { universe ->
                                assertContains(universe, pu1v)
                                assertContains(universe, pu5v)
                            }

                            parentSolver.pop()
                        }

                        parentSolver.fork().also { fork ->
                            assertEquals(KSolverStatus.SAT, fork.check())
                            fork.assert(u1 eq pu1v)
                            assertEquals(KSolverStatus.SAT, fork.check())
                            assertEquals(pu1v, fork.model().eval(u1))

                            fork.fork().also { ff ->
                                assertEquals(KSolverStatus.SAT, ff.check())
                                assertEquals(pu1v, ff.model().eval(u1))
                                ff.model().uninterpretedSortUniverse(uSort)?.also { universe ->
                                    assertContains(universe, pu1v)
                                }
                            }
                        }

                        parentSolver.assert(u1 neq pu1v)
                        assertEquals(KSolverStatus.SAT, parentSolver.check())
                        parentSolver.model().uninterpretedSortUniverse(uSort)?.also { universe ->
                            assertContains(universe, pu1v)
                        }

                    }
                }
            }
        }

    fun testLifeTime(createForkingSolverManager: (KContext) -> KForkingSolverManager<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            createForkingSolverManager(ctx).use { man ->
                with(ctx) {
                    val parent = man.createForkingSolver()
                    val x by bv8Sort
                    val f = mkBvSignedGreaterExpr(x, mkBv(100, bv8Sort))

                    parent.assert(f)
                    parent.check().also { require(it == KSolverStatus.SAT) }

                    val xVal = parent.model().eval(x)

                    val fork = parent.fork().fork().fork()
                    parent.close()

                    fork.assert(f and (x eq xVal))
                    fork.check().also { assertEquals(KSolverStatus.SAT, it) }
                    assertEquals(xVal, fork.model().eval(x))
                }
            }
        }
}
