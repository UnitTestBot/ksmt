package io.ksmt.test

import io.ksmt.KContext
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.cvc5.KCvc5ForkingSolverManager
import io.ksmt.utils.getValue
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test

class KForkingSolverTest {
    @Nested
    inner class KForkingSolverTestCvc5 {
        @Test
        fun testCheckSat() = testCheckSat(::mkCvc5ForkingSolver)

        @Test
        fun testModel() = testModel(::mkCvc5ForkingSolver)

        @Test
        fun testUnsatCore() = testUnsatCore(::mkCvc5ForkingSolver)

        @Test
        fun testUninterpretedSort() = testUninterpretedSort(::mkCvc5ForkingSolver)

        @Test
        fun testScopedAssertions() = testScopedAssertions(::mkCvc5ForkingSolver)

        private fun mkCvc5ForkingSolver(ctx: KContext) = KCvc5ForkingSolverManager(ctx).mkForkingSolver()
    }

    private fun testCheckSat(mkSolver: (KContext) -> KForkingSolver<*>) =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            mkSolver(ctx).use { parentSolver ->
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

    private fun testUnsatCore(mkSolver: (KContext) -> KForkingSolver<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            mkSolver(ctx).use { parentSolver ->
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

    private fun testModel(mkSolver: (KContext) -> KForkingSolver<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            mkSolver(ctx).use { parentSolver ->
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

    private fun testScopedAssertions(mkSolver: (KContext) -> KForkingSolver<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            mkSolver(ctx).use { parent ->
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

    private fun testUninterpretedSort(mkSolver: (KContext) -> KForkingSolver<*>): Unit =
        KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY).use { ctx ->
            mkSolver(ctx).use { parentSolver ->
                with(ctx) {
                    val uSort = mkUninterpretedSort("u")
                    val u1 by uSort
                    val u2 by uSort

                    val eq12 = u1 eq u2

                    parentSolver.push()
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

                    parentSolver.model().uninterpretedSortUniverse(uSort)?.also { universe ->
                        assertContains(universe, pu1v)
                    }

                }
            }
        }
}
