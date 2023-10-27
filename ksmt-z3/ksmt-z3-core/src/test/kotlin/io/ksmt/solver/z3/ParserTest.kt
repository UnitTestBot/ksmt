package io.ksmt.solver.z3

import io.ksmt.KContext
import kotlin.test.Test
import kotlin.test.assertEquals

class ParserTest {

    @Test
    fun parseSimple(): Unit = with(KContext()) {
        val assertions = KZ3SMTLibParser(this).parse(sampleSimple)
        assertEquals(assertions.size, 2)
    }

    @Test
    fun parsePushPop(): Unit = with(KContext()) {
        val assertions = KZ3SMTLibParser(this).parse(samplePushPop)
        assertEquals(assertions.size, 1)
    }

    companion object {
        private val sampleSimple = """
            (declare-fun x () Int)
            (declare-fun y () Int)
            (assert (>= x y))
            (assert (>= y x))
        """.trimIndent()

        private val samplePushPop = """
            (declare-fun x () Int)
            (declare-fun y () Int)
            (assert (>= x y))
            (push)
            (assert (>= y x))
            (pop)
        """.trimIndent()
    }
}
