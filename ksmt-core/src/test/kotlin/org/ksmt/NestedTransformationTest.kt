package org.ksmt

import org.ksmt.expr.KExpr
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort
import org.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.assertEquals

class NestedTransformationTest {

    @Test
    fun nestedTransformationTest(): Unit = with(KContext()) {
        val a by intSort
        val b by intSort

        val transformer = TestAuxTransformer(this)

        val e0 = TestAuxExpr(this, TestAuxExpr(this, a) + TestAuxExpr(this, b))
        val e1 = TestAuxExpr(this, e0 * TestAuxExpr(this, a))
        val expr = TestAuxExpr(this, e1 / TestAuxExpr(this, b))
        val actual = transformer.apply(expr)

        val expected = transformer.apply(((a + b) * a) / b)

        assertEquals(expected, actual)
    }

    class TestAuxTransformer(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        fun <T : KSort> transformAux(expr: TestAuxExpr<T>): KExpr<T> {
            // nested transformation
            return apply(expr.nested)
        }
    }

    class TestAuxExpr<T : KSort>(ctx: KContext, val nested: KExpr<T>) : KExpr<T>(ctx) {
        override val sort: T
            get() = nested.sort

        override fun accept(transformer: KTransformerBase): KExpr<T> {
            transformer as TestAuxTransformer
            return transformer.transformAux(this)
        }

        override fun print(printer: ExpressionPrinter) {
            printer.append(nested)
        }

        override fun internEquals(other: Any): Boolean = error("Interning is not used")
        override fun internHashCode(): Int = error("Interning is not used")
    }
}
