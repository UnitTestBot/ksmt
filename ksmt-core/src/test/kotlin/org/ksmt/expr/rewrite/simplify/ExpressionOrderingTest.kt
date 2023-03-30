package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.decl.KIntNumDecl
import org.ksmt.expr.*
import kotlin.math.sign
import kotlin.test.Test

class ExpressionOrderingTest {
    private val ctx = KContext()
    private val values = arrayOf(
        KConst(ctx, KIntNumDecl(ctx, "1")),
        KConst(ctx, KIntNumDecl(ctx, "2")),
        KInt32NumExpr(ctx, 3),
        KInt32NumExpr(ctx, 4),
        KAddArithExpr(ctx, listOf(KInt32NumExpr(ctx, 10), KInt32NumExpr(ctx, 20))),
        KAddArithExpr(ctx, listOf(KInt32NumExpr(ctx, 30), KInt32NumExpr(ctx, 40))),
    )

    @Test
    fun testAntisymmetry() = with(ExpressionOrdering) {
        for (value1 in values) {
            for (value2 in values) {
                assert(compare(value1, value2).sign == -compare(value2, value1).sign) {
                    "Compare $value1 with $value2"
                }
            }
        }
    }

    @Test
    fun testReflexivity() {
        for (value in values) {
            assert(ExpressionOrdering.compare(value, value) == 0) {
                "Compare $value with itself"
            }
        }
    }

    @Test
    fun testTransitivity() = with(ExpressionOrdering) {
        for (value1 in values) {
            for (value2 in values) {
                for (value3 in values) {
                    if (compare(value1, value2) <= 0 && compare(value2, value3) <= 0) {
                        assert(compare(value1, value3) <= 0) {
                            "Compare $value1 and $value3 when [$value1] <= [$value2] <= [$value3]"
                        }
                    }
                }
            }
        }
    }
}
