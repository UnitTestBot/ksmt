package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KIntNumDecl
import io.ksmt.expr.*
import io.ksmt.utils.mkConst
import kotlin.math.sign
import kotlin.test.BeforeTest
import kotlin.test.Test

class ExpressionOrderingTest {
    private val ctx = KContext()
    private val values = arrayOf(
        ctx.mkConstApp(ctx.mkIntNumDecl("1")),
        ctx.mkConstApp(ctx.mkIntNumDecl("2")),
        ctx.mkIntNum(3),
        ctx.mkIntNum(4),
        ctx.mkArithAddNoSimplify(listOf(ctx.mkIntNum(10), ctx.mkIntNum(20))),
        ctx.mkArithAddNoSimplify(listOf(ctx.mkIntNum(30), ctx.mkIntNum(40))),
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
