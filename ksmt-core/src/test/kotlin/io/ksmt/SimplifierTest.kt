package io.ksmt

import io.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.rewrite.simplify.KExprSimplifier
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KSort
import io.ksmt.utils.BvUtils.shiftLeft
import io.ksmt.utils.BvUtils.shiftRightArith
import io.ksmt.utils.BvUtils.shiftRightLogical
import io.ksmt.utils.getValue
import io.ksmt.utils.mkConst
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SimplifierTest {

    @Test
    fun testNestedRewrite(): Unit = with(KContext()) {
        val a by boolSort
        val expr = mkEq(falseExpr, mkNot(a))
        val result = KExprSimplifier(this).apply(expr)
        assertEquals(a, result)

        val arraySort = mkArraySort(intSort, boolSort)
        val expr1 = mkArrayConst(arraySort, falseExpr) eq mkArrayConst(arraySort, mkNot(a))
        val result1 = KExprSimplifier(this).apply(expr1)
        assertEquals(a, result1)
    }

    @Test
    fun testArrayEqSimplification(): Unit = with(KContext()) {
        val array by mkArraySort(intSort, intSort)
        val specialIdx by intSort
        val indices1 = (0..3).map { it.expr } + listOf(specialIdx) + (7..10).map { it.expr }
        val indices2 = (0..1).map { it.expr } + listOf(specialIdx) + ((2..3) + (7..10)).map { it.expr }
        val array1 = indices1.fold(array as KExpr<KArraySort<KIntSort, KIntSort>>) { a, idx -> a.store(idx, idx) }
        val array2 = indices2.fold(array as KExpr<KArraySort<KIntSort, KIntSort>>) { a, idx -> a.store(idx, idx) }
        val arrayEq = array1 eq array2
        val allSelectsEq = mkAnd((indices1 + indices2).map {
            array1.select(it) eq array2.select(it)
        })

        val simplifiedEq = KExprSimplifier(this).apply(arrayEq)
        val simplifiedSelects = KExprSimplifier(this).apply(allSelectsEq)

        // Compare sets of Eq expressions
        val simplifiedEqParts = (simplifiedEq as KAndExpr).args.map { (it as KEqExpr<*>).args.toSet() }
        val simplifiedAllSelectsParts = (simplifiedSelects as KAndExpr).args.map { (it as KEqExpr<*>).args.toSet() }
        assertEquals(simplifiedAllSelectsParts.toSet(), simplifiedEqParts.toSet())
    }

    @Test
    fun testBvShl(): Unit = with(KContext()) {
        testBvShiftLeft(bv1Sort, 0x01u, 0x01u, 0x00u)

        testBvShiftLeft(bv8Sort, 0x8cu, 0x01u, 0x18u)
        testBvShiftLeft(bv8Sort, 0x8cu, 0xffu, 0x00u)
        testBvShiftLeft(bv8Sort, 0x8cu, 0xf1u, 0x00u)

        testBvShiftLeft(bv16Sort, 0x800cu, 0x0001u, 0x0018u)
        testBvShiftLeft(bv16Sort, 0x800cu, 0xffffu, 0x0000u)
        testBvShiftLeft(bv16Sort, 0x800cu, 0xff01u, 0x0000u)

        testBvShiftLeft(mkBvSort(24u), 0x80000cu, 0x000001u, 0x000018u)
        testBvShiftLeft(mkBvSort(24u), 0x80000cu, 0xffffffu, 0x000000u)
        testBvShiftLeft(mkBvSort(24u), 0x80000cu, 0xffff01u, 0x000000u)

        testBvShiftLeft(bv32Sort, 0x8000000cu, 0x00000001u, 0x00000018u)
        testBvShiftLeft(bv32Sort, 0x8000000cu, 0xffffffffu, 0x00000000u)
        testBvShiftLeft(bv32Sort, 0x8000000cu, 0xffffff0fu, 0x00000000u)
    }

    @Test
    fun testBvLshr() = with(KContext()) {
        testBvLshr(bv1Sort, 0x01u, 0x01u, 0x00u)

        testBvLshr(bv8Sort, 0x8cu, 0x01u, 0x46u)
        testBvLshr(bv8Sort, 0x8cu, 0xffu, 0x00u)
        testBvLshr(bv8Sort, 0x8cu, 0xf1u, 0x00u)

        testBvLshr(bv16Sort, 0x800cu, 0x0001u, 0x4006u)
        testBvLshr(bv16Sort, 0x800cu, 0xffffu, 0x0000u)
        testBvLshr(bv16Sort, 0x800cu, 0xff01u, 0x0000u)

        testBvLshr(mkBvSort(24u), 0x80000cu, 0x000001u, 0x400006u)
        testBvLshr(mkBvSort(24u), 0x80000cu, 0xffffffu, 0x000000u)
        testBvLshr(mkBvSort(24u), 0x80000cu, 0xffff01u, 0x000000u)

        testBvLshr(bv32Sort, 0x8000000cu, 0x00000001u, 0x40000006u)
        testBvLshr(bv32Sort, 0x8000000cu, 0xffffffffu, 0x00000000u)
        testBvLshr(bv32Sort, 0x8000000cu, 0xffffff0fu, 0x00000000u)
    }

    @Test
    fun testBvAshr() = with(KContext()) {
        testBvAshr(bv1Sort, 0x01u, 0x01u, 0x01u)

        testBvAshr(bv8Sort, 0x8cu, 0x01u, 0xc6u)
        testBvAshr(bv8Sort, 0x8cu, 0xffu, 0xffu)
        testBvAshr(bv8Sort, 0x8cu, 0xf1u, 0xffu)
        testBvAshr(bv8Sort, 0x0cu, 0x01u, 0x06u)
        testBvAshr(bv8Sort, 0x0cu, 0xf1u, 0x00u)

        testBvAshr(bv16Sort, 0x800cu, 0x0001u, 0xc006u)
        testBvAshr(bv16Sort, 0x800cu, 0xffffu, 0xffffu)
        testBvAshr(bv16Sort, 0x800cu, 0xff01u, 0xffffu)
        testBvAshr(bv16Sort, 0x000cu, 0x0001u, 0x0006u)
        testBvAshr(bv16Sort, 0x000cu, 0xff01u, 0x0000u)

        testBvAshr(mkBvSort(24u), 0x80000cu, 0x000001u, 0xc00006u)
        testBvAshr(mkBvSort(24u), 0x80000cu, 0xffffffu, 0xffffffu)
        testBvAshr(mkBvSort(24u), 0x80000cu, 0xffff01u, 0xffffffu)
        testBvAshr(mkBvSort(24u), 0x00000cu, 0x000001u, 0x000006u)
        testBvAshr(mkBvSort(24u), 0x00000cu, 0xffff01u, 0x000000u)

        testBvAshr(bv32Sort, 0x8000000cu, 0x00000001u, 0xc0000006u)
        testBvAshr(bv32Sort, 0x8000000cu, 0xffffffffu, 0xffffffffu)
        testBvAshr(bv32Sort, 0x8000000cu, 0xffffff0fu, 0xffffffffu)
        testBvAshr(bv32Sort, 0x0000000cu, 0x00000001u, 0x00000006u)
        testBvAshr(bv32Sort, 0x0000000cu, 0xffffff01u, 0x00000000u)


        val bigSignedArg = mkBv("1".repeat(127) + "000", 130u)
        val shiftOutOfBounds = mkBv("1".repeat(130), 130u)
        val shiftByOne = mkBv("0".repeat(129) + "1", 130u)

        assertEquals(mkBv("1".repeat(130), 130u), bigSignedArg.shiftRightArith(shiftOutOfBounds))
        assertEquals(mkBv("1".repeat(128) + "00", 130u), bigSignedArg.shiftRightArith(shiftByOne))
    }

    private fun testBvShiftLeft(
        sort: KBvSort,
        arg: UInt,
        shift: UInt,
        expected: UInt
    ) = testBvShift(sort, arg, shift, expected) { a, s -> a.shiftLeft(s) }

    private fun testBvLshr(
        sort: KBvSort,
        arg: UInt,
        shift: UInt,
        expected: UInt
    ) = testBvShift(sort, arg, shift, expected) { a, s -> a.shiftRightLogical(s) }

    private fun testBvAshr(
        sort: KBvSort,
        arg: UInt,
        shift: UInt,
        expected: UInt
    ) = testBvShift(sort, arg, shift, expected) { a, s -> a.shiftRightArith(s) }

    private fun testBvShift(
        sort: KBvSort,
        arg: UInt,
        shift: UInt,
        expected: UInt,
        op: (KBitVecValue<*>, KBitVecValue<*>) -> KBitVecValue<*>
    ) = with(sort.ctx) {
        val expectedBv = mkBv(expected.toInt(), sort)
        val actual = op(mkBv(arg.toInt(), sort), mkBv(shift.toInt(), sort))
        assertEquals(expectedBv, actual)
    }

    private fun <T : KSort> varGenerator(
        sort: T,
        usedVars: MutableList<KExpr<T>>
    ) = sequence<KExpr<T>> {
        var idx = 0
        while (true) {
            val const = sort.mkConst("v${idx++}")
            usedVars.add(const)
            yield(const)
        }
    }

    @Test
    fun testStagedAndSimplification(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        val usedVars = mutableListOf<KExpr<KBoolSort>>()
        val vars = varGenerator(boolSort, usedVars).iterator()
        val expr = mkAnd(
            vars.next(),
            vars.next(),
            mkAnd(
                mkAnd(
                    vars.next(),
                    vars.next()
                ),
                vars.next(),
            )
        )
        val simplifiedExpr = KExprSimplifier(this).apply(expr)

        assertTrue(simplifiedExpr is KAndExpr)
        assertEquals(usedVars.toSet(), simplifiedExpr.args.toSet())
    }

}
