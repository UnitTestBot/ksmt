package io.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort
import io.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.assertEquals

class ArrayTest {

    @Test
    fun testArrayConversion(): Unit = with(KContext()) {
        val sort = mkArraySort(bv32Sort, bv64Sort)

        val array by sort
        val index by bv32Sort
        val value by bv64Sort
        val decl = mkFuncDecl("F", bv64Sort, listOf(bv32Sort))

        testArrayConvert(
            mkConst = { mkArrayConst(sort, value) },
            mkSelect = { mkArraySelect(array, index) },
            mkStore = { mkArrayStore(array, index, value) },
            mkLambda = { mkArrayLambda(index.decl, mkArraySelect(array, index)) },
            mkAsArray = { mkFunctionAsArray(sort, decl) }
        )
    }

    @Test
    fun testArray2Conversion(): Unit = with(KContext()) {
        val sort = mkArraySort(bv32Sort, bv8Sort, bv64Sort)

        val array by sort
        val index0 by bv32Sort
        val index1 by bv8Sort
        val value by bv64Sort
        val decl = mkFuncDecl("F", bv64Sort, listOf(bv32Sort, bv8Sort))

        testArrayConvert(
            mkConst = { mkArrayConst(sort, value) },
            mkSelect = { mkArraySelect(array, index0, index1) },
            mkStore = { mkArrayStore(array, index0, index1, value) },
            mkLambda = { mkArrayLambda(index0.decl, index1.decl, mkArraySelect(array, index0, index1)) },
            mkAsArray = { mkFunctionAsArray(sort, decl) }
        )
    }

    @Test
    fun testArray3Conversion(): Unit = with(KContext()) {
        val sort = mkArraySort(bv32Sort, bv8Sort, bv16Sort, bv64Sort)

        val array by sort
        val index0 by bv32Sort
        val index1 by bv8Sort
        val index2 by bv16Sort
        val value by bv64Sort
        val decl = mkFuncDecl("F", bv64Sort, listOf(bv32Sort, bv8Sort, bv16Sort))

        testArrayConvert(
            mkConst = { mkArrayConst(sort, value) },
            mkSelect = { mkArraySelect(array, index0, index1, index2) },
            mkStore = { mkArrayStore(array, index0, index1, index2, value) },
            mkLambda = {
                mkArrayLambda(
                    index0.decl, index1.decl, index2.decl,
                    mkArraySelect(array, index0, index1, index2)
                )
            },
            mkAsArray = { mkFunctionAsArray(sort, decl) }
        )
    }

    @Test
    fun testArrayNConversion(): Unit = with(KContext()) {
        val domain = listOf(bv32Sort, bv8Sort, bv32Sort, bv8Sort, bv32Sort)
        val sort = mkArrayNSort(domain, bv64Sort)

        val array by sort
        val indices = domain.mapIndexed { i, s -> mkConst("x$i", s) }
        val value by bv64Sort
        val decl = mkFuncDecl("F", bv64Sort, domain)

        testArrayConvert(
            mkConst = { mkArrayConst(sort, value) },
            mkSelect = { mkArrayNSelect(array, indices) },
            mkStore = { mkArrayNStore(array, indices, value) },
            mkLambda = { mkArrayNLambda(indices.map { it.decl }, mkArrayNSelect(array, indices)) },
            mkAsArray = { mkFunctionAsArray(sort, decl) }
        )
    }

    private inline fun KContext.testArrayConvert(
        mkConst: KContext.() -> KExpr<*>,
        mkSelect: KContext.() -> KExpr<*>,
        mkStore: KContext.() -> KExpr<*>,
        mkLambda: KContext.() -> KExpr<*>,
        mkAsArray: KContext.() -> KExpr<*>
    ) {
        val const = mkConst()
        val select = mkSelect()
        val store = mkStore()
        val lambda = mkLambda()
        val asArray = mkAsArray()

        val ctx = this
        Context().use { z3Native ->
            val z3InternCtx = KZ3Context(this, z3Native)
            val z3ConvertCtx = KZ3Context(this, z3Native)

            with(KZ3ExprInternalizer(ctx, z3InternCtx)) {
                val iConst = const.internalizeExpr()

                // Simplify const since we use lambdas for n-ary array consts
                val iConstExpr = z3Native.wrapAST(iConst) as Expr<*>
                val iConstExprSimplified = iConstExpr.simplify()
                val iConstSimplified = z3Native.unwrapAST(iConstExprSimplified)

                val iSelect = select.internalizeExpr()
                val iStore = store.internalizeExpr()
                val iLambda = lambda.internalizeExpr()
                val iAsArray = asArray.internalizeExpr()

                with(KZ3ExprConverter(ctx, z3ConvertCtx)) {
                    val cConst = iConstSimplified.convertExpr<KSort>()
                    val cSelect = iSelect.convertExpr<KSort>()
                    val cStore = iStore.convertExpr<KSort>()
                    val cLambda = iLambda.convertExpr<KSort>()
                    val cAsArray = iAsArray.convertExpr<KSort>()

                    assertEquals(const, cConst)
                    assertEquals(select, cSelect)
                    assertEquals(store, cStore)
                    assertEquals(lambda, cLambda)
                    assertEquals(asArray, cAsArray)
                }
            }
        }
    }

    companion object {
        init {
            KZ3Solver(KContext()).close()
        }
    }
}
