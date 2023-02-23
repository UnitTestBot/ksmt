package org.ksmt.solver.z3

import com.microsoft.z3.Context
import org.ksmt.KContext
import org.ksmt.sort.KSort
import org.ksmt.utils.getValue
import kotlin.test.Test

class ArrayTest {

    @Test
    fun testArrayConversion(): Unit = with(KContext()) {
        val sort = mkArraySort(bv32Sort, bv64Sort)

        val array by sort
        val index by bv32Sort
        val value by bv64Sort

        val const = mkArrayConst(sort, value)
        val select = mkArraySelect(array, index)
        val store = mkArrayStore(array, index, value)
        val lambda = mkArrayLambda(index.decl, select)

        val ctx = this
        Context().use { z3Native ->
            val z3InternCtx = KZ3Context(z3Native)
            val z3ConvertCtx = KZ3Context(z3Native)

            with(KZ3ExprInternalizer(ctx, z3InternCtx)) {
                val iConst = const.internalizeExprWrapped()
                val iSelect = select.internalizeExprWrapped()
                val iStore = store.internalizeExprWrapped()
                val iLambda = lambda.internalizeExprWrapped()

                with(KZ3ExprConverter(ctx, z3ConvertCtx)) {
                    iConst.convertExprWrapped<KSort>()
                    iSelect.convertExprWrapped<KSort>()
                    iStore.convertExprWrapped<KSort>()
                    iLambda.convertExprWrapped<KSort>()
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
