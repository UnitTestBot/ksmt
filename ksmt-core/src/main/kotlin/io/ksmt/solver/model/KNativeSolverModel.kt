package io.ksmt.solver.model

import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KModel
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort

class KNativeSolverModel(nativeModel: KModel): KModel {
    private var model: KModel = nativeModel

    override val declarations: Set<KDecl<*>> get() = model.declarations

    override val uninterpretedSorts: Set<KUninterpretedSort> get() = model.uninterpretedSorts

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> =
        model.eval(expr, isComplete)

    override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? =
        model.interpretation(decl)

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? =
        model.uninterpretedSortUniverse(sort)

    override fun detach(): KModel {
        model = model.detach()
        return model
    }

    override fun close() {
        model.close()
    }

    override fun toString(): String = detach().toString()
    override fun hashCode(): Int = detach().hashCode()
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KModel) return false
        return detach() == other
    }
}
