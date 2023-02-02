package org.ksmt.expr

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.cache.KInternedObject
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort
import org.ksmt.expr.printer.ExpressionPrinterWithLetBindings

abstract class KExpr<T : KSort>(ctx: KContext) : KAst(ctx), KInternedObject {

    abstract val sort: T

    @Deprecated("Use property", ReplaceWith("sort"))
    fun sort(): T = sort

    abstract fun accept(transformer: KTransformerBase): KExpr<T>

    //  Contexts guarantee that any two equivalent expressions will be the same kotlin object
    override fun equals(other: Any?): Boolean = this === other

    override fun hashCode(): Int = System.identityHashCode(this)

    /**
     * Some expressions require evaluation of nested expressions sorts in order to compute the sort.
     * To compute sort non-recursively, override this method and use [KContext.getExprSort].
     * */
    open fun computeExprSort(): T = sort

    /**
     * Add the expressions, needed to compute the sort, to the provided [dependency] list.
     * To compute sort non-recursively, override this method.
     * @see [computeExprSort]
     * */
    open fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {}

    override fun print(builder: StringBuilder) {
        ExpressionPrinterWithLetBindings().print(this, builder)
    }

    abstract fun print(printer: ExpressionPrinter)
}
