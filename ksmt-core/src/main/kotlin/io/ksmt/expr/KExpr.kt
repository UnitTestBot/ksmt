package io.ksmt.expr

import io.ksmt.KAst
import io.ksmt.KContext
import io.ksmt.cache.KInternedObject
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KSort
import io.ksmt.expr.printer.ExpressionPrinterWithLetBindings

abstract class KExpr<T : KSort>(ctx: KContext) : KAst(ctx), KInternedObject {

    abstract val sort: T

    abstract fun accept(transformer: KTransformerBase): KExpr<T>

    //  Contexts guarantee that any two equivalent expressions will be the same kotlin object
    override fun equals(other: Any?): Boolean = this === other

    override fun hashCode(): Int = System.identityHashCode(this)

    override fun print(builder: StringBuilder) {
        ExpressionPrinterWithLetBindings().print(this, builder)
    }

    abstract fun print(printer: ExpressionPrinter)
}
