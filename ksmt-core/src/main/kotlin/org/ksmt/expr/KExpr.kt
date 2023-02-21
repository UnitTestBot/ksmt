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

    abstract fun accept(transformer: KTransformerBase): KExpr<T>

    //  Contexts guarantee that any two equivalent expressions will be the same kotlin object
    override fun equals(other: Any?): Boolean = this === other

    override fun hashCode(): Int = System.identityHashCode(this)

    override fun print(builder: StringBuilder) {
        ExpressionPrinterWithLetBindings().print(this, builder)
    }

    abstract fun print(printer: ExpressionPrinter)
}
