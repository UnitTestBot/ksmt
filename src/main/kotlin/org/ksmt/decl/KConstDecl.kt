package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

open class KConstDecl<T : KSort>(name: String, sort: T) : KFuncDecl<T>(name, sort, emptyList()) {
    fun KContext.apply() = apply(emptyList())
    override fun KContext.apply(args: List<KExpr<*>>): KApp<T, *> {
        require(args.isEmpty())
        return mkConstApp(this@KConstDecl)
    }
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
