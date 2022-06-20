package org.ksmt.decl

import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.mkConstApp
import org.ksmt.sort.KSort

open class KConstDecl<T : KSort>(name: String, sort: T) : KFuncDecl<T>(name, sort, emptyList()) {
    fun apply() = apply(emptyList())
    override fun apply(args: List<KExpr<*>>): KApp<T, *> {
        require(args.isEmpty())
        return mkConstApp(this)
    }
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

fun <T : KSort> mkConstDecl(name: String, sort: T) = KConstDecl(name, sort)
fun <T : KSort> T.mkConstDecl(name: String) = mkConstDecl(name, this)
fun <T : KSort> T.mkConst(name: String) = mkConstDecl(name).apply()
