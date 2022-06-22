package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


class KAndExpr internal constructor(args: List<KExpr<KBoolSort>>) : KBoolExpr<KExpr<KBoolSort>>(args) {
    override fun KContext.decl() = mkAndDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KOrExpr internal constructor(args: List<KExpr<KBoolSort>>) : KBoolExpr<KExpr<KBoolSort>>(args) {
    override fun KContext.decl() = mkOrDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KNotExpr internal constructor(val arg: KExpr<KBoolSort>) : KBoolExpr<KExpr<KBoolSort>>(listOf(arg)) {
    override fun KContext.decl() = mkNotDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KEqExpr<T : KSort> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr<KExpr<T>>(listOf(lhs, rhs)) {
    override fun KContext.decl() = mkEqDecl(lhs.sort)
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KIteExpr<T : KSort> internal constructor(
    val condition: KExpr<KBoolSort>,
    val trueBranch: KExpr<T>,
    val falseBranch: KExpr<T>
) : KApp<T, KExpr<*>>(listOf(condition, trueBranch, falseBranch)) {
    override fun KContext.sort() = trueBranch.sort
    override fun KContext.decl() = mkIteDecl(trueBranch.sort)
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

object KTrue : KBoolExpr<KExpr<*>>(emptyList()) {
    override fun KContext.decl() = mkTrueDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

object KFalse : KBoolExpr<KExpr<*>>(emptyList()) {
    override fun KContext.decl() = mkFalseDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}
