package org.ksmt.sort

import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.mkArithConst
import org.ksmt.expr.mkArrayConst
import org.ksmt.expr.mkBoolConst

abstract class KSort<T : KSort<T>> {
    abstract fun mkConst(decl: KConstDecl<T>): KExpr<T>
}

fun <T : KSort<T>> T.mkConst(name: String) = mkConst(KConstDecl(name, this))


object KBoolSort : KSort<KBoolSort>() {
    override fun mkConst(decl: KConstDecl<KBoolSort>): KExpr<KBoolSort> = mkBoolConst(decl)
}

object KArithSort : KSort<KArithSort>() {
    override fun mkConst(decl: KConstDecl<KArithSort>): KExpr<KArithSort> = mkArithConst(decl)
}

class KArraySort<D : KSort<D>, R : KSort<R>>(val domain: D, val range: R) : KSort<KArraySort<D, R>>() {
    override fun mkConst(decl: KConstDecl<KArraySort<D, R>>): KExpr<KArraySort<D, R>> = mkArrayConst(decl)
}
