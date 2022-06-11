package org.ksmt.sort

import org.ksmt.decl.KConstDecl
import org.ksmt.expr.*

abstract class KSort<T : KExpr<T>> {
    abstract fun mkConst(decl: KConstDecl<T>): KExpr<T>
}

fun <T : KExpr<T>> KSort<T>.mkConst(name: String) = mkConst(KConstDecl(name, this))


object KBoolSort : KSort<KBoolExpr>(){
    override fun mkConst(decl: KConstDecl<KBoolExpr>): KExpr<KBoolExpr> = mkBoolConst(decl)
}

object KArithSort : KSort<KArithExpr>(){
    override fun mkConst(decl: KConstDecl<KArithExpr>): KExpr<KArithExpr> = mkArithConst(decl)
}

class KArraySort<Domain : KExpr<Domain>, Range : KExpr<Range>>(val domain: KSort<Domain>, val range: KSort<Range>) : KSort<KArrayExpr<Domain, Range>>(){
    override fun mkConst(decl: KConstDecl<KArrayExpr<Domain, Range>>): KExpr<KArrayExpr<Domain, Range>> = mkArrayConst(decl)
}

val <Domain : KExpr<Domain>, Range : KExpr<Range>> KSort<KArrayExpr<Domain, Range>>.range: KSort<Range>
    get() = (this as KArraySort<Domain, Range>).range

val <Domain : KExpr<Domain>, Range : KExpr<Range>> KSort<KArrayExpr<Domain, Range>>.domain: KSort<Domain>
    get() = (this as KArraySort<Domain, Range>).domain
