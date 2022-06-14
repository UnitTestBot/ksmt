package org.ksmt.sort

import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.mkApp
import org.ksmt.expr.mkArithConst
import org.ksmt.expr.mkArrayConst
import org.ksmt.expr.mkBoolConst

abstract class KSort {
}

fun <T : KSort<T>> T.mkConst(name: String) = mkApp(KConstDecl(name, this), emptyList())


object KBoolSort : KSort<KBoolSort>() {
}

object KArithSort : KSort<KArithSort>() {
}

class KArraySort<D : KSort<D>, R : KSort<R>>(val domain: D, val range: R) : KSort<KArraySort<D, R>>() {
}
