package org.ksmt.sort

import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.mkApp
import org.ksmt.expr.mkArithConst
import org.ksmt.expr.mkArrayConst
import org.ksmt.expr.mkBoolConst

abstract class KSort

object KBoolSort : KSort()

object KArithSort : KSort()

class KArraySort<D : KSort, R : KSort>(val domain: D, val range: R) : KSort()

fun <T : KSort> T.mkConst(name: String) = mkApp(KConstDecl(name, this), emptyList())
