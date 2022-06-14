package org.ksmt.sort

import org.ksmt.decl.KConstDecl
import org.ksmt.expr.mkApp

abstract class KSort

object KBoolSort : KSort()

object KArithSort : KSort()

class KArraySort<D : KSort, R : KSort>(val domain: D, val range: R) : KSort()

fun <T : KSort> T.mkConst(name: String) = mkApp(KConstDecl(name, this), emptyList())
