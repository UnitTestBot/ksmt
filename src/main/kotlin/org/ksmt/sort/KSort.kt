package org.ksmt.sort

import org.ksmt.expr.KBVSize

abstract class KSort

object KBoolSort : KSort()

abstract class KArithSort<T: KArithSort<T>> : KSort()

object KIntSort: KArithSort<KIntSort>()

object KRealSort: KArithSort<KRealSort>()

class KBVSort<S : KBVSize> : KSort()

class KArraySort<D : KSort, R : KSort>(val domain: D, val range: R) : KSort()
