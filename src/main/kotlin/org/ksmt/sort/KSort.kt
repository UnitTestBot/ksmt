package org.ksmt.sort

import org.ksmt.expr.KBVSize

interface KSort

object KBoolSort : KSort

interface KArithSort<T: KArithSort<T>> : KSort

object KIntSort: KArithSort<KIntSort>

object KRealSort: KArithSort<KRealSort>

class KBVSort<S : KBVSize> : KSort

class KArraySort<D : KSort, R : KSort>(val domain: D, val range: R) : KSort
