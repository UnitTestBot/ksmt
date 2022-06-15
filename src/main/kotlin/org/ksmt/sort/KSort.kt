package org.ksmt.sort

abstract class KSort

object KBoolSort : KSort()

object KArithSort : KSort()

class KArraySort<D : KSort, R : KSort>(val domain: D, val range: R) : KSort()
