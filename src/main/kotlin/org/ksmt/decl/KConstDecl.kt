package org.ksmt.decl

import org.ksmt.sort.KSort

class KConstDecl<T : KSort>(val name: String, override val sort: T) : KDecl<T>()
