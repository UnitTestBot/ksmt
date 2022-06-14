package org.ksmt.decl

import org.ksmt.sort.KSort

class KConstDecl<T : KSort>(name: String, sort: T) : KDecl<T>(name, sort)
