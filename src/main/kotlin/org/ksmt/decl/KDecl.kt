package org.ksmt.decl

import org.ksmt.sort.KSort

open class KDecl<T : KSort>(val name: String, val sort: T)

// todo: resolve known declarations
fun <T : KSort> mkDecl(name: String, sort: T) = KDecl(name, sort)
fun <T : KSort> T.mkDecl(name: String) = mkDecl(name, this)
