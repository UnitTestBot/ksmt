package org.ksmt.decl

import org.ksmt.sort.KSort

abstract class KDecl<T : KSort> {
    abstract val sort: T
}
