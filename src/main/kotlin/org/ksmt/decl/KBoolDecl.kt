package org.ksmt.decl

import org.ksmt.sort.KBoolSort

abstract class KBoolDecl : KDecl<KBoolSort>() {
    override val sort = KBoolSort
}
