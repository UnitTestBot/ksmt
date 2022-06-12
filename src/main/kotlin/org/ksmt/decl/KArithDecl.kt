package org.ksmt.decl

import org.ksmt.sort.KArithSort

abstract class KArithDecl : KDecl<KArithSort>() {
    override val sort = KArithSort
}
