package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort

class KExistentialQuantifier internal constructor(
    body: KExpr<KBoolSort>,
    bounds: List<KDecl<*>>
) : KQuantifier(body, bounds) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KUniversalQuantifier internal constructor(
    body: KExpr<KBoolSort>,
    bounds: List<KDecl<*>>
) : KQuantifier(body, bounds) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}
