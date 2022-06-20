package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort
import org.ksmt.expr.manager.ExprManager.intern

class KExistentialQuantifier internal constructor(
    body: KExpr<KBoolSort>,
    bounds: List<KDecl<*>>
) : KQuantifier(body, bounds) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

class KUniversalQuantifier internal constructor(
    body: KExpr<KBoolSort>,
    bounds: List<KDecl<*>>
) : KQuantifier(body, bounds) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
    KExistentialQuantifier(body, bounds).intern()

fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
    KUniversalQuantifier(body, bounds).intern()
