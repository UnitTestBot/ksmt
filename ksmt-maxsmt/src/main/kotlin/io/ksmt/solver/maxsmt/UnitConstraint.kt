package io.ksmt.solver.maxsmt

import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.expr.KExpr
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort

// TODO: do I correctly work with context?
class UnitConstraint(val i: Int, val j: Int, val formula: KExpr<KBoolSort>, override val sort: KBoolSort)
            : KExpr<KBoolSort>(formula.ctx) {
    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun print(printer: ExpressionPrinter) {
        TODO("Not yet implemented")
    }

    override fun internEquals(other: Any): Boolean {
        // Тут не учитываются i и j.
        return structurallyEqual(other) { formula }
    }

    override fun internHashCode(): Int {
        // Is this hash ok for me?
        return hash(i, j, formula)
    }

    override fun toString(): String {
        return ""
    }
}
