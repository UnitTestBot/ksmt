package io.ksmt.symfpu.operations

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KFpSort


internal fun <Fp : KFpSort> isNormal(
    uf: UnpackedFp<Fp>
): KExpr<KBoolSort> = with(uf.ctx) { !uf.isNaN and !uf.isInf and !uf.isZero and uf.inNormalRange() }

internal fun <Fp : KFpSort> isSubnormal(
    uf: UnpackedFp<Fp>
): KExpr<KBoolSort> = with(uf.ctx) { !uf.isNaN and !uf.isInf and !uf.isZero and !uf.inNormalRange() }


internal fun <Fp : KFpSort> isPositive(
    uf: UnpackedFp<Fp>
): KExpr<KBoolSort> = with(uf.ctx) { !uf.isNaN and !uf.isNegative }

internal fun <Fp : KFpSort> isNegative(
    uf: UnpackedFp<Fp>
): KExpr<KBoolSort> = with(uf.ctx) { !uf.isNaN and uf.isNegative }



