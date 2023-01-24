package org.ksmt.runner.serializer

import com.jetbrains.rd.framework.AbstractBuffer
import com.jetbrains.rd.framework.readEnum
import org.ksmt.KAst
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KSort

@OptIn(ExperimentalUnsignedTypes::class)
class AstDeserializer(
    private val serializationCtx: AstSerializationCtx,
    private val input: AbstractBuffer
) {
    private inline fun <reified T : KAst> readAst(): T {
        val idx = input.readInt()
        val ast = serializationCtx.getAstByIndexOrError(idx)
        return ast as T
    }

    private fun readAstArray(): List<KAst> {
        val indices = input.readIntArray()
        return indices.map { serializationCtx.getAstByIndexOrError(it) }
    }

    private fun readDecl(): KDecl<*> = readAst()
    private fun readSort(): KSort = readAst()
    private fun <T : KSort> readExpr(): KExpr<T> = readAst()

    fun deserializeAst(): KAst {
        while (true) {
            val idx = input.readInt()

            if (idx == AstSerializationCtx.SERIALIZED_DATA_END_IDX) {
                break
            }

            val kind = input.readEnum<AstKind>()
            val deserialized = when (kind) {
                AstKind.Expr -> {
                    val exprKind = input.readEnum<ExprKind>()
                    input.deserializeExpr(exprKind)
                }

                AstKind.Sort -> {
                    val sortKind = input.readEnum<SortKind>()
                    input.deserializeSort(sortKind)
                }

                AstKind.Decl -> input.deserializeDecl()
            }

            val entryEndMarker = input.readInt()
            check(entryEndMarker == AstSerializationCtx.SERIALIZED_AST_ENTRY_END) {
                "Serialization failed: end marker expected"
            }

            serializationCtx.writeAst(idx, deserialized)
        }

        val serializedAstIdx = input.readInt()
        return serializationCtx.getAstByIndexOrError(serializedAstIdx)
    }

    @Suppress("UNCHECKED_CAST")
    private fun AbstractBuffer.deserializeDecl(): KDecl<*> = with(serializationCtx.ctx) {
        val name = readString()
        val argSorts = readAstArray()
        val sort = readSort()
        mkFuncDecl(name, sort, argSorts as List<KSort>)
    }

    private fun AbstractBuffer.deserializeSort(sortKind: SortKind): KSort = with(serializationCtx.ctx) {
        when (sortKind) {
            SortKind.Bool -> boolSort
            SortKind.Int -> intSort
            SortKind.Real -> realSort
            SortKind.FpRM -> mkFpRoundingModeSort()
            SortKind.Bv -> mkBvSort(readUInt())
            SortKind.Fp -> mkFpSort(readUInt(), readUInt())
            SortKind.Array -> mkArraySort(readSort(), readSort())
            SortKind.Uninterpreted -> mkUninterpretedSort(readString())
        }
    }

    @Suppress("UNCHECKED_CAST", "LongMethod", "ComplexMethod")
    private fun AbstractBuffer.deserializeExpr(kind: ExprKind): KExpr<*> = with(serializationCtx.ctx) {
        when (kind) {
            ExprKind.FunctionApp -> mkApp(readDecl(), readAstArray() as List<KExpr<*>>)
            ExprKind.Const -> mkConstApp(readDecl())
            ExprKind.AndExpr -> mkAnd(readAstArray() as List<KExpr<KBoolSort>>)
            ExprKind.OrExpr -> mkOr(readAstArray() as List<KExpr<KBoolSort>>)
            ExprKind.NotExpr -> deserialize(::mkNot)
            ExprKind.ImpliesExpr -> deserialize(::mkImplies)
            ExprKind.XorExpr -> deserialize(::mkXor)
            ExprKind.True -> trueExpr
            ExprKind.False -> falseExpr
            ExprKind.EqExpr -> deserialize(::mkEq)
            ExprKind.DistinctExpr -> mkDistinct(readAstArray() as List<KExpr<KSort>>)
            ExprKind.IteExpr -> deserialize(::mkIte)
            ExprKind.BitVec1Value -> mkBv(readBoolean())
            ExprKind.BitVec8Value -> mkBv(readByte())
            ExprKind.BitVec16Value -> mkBv(readShort())
            ExprKind.BitVec32Value -> mkBv(readInt())
            ExprKind.BitVec64Value -> mkBv(readLong())
            ExprKind.BitVecCustomValue -> mkBv(readBigInteger(), readUInt())
            ExprKind.BvNotExpr -> deserialize(::mkBvNotExpr)
            ExprKind.BvReductionAndExpr -> deserialize(::mkBvReductionAndExpr)
            ExprKind.BvReductionOrExpr -> deserialize(::mkBvReductionOrExpr)
            ExprKind.BvAndExpr -> deserialize(::mkBvAndExpr)
            ExprKind.BvOrExpr -> deserialize(::mkBvOrExpr)
            ExprKind.BvXorExpr -> deserialize(::mkBvXorExpr)
            ExprKind.BvNAndExpr -> deserialize(::mkBvNAndExpr)
            ExprKind.BvNorExpr -> deserialize(::mkBvNorExpr)
            ExprKind.BvXNorExpr -> deserialize(::mkBvXNorExpr)
            ExprKind.BvNegationExpr -> deserialize(::mkBvNegationExpr)
            ExprKind.BvAddExpr -> deserialize(::mkBvAddExpr)
            ExprKind.BvSubExpr -> deserialize(::mkBvSubExpr)
            ExprKind.BvMulExpr -> deserialize(::mkBvMulExpr)
            ExprKind.BvUnsignedDivExpr -> deserialize(::mkBvUnsignedDivExpr)
            ExprKind.BvSignedDivExpr -> deserialize(::mkBvSignedDivExpr)
            ExprKind.BvUnsignedRemExpr -> deserialize(::mkBvUnsignedRemExpr)
            ExprKind.BvSignedRemExpr -> deserialize(::mkBvSignedRemExpr)
            ExprKind.BvSignedModExpr -> deserialize(::mkBvSignedModExpr)
            ExprKind.BvUnsignedLessExpr -> deserialize(::mkBvUnsignedLessExpr)
            ExprKind.BvSignedLessExpr -> deserialize(::mkBvSignedLessExpr)
            ExprKind.BvUnsignedLessOrEqualExpr -> deserialize(::mkBvUnsignedLessOrEqualExpr)
            ExprKind.BvSignedLessOrEqualExpr -> deserialize(::mkBvSignedLessOrEqualExpr)
            ExprKind.BvUnsignedGreaterOrEqualExpr -> deserialize(::mkBvUnsignedGreaterOrEqualExpr)
            ExprKind.BvSignedGreaterOrEqualExpr -> deserialize(::mkBvSignedGreaterOrEqualExpr)
            ExprKind.BvUnsignedGreaterExpr -> deserialize(::mkBvUnsignedGreaterExpr)
            ExprKind.BvSignedGreaterExpr -> deserialize(::mkBvSignedGreaterExpr)
            ExprKind.BvConcatExpr -> deserialize(::mkBvConcatExpr)
            ExprKind.BvExtractExpr -> mkBvExtractExpr(readInt(), readInt(), readExpr())
            ExprKind.BvSignExtensionExpr -> mkBvSignExtensionExpr(readInt(), readExpr())
            ExprKind.BvZeroExtensionExpr -> mkBvZeroExtensionExpr(readInt(), readExpr())
            ExprKind.BvRepeatExpr -> mkBvRepeatExpr(readInt(), readExpr())
            ExprKind.BvShiftLeftExpr -> deserialize(::mkBvShiftLeftExpr)
            ExprKind.BvLogicalShiftRightExpr -> deserialize(::mkBvLogicalShiftRightExpr)
            ExprKind.BvArithShiftRightExpr -> deserialize(::mkBvArithShiftRightExpr)
            ExprKind.BvRotateLeftExpr -> deserialize(::mkBvRotateLeftExpr)
            ExprKind.BvRotateLeftIndexedExpr -> mkBvRotateLeftIndexedExpr(readInt(), readExpr())
            ExprKind.BvRotateRightExpr -> deserialize(::mkBvRotateRightExpr)
            ExprKind.BvRotateRightIndexedExpr -> mkBvRotateRightIndexedExpr(readInt(), readExpr())
            ExprKind.Bv2IntExpr -> mkBv2IntExpr(readExpr(), readBoolean())
            ExprKind.BvAddNoOverflowExpr -> mkBvAddNoOverflowExpr(readExpr(), readExpr(), readBoolean())
            ExprKind.BvAddNoUnderflowExpr -> deserialize(::mkBvAddNoUnderflowExpr)
            ExprKind.BvSubNoOverflowExpr -> deserialize(::mkBvSubNoOverflowExpr)
            ExprKind.BvSubNoUnderflowExpr -> mkBvSubNoUnderflowExpr(readExpr(), readExpr(), readBoolean())
            ExprKind.BvDivNoOverflowExpr -> deserialize(::mkBvDivNoOverflowExpr)
            ExprKind.BvNegNoOverflowExpr -> deserialize(::mkBvNegationNoOverflowExpr)
            ExprKind.BvMulNoOverflowExpr -> mkBvMulNoOverflowExpr(readExpr(), readExpr(), readBoolean())
            ExprKind.BvMulNoUnderflowExpr -> deserialize(::mkBvMulNoUnderflowExpr)
            ExprKind.Fp16Value -> mkFp16(readFloat())
            ExprKind.Fp32Value -> mkFp32(readFloat())
            ExprKind.Fp64Value -> mkFp64(readDouble())
            ExprKind.Fp128Value -> mkFp128Biased(readAst(), readAst(), readBoolean())
            ExprKind.FpCustomSizeValue -> mkFpCustomSizeBiased(
                significandSize = readUInt(),
                exponentSize = readUInt(),
                significand = readAst(),
                biasedExponent = readAst(),
                signBit = readBoolean()
            )
            ExprKind.FpRoundingModeExpr -> mkFpRoundingModeExpr(readEnum())
            ExprKind.FpAbsExpr -> deserialize(::mkFpAbsExpr)
            ExprKind.FpNegationExpr -> deserialize(::mkFpNegationExpr)
            ExprKind.FpAddExpr -> deserialize(::mkFpAddExpr)
            ExprKind.FpSubExpr -> deserialize(::mkFpSubExpr)
            ExprKind.FpMulExpr -> deserialize(::mkFpMulExpr)
            ExprKind.FpDivExpr -> deserialize(::mkFpDivExpr)
            ExprKind.FpFusedMulAddExpr -> deserialize(::mkFpFusedMulAddExpr)
            ExprKind.FpSqrtExpr -> deserialize(::mkFpSqrtExpr)
            ExprKind.FpRemExpr -> deserialize(::mkFpRemExpr)
            ExprKind.FpRoundToIntegralExpr -> deserialize(::mkFpRoundToIntegralExpr)
            ExprKind.FpMinExpr -> deserialize(::mkFpMinExpr)
            ExprKind.FpMaxExpr -> deserialize(::mkFpMaxExpr)
            ExprKind.FpLessOrEqualExpr -> deserialize(::mkFpLessOrEqualExpr)
            ExprKind.FpLessExpr -> deserialize(::mkFpLessExpr)
            ExprKind.FpGreaterOrEqualExpr -> deserialize(::mkFpGreaterOrEqualExpr)
            ExprKind.FpGreaterExpr -> deserialize(::mkFpGreaterExpr)
            ExprKind.FpEqualExpr -> deserialize(::mkFpEqualExpr)
            ExprKind.FpIsNormalExpr -> deserialize(::mkFpIsNormalExpr)
            ExprKind.FpIsSubnormalExpr -> deserialize(::mkFpIsSubnormalExpr)
            ExprKind.FpIsZeroExpr -> deserialize(::mkFpIsZeroExpr)
            ExprKind.FpIsInfiniteExpr -> deserialize(::mkFpIsInfiniteExpr)
            ExprKind.FpIsNaNExpr -> deserialize(::mkFpIsNaNExpr)
            ExprKind.FpIsNegativeExpr -> deserialize(::mkFpIsNegativeExpr)
            ExprKind.FpIsPositiveExpr -> deserialize(::mkFpIsPositiveExpr)
            ExprKind.FpToBvExpr -> mkFpToBvExpr(readExpr(), readExpr(), readInt(), readBoolean())
            ExprKind.FpToRealExpr -> deserialize(::mkFpToRealExpr)
            ExprKind.FpToIEEEBvExpr -> deserialize(::mkFpToIEEEBvExpr)
            ExprKind.FpFromBvExpr -> deserialize(::mkFpFromBvExpr)
            ExprKind.FpToFpExpr -> mkFpToFpExpr(readSort() as KFpSort, readExpr(), readExpr())
            ExprKind.RealToFpExpr -> mkRealToFpExpr(readSort() as KFpSort, readExpr(), readExpr())
            ExprKind.BvToFpExpr -> mkBvToFpExpr(readSort() as KFpSort, readExpr(), readExpr(), readBoolean())
            ExprKind.ArrayStore -> deserialize(::mkArrayStore)
            ExprKind.ArraySelect -> deserialize(::mkArraySelect)
            ExprKind.ArrayConst -> mkArrayConst(readSort() as KArraySort<KSort, KSort>, readExpr())
            ExprKind.FunctionAsArray -> mkFunctionAsArray<KSort, KSort>(readDecl() as KFuncDecl<KSort>)
            ExprKind.ArrayLambda -> mkArrayLambda(readDecl(), readExpr())
            ExprKind.AddArithExpr -> mkArithAdd(readAstArray() as List<KExpr<KArithSort>>)
            ExprKind.MulArithExpr -> mkArithMul(readAstArray() as List<KExpr<KArithSort>>)
            ExprKind.SubArithExpr -> mkArithSub(readAstArray() as List<KExpr<KArithSort>>)
            ExprKind.UnaryMinusArithExpr -> deserialize(::mkArithUnaryMinus)
            ExprKind.DivArithExpr -> deserialize(::mkArithDiv)
            ExprKind.PowerArithExpr -> deserialize(::mkArithPower)
            ExprKind.LtArithExpr -> deserialize(::mkArithLt)
            ExprKind.LeArithExpr -> deserialize(::mkArithLe)
            ExprKind.GtArithExpr -> deserialize(::mkArithGt)
            ExprKind.GeArithExpr -> deserialize(::mkArithGe)
            ExprKind.ModIntExpr -> deserialize(::mkIntMod)
            ExprKind.RemIntExpr -> deserialize(::mkIntRem)
            ExprKind.ToRealIntExpr -> deserialize(::mkIntToReal)
            ExprKind.Int32NumExpr -> mkIntNum(readInt())
            ExprKind.Int64NumExpr -> mkIntNum(readLong())
            ExprKind.IntBigNumExpr -> mkIntNum(readString().toBigInteger())
            ExprKind.ToIntRealExpr -> deserialize(::mkRealToInt)
            ExprKind.IsIntRealExpr -> deserialize(::mkRealIsInt)
            ExprKind.RealNumExpr -> mkRealNum(readExpr<KIntSort>() as KIntNumExpr, readExpr<KIntSort>() as KIntNumExpr)
            ExprKind.ExistentialQuantifier -> {
                val bounds = readAstArray()
                mkExistentialQuantifier(readExpr(), bounds as List<KDecl<*>>)
            }
            ExprKind.UniversalQuantifier -> {
                val bounds = readAstArray()
                mkUniversalQuantifier(readExpr(), bounds as List<KDecl<*>>)
            }
        }
    }
    
    private inline fun <S : KSort, A0 : KSort> deserialize(
        op: (KExpr<A0>) -> KExpr<S>
    ): KExpr<S> = op(readExpr())

    private inline fun <S : KSort, A0 : KSort, A1 : KSort> deserialize(
        op: (KExpr<A0>, KExpr<A1>) -> KExpr<S>
    ): KExpr<S> = op(readExpr(), readExpr())

    private inline fun <S : KSort, A0 : KSort, A1 : KSort, A2 : KSort> deserialize(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<S>
    ): KExpr<S> = op(readExpr(), readExpr(), readExpr())

    private inline fun <S : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> deserialize(
        op: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<S>
    ): KExpr<S> = op(readExpr(), readExpr(), readExpr(), readExpr())

}
