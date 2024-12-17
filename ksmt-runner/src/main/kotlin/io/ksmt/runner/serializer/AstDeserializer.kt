package io.ksmt.runner.serializer

import com.jetbrains.rd.framework.AbstractBuffer
import com.jetbrains.rd.framework.readEnum
import io.ksmt.KAst
import io.ksmt.decl.KDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.uncheckedCast

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

        /**
         * Generate fresh decl because:
         * 1. Declaration is not in the [serializationCtx] --> declaration was not previously serialized.
         * 2. Was not serialized --> was not registered in our [KContext].
         * 3. Was not registered --> may clash with some another registered declaration.
         * 4. Use fresh declaration to ensure that this decl will never overlap with some other registered decl.
         * */
        mkFreshFuncDecl(name, sort, argSorts as List<KSort>)
    }

    private fun AbstractBuffer.deserializeSort(sortKind: SortKind): KSort = with(serializationCtx.ctx) {
        when (sortKind) {
            SortKind.Bool -> boolSort
            SortKind.Int -> intSort
            SortKind.Real -> realSort
            SortKind.String -> stringSort
            SortKind.Regex -> regexSort
            SortKind.FpRM -> mkFpRoundingModeSort()
            SortKind.Bv -> mkBvSort(readUInt())
            SortKind.Fp -> mkFpSort(readUInt(), readUInt())
            SortKind.Array -> mkArraySort(readSort(), readSort())
            SortKind.Array2 -> mkArraySort(readSort(), readSort(), readSort())
            SortKind.Array3 -> mkArraySort(readSort(), readSort(), readSort(), readSort())
            SortKind.ArrayN -> mkArrayNSort(readAstArray().uncheckedCast(), readSort())
            SortKind.Uninterpreted -> mkUninterpretedSort(readString())
        }
    }

    @Suppress("UNCHECKED_CAST", "LongMethod", "ComplexMethod")
    private fun AbstractBuffer.deserializeExpr(kind: ExprKind): KExpr<*> = with(serializationCtx.ctx) {
        when (kind) {
            ExprKind.FunctionApp -> mkApp(readDecl(), readAstArray() as List<KExpr<*>>)
            ExprKind.Const -> mkConstApp(readDecl())
            ExprKind.AndExpr -> mkAndNoSimplify(readAstArray() as List<KExpr<KBoolSort>>)
            ExprKind.AndBinaryExpr -> mkAndNoSimplify(readExpr(), readExpr())
            ExprKind.OrExpr -> mkOrNoSimplify(readAstArray() as List<KExpr<KBoolSort>>)
            ExprKind.OrBinaryExpr -> mkOrNoSimplify(readExpr(), readExpr())
            ExprKind.NotExpr -> deserialize(::mkNotNoSimplify)
            ExprKind.ImpliesExpr -> deserialize(::mkImpliesNoSimplify)
            ExprKind.XorExpr -> deserialize(::mkXorNoSimplify)
            ExprKind.True -> trueExpr
            ExprKind.False -> falseExpr
            ExprKind.EqExpr -> deserialize(::mkEqNoSimplify)
            ExprKind.DistinctExpr -> mkDistinctNoSimplify(readAstArray() as List<KExpr<KSort>>)
            ExprKind.IteExpr -> deserialize(::mkIteNoSimplify)
            ExprKind.BitVec1Value -> mkBv(readBoolean())
            ExprKind.BitVec8Value -> mkBv(readByte())
            ExprKind.BitVec16Value -> mkBv(readShort())
            ExprKind.BitVec32Value -> mkBv(readInt())
            ExprKind.BitVec64Value -> mkBv(readLong())
            ExprKind.BitVecCustomValue -> mkBv(readBigInteger(), readUInt())
            ExprKind.BvNotExpr -> deserialize(::mkBvNotExprNoSimplify)
            ExprKind.BvReductionAndExpr -> deserialize(::mkBvReductionAndExprNoSimplify)
            ExprKind.BvReductionOrExpr -> deserialize(::mkBvReductionOrExprNoSimplify)
            ExprKind.BvAndExpr -> deserialize(::mkBvAndExprNoSimplify)
            ExprKind.BvOrExpr -> deserialize(::mkBvOrExprNoSimplify)
            ExprKind.BvXorExpr -> deserialize(::mkBvXorExprNoSimplify)
            ExprKind.BvNAndExpr -> deserialize(::mkBvNAndExprNoSimplify)
            ExprKind.BvNorExpr -> deserialize(::mkBvNorExprNoSimplify)
            ExprKind.BvXNorExpr -> deserialize(::mkBvXNorExprNoSimplify)
            ExprKind.BvNegationExpr -> deserialize(::mkBvNegationExprNoSimplify)
            ExprKind.BvAddExpr -> deserialize(::mkBvAddExprNoSimplify)
            ExprKind.BvSubExpr -> deserialize(::mkBvSubExprNoSimplify)
            ExprKind.BvMulExpr -> deserialize(::mkBvMulExprNoSimplify)
            ExprKind.BvUnsignedDivExpr -> deserialize(::mkBvUnsignedDivExprNoSimplify)
            ExprKind.BvSignedDivExpr -> deserialize(::mkBvSignedDivExprNoSimplify)
            ExprKind.BvUnsignedRemExpr -> deserialize(::mkBvUnsignedRemExprNoSimplify)
            ExprKind.BvSignedRemExpr -> deserialize(::mkBvSignedRemExprNoSimplify)
            ExprKind.BvSignedModExpr -> deserialize(::mkBvSignedModExprNoSimplify)
            ExprKind.BvUnsignedLessExpr -> deserialize(::mkBvUnsignedLessExprNoSimplify)
            ExprKind.BvSignedLessExpr -> deserialize(::mkBvSignedLessExprNoSimplify)
            ExprKind.BvUnsignedLessOrEqualExpr -> deserialize(::mkBvUnsignedLessOrEqualExprNoSimplify)
            ExprKind.BvSignedLessOrEqualExpr -> deserialize(::mkBvSignedLessOrEqualExprNoSimplify)
            ExprKind.BvUnsignedGreaterOrEqualExpr -> deserialize(::mkBvUnsignedGreaterOrEqualExprNoSimplify)
            ExprKind.BvSignedGreaterOrEqualExpr -> deserialize(::mkBvSignedGreaterOrEqualExprNoSimplify)
            ExprKind.BvUnsignedGreaterExpr -> deserialize(::mkBvUnsignedGreaterExprNoSimplify)
            ExprKind.BvSignedGreaterExpr -> deserialize(::mkBvSignedGreaterExprNoSimplify)
            ExprKind.BvConcatExpr -> deserialize(::mkBvConcatExprNoSimplify)
            ExprKind.BvExtractExpr -> mkBvExtractExprNoSimplify(readInt(), readInt(), readExpr())
            ExprKind.BvSignExtensionExpr -> mkBvSignExtensionExprNoSimplify(readInt(), readExpr())
            ExprKind.BvZeroExtensionExpr -> mkBvZeroExtensionExprNoSimplify(readInt(), readExpr())
            ExprKind.BvRepeatExpr -> mkBvRepeatExprNoSimplify(readInt(), readExpr())
            ExprKind.BvShiftLeftExpr -> deserialize(::mkBvShiftLeftExprNoSimplify)
            ExprKind.BvLogicalShiftRightExpr -> deserialize(::mkBvLogicalShiftRightExprNoSimplify)
            ExprKind.BvArithShiftRightExpr -> deserialize(::mkBvArithShiftRightExprNoSimplify)
            ExprKind.BvRotateLeftExpr -> deserialize(::mkBvRotateLeftExprNoSimplify)
            ExprKind.BvRotateLeftIndexedExpr -> mkBvRotateLeftIndexedExprNoSimplify(readInt(), readExpr())
            ExprKind.BvRotateRightExpr -> deserialize(::mkBvRotateRightExprNoSimplify)
            ExprKind.BvRotateRightIndexedExpr -> mkBvRotateRightIndexedExprNoSimplify(readInt(), readExpr())
            ExprKind.Bv2IntExpr -> mkBv2IntExprNoSimplify(readExpr(), readBoolean())
            ExprKind.BvAddNoOverflowExpr -> mkBvAddNoOverflowExprNoSimplify(readExpr(), readExpr(), readBoolean())
            ExprKind.BvAddNoUnderflowExpr -> deserialize(::mkBvAddNoUnderflowExprNoSimplify)
            ExprKind.BvSubNoOverflowExpr -> deserialize(::mkBvSubNoOverflowExprNoSimplify)
            ExprKind.BvSubNoUnderflowExpr -> mkBvSubNoUnderflowExprNoSimplify(readExpr(), readExpr(), readBoolean())
            ExprKind.BvDivNoOverflowExpr -> deserialize(::mkBvDivNoOverflowExprNoSimplify)
            ExprKind.BvNegNoOverflowExpr -> deserialize(::mkBvNegationNoOverflowExprNoSimplify)
            ExprKind.BvMulNoOverflowExpr -> mkBvMulNoOverflowExprNoSimplify(readExpr(), readExpr(), readBoolean())
            ExprKind.BvMulNoUnderflowExpr -> deserialize(::mkBvMulNoUnderflowExprNoSimplify)
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
            ExprKind.FpAbsExpr -> deserialize(::mkFpAbsExprNoSimplify)
            ExprKind.FpNegationExpr -> deserialize(::mkFpNegationExprNoSimplify)
            ExprKind.FpAddExpr -> deserialize(::mkFpAddExprNoSimplify)
            ExprKind.FpSubExpr -> deserialize(::mkFpSubExprNoSimplify)
            ExprKind.FpMulExpr -> deserialize(::mkFpMulExprNoSimplify)
            ExprKind.FpDivExpr -> deserialize(::mkFpDivExprNoSimplify)
            ExprKind.FpFusedMulAddExpr -> deserialize(::mkFpFusedMulAddExprNoSimplify)
            ExprKind.FpSqrtExpr -> deserialize(::mkFpSqrtExprNoSimplify)
            ExprKind.FpRemExpr -> deserialize(::mkFpRemExprNoSimplify)
            ExprKind.FpRoundToIntegralExpr -> deserialize(::mkFpRoundToIntegralExprNoSimplify)
            ExprKind.FpMinExpr -> deserialize(::mkFpMinExprNoSimplify)
            ExprKind.FpMaxExpr -> deserialize(::mkFpMaxExprNoSimplify)
            ExprKind.FpLessOrEqualExpr -> deserialize(::mkFpLessOrEqualExprNoSimplify)
            ExprKind.FpLessExpr -> deserialize(::mkFpLessExprNoSimplify)
            ExprKind.FpGreaterOrEqualExpr -> deserialize(::mkFpGreaterOrEqualExprNoSimplify)
            ExprKind.FpGreaterExpr -> deserialize(::mkFpGreaterExprNoSimplify)
            ExprKind.FpEqualExpr -> deserialize(::mkFpEqualExprNoSimplify)
            ExprKind.FpIsNormalExpr -> deserialize(::mkFpIsNormalExprNoSimplify)
            ExprKind.FpIsSubnormalExpr -> deserialize(::mkFpIsSubnormalExprNoSimplify)
            ExprKind.FpIsZeroExpr -> deserialize(::mkFpIsZeroExprNoSimplify)
            ExprKind.FpIsInfiniteExpr -> deserialize(::mkFpIsInfiniteExprNoSimplify)
            ExprKind.FpIsNaNExpr -> deserialize(::mkFpIsNaNExprNoSimplify)
            ExprKind.FpIsNegativeExpr -> deserialize(::mkFpIsNegativeExprNoSimplify)
            ExprKind.FpIsPositiveExpr -> deserialize(::mkFpIsPositiveExprNoSimplify)
            ExprKind.FpToBvExpr -> mkFpToBvExprNoSimplify(readExpr(), readExpr(), readInt(), readBoolean())
            ExprKind.FpToRealExpr -> deserialize(::mkFpToRealExprNoSimplify)
            ExprKind.FpToIEEEBvExpr -> deserialize(::mkFpToIEEEBvExprNoSimplify)
            ExprKind.FpFromBvExpr -> deserialize(::mkFpFromBvExprNoSimplify)
            ExprKind.FpToFpExpr -> mkFpToFpExprNoSimplify(readSort() as KFpSort, readExpr(), readExpr())
            ExprKind.RealToFpExpr -> mkRealToFpExprNoSimplify(readSort() as KFpSort, readExpr(), readExpr())
            ExprKind.BvToFpExpr -> mkBvToFpExprNoSimplify(readSort() as KFpSort, readExpr(), readExpr(), readBoolean())
            ExprKind.ArrayStore -> mkArrayStoreNoSimplify(
                readExpr(), readExpr(), readExpr()
            )
            ExprKind.Array2Store -> mkArrayStoreNoSimplify(
                readExpr(), readExpr(), readExpr(), readExpr()
            )
            ExprKind.Array3Store -> mkArrayStoreNoSimplify(
                readExpr(), readExpr(), readExpr(), readExpr(), readExpr()
            )
            ExprKind.ArrayNStore -> mkArrayNStoreNoSimplify(
                readExpr(), readAstArray() as List<KExpr<KSort>>, readExpr()
            )
            ExprKind.ArraySelect -> mkArraySelectNoSimplify(
                readExpr(), readExpr()
            )
            ExprKind.Array2Select -> mkArraySelectNoSimplify(
                readExpr(), readExpr(), readExpr()
            )
            ExprKind.Array3Select -> mkArraySelectNoSimplify(
                readExpr(), readExpr(), readExpr(), readExpr()
            )
            ExprKind.ArrayNSelect -> mkArrayNSelectNoSimplify(
                readExpr(), readAstArray() as List<KExpr<KSort>>
            )
            ExprKind.ArrayConst -> mkArrayConst(readSort() as KArraySortBase<KSort>, readExpr())
            ExprKind.FunctionAsArray -> mkFunctionAsArray(
                readSort() as KArraySortBase<KSort>,
                readDecl() as KFuncDecl<KSort>
            )
            ExprKind.ArrayLambda -> mkArrayLambda(
                readDecl(), readExpr()
            )
            ExprKind.Array2Lambda -> mkArrayLambda(
                readDecl(), readDecl(), readExpr()
            )
            ExprKind.Array3Lambda -> mkArrayLambda(
                readDecl(), readDecl(), readDecl(), readExpr()
            )
            ExprKind.ArrayNLambda -> mkArrayNLambda(
                readAstArray() as List<KDecl<KSort>>, readExpr()
            )
            ExprKind.AddArithExpr -> mkArithAddNoSimplify(readAstArray() as List<KExpr<KArithSort>>)
            ExprKind.MulArithExpr -> mkArithMulNoSimplify(readAstArray() as List<KExpr<KArithSort>>)
            ExprKind.SubArithExpr -> mkArithSubNoSimplify(readAstArray() as List<KExpr<KArithSort>>)
            ExprKind.UnaryMinusArithExpr -> deserialize(::mkArithUnaryMinusNoSimplify)
            ExprKind.DivArithExpr -> deserialize(::mkArithDivNoSimplify)
            ExprKind.PowerArithExpr -> deserialize(::mkArithPowerNoSimplify)
            ExprKind.LtArithExpr -> deserialize(::mkArithLtNoSimplify)
            ExprKind.LeArithExpr -> deserialize(::mkArithLeNoSimplify)
            ExprKind.GtArithExpr -> deserialize(::mkArithGtNoSimplify)
            ExprKind.GeArithExpr -> deserialize(::mkArithGeNoSimplify)
            ExprKind.ModIntExpr -> deserialize(::mkIntModNoSimplify)
            ExprKind.RemIntExpr -> deserialize(::mkIntRemNoSimplify)
            ExprKind.ToRealIntExpr -> deserialize(::mkIntToRealNoSimplify)
            ExprKind.Int32NumExpr -> mkIntNum(readInt())
            ExprKind.Int64NumExpr -> mkIntNum(readLong())
            ExprKind.IntBigNumExpr -> mkIntNum(readString().toBigInteger())
            ExprKind.ToIntRealExpr -> deserialize(::mkRealToIntNoSimplify)
            ExprKind.IsIntRealExpr -> deserialize(::mkRealIsIntNoSimplify)
            ExprKind.RealNumExpr -> mkRealNum(readExpr<KIntSort>() as KIntNumExpr, readExpr<KIntSort>() as KIntNumExpr)
            ExprKind.ExistentialQuantifier -> {
                val bounds = readAstArray()
                mkExistentialQuantifier(readExpr(), bounds as List<KDecl<*>>)
            }
            ExprKind.UniversalQuantifier -> {
                val bounds = readAstArray()
                mkUniversalQuantifier(readExpr(), bounds as List<KDecl<*>>)
            }
            ExprKind.UninterpretedSortValue -> {
                mkUninterpretedSortValue(readSort() as KUninterpretedSort, readInt())
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
