package org.ksmt.runner.serializer

import com.jetbrains.rd.framework.AbstractBuffer
import com.jetbrains.rd.framework.writeEnum
import org.ksmt.KAst
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBv2IntExpr
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
import org.ksmt.expr.KBvExtractExpr
import org.ksmt.expr.KBvLogicalShiftRightExpr
import org.ksmt.expr.KBvMulExpr
import org.ksmt.expr.KBvMulNoOverflowExpr
import org.ksmt.expr.KBvMulNoUnderflowExpr
import org.ksmt.expr.KBvNAndExpr
import org.ksmt.expr.KBvNegNoOverflowExpr
import org.ksmt.expr.KBvNegationExpr
import org.ksmt.expr.KBvNorExpr
import org.ksmt.expr.KBvNotExpr
import org.ksmt.expr.KBvOrExpr
import org.ksmt.expr.KBvReductionAndExpr
import org.ksmt.expr.KBvReductionOrExpr
import org.ksmt.expr.KBvRepeatExpr
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvShiftLeftExpr
import org.ksmt.expr.KBvSignExtensionExpr
import org.ksmt.expr.KBvSignedDivExpr
import org.ksmt.expr.KBvSignedGreaterExpr
import org.ksmt.expr.KBvSignedGreaterOrEqualExpr
import org.ksmt.expr.KBvSignedLessExpr
import org.ksmt.expr.KBvSignedLessOrEqualExpr
import org.ksmt.expr.KBvSignedModExpr
import org.ksmt.expr.KBvSignedRemExpr
import org.ksmt.expr.KBvSubExpr
import org.ksmt.expr.KBvSubNoOverflowExpr
import org.ksmt.expr.KBvSubNoUnderflowExpr
import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KBvUnsignedDivExpr
import org.ksmt.expr.KBvUnsignedGreaterExpr
import org.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.KBvUnsignedLessExpr
import org.ksmt.expr.KBvUnsignedLessOrEqualExpr
import org.ksmt.expr.KBvUnsignedRemExpr
import org.ksmt.expr.KBvXNorExpr
import org.ksmt.expr.KBvXorExpr
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFp128Value
import org.ksmt.expr.KFp16Value
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpCustomSizeValue
import org.ksmt.expr.KFpDivExpr
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.expr.KFpFromBvExpr
import org.ksmt.expr.KFpFusedMulAddExpr
import org.ksmt.expr.KFpGreaterExpr
import org.ksmt.expr.KFpGreaterOrEqualExpr
import org.ksmt.expr.KFpIsInfiniteExpr
import org.ksmt.expr.KFpIsNaNExpr
import org.ksmt.expr.KFpIsNegativeExpr
import org.ksmt.expr.KFpIsNormalExpr
import org.ksmt.expr.KFpIsPositiveExpr
import org.ksmt.expr.KFpIsSubnormalExpr
import org.ksmt.expr.KFpIsZeroExpr
import org.ksmt.expr.KFpLessExpr
import org.ksmt.expr.KFpLessOrEqualExpr
import org.ksmt.expr.KFpMaxExpr
import org.ksmt.expr.KFpMinExpr
import org.ksmt.expr.KFpMulExpr
import org.ksmt.expr.KFpNegationExpr
import org.ksmt.expr.KFpRemExpr
import org.ksmt.expr.KFpRoundToIntegralExpr
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.solver.util.KExprInternalizerBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort

@Suppress("ArrayPrimitive", "LargeClass")
@OptIn(ExperimentalUnsignedTypes::class)
class AstSerializer(
    private val serializationCtx: AstSerializationCtx,
    private val output: AbstractBuffer
) : KExprInternalizerBase<Int>() {

    private val exprKindMapper = ExprKindMapper()

    fun serializeAst(ast: KAst) {
        val serializedAst = run {
            when (ast) {
                is KDecl<*> -> ast.serializeDecl()
                is KSort -> ast.serializeSort()
                is KExpr<*> -> ast.serializeExpr()
                else -> error("Unexpected ast: ${ast::class}")
            }
        }
        output.writeInt(AstSerializationCtx.SERIALIZED_DATA_END_IDX)
        output.writeInt(serializedAst)
    }

    private fun <T : KSort> KExpr<T>.serializeExpr(): Int = internalizeExpr()

    private fun <T : KDecl<*>> T.serializeDecl(): Int {
        val idx = serializationCtx.getAstIndex(this)
        if (idx != null) return idx
        return accept(declSerializer)
    }

    private fun <T : KSort> T.serializeSort(): Int {
        val idx = serializationCtx.getAstIndex(this)
        if (idx != null) return idx
        return accept(sortSerializer)
    }

    override fun findInternalizedExpr(expr: KExpr<*>): Int? = serializationCtx.getAstIndex(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: Int) {
        // Do nothing since expr is already saved into serializationCtx
    }

    private val sortSerializer = SortSerializer()
    private val declSerializer = DeclSerializer()

    @OptIn(ExperimentalUnsignedTypes::class)
    private inner class SortSerializer : KSortVisitor<Int> {
        override fun visit(sort: KBoolSort): Int = serializeSort(sort, SortKind.Bool) {}

        override fun visit(sort: KIntSort): Int = serializeSort(sort, SortKind.Int) {}

        override fun visit(sort: KRealSort): Int = serializeSort(sort, SortKind.Real) {}

        override fun <S : KBvSort> visit(sort: S): Int = serializeSort(sort, SortKind.Bv) {
            writeUInt(sort.sizeBits)
        }

        override fun <S : KFpSort> visit(sort: S): Int = serializeSort(sort, SortKind.Fp) {
            writeUInt(sort.exponentBits)
            writeUInt(sort.significandBits)
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): Int {
            val domain = sort.domain.serializeSort()
            val range = sort.range.serializeSort()
            return serializeSort(sort, SortKind.Array) {
                writeAst(domain)
                writeAst(range)
            }
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>): Int {
            val domain0 = sort.domain0.serializeSort()
            val domain1 = sort.domain1.serializeSort()
            val range = sort.range.serializeSort()
            return serializeSort(sort, SortKind.Array2) {
                writeAst(domain0)
                writeAst(domain1)
                writeAst(range)
            }
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>): Int {
            val domain0 = sort.domain0.serializeSort()
            val domain1 = sort.domain1.serializeSort()
            val domain2 = sort.domain2.serializeSort()
            val range = sort.range.serializeSort()
            return serializeSort(sort, SortKind.Array3) {
                writeAst(domain0)
                writeAst(domain1)
                writeAst(domain2)
                writeAst(range)
            }
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>): Int {
            val domain = sort.domainSorts.map { it.serializeSort() }
            val range = sort.range.serializeSort()
            return serializeSort(sort, SortKind.ArrayN) {
                writeAstArray(domain)
                writeAst(range)
            }
        }

        override fun visit(sort: KFpRoundingModeSort): Int = serializeSort(sort, SortKind.FpRM) {}

        override fun visit(sort: KUninterpretedSort): Int = serializeSort(sort, SortKind.Uninterpreted) {
            writeString(sort.name)
        }

        private inline fun serializeSort(sort: KSort, kind: SortKind, sortArgs: AbstractBuffer.() -> Unit): Int =
            sort.serializeAst {
                output.writeEnum(AstKind.Sort)
                output.writeEnum(kind)
                sortArgs()
            }
    }

    private inner class DeclSerializer : KDeclVisitor<Int> {
        override fun <S : KSort> visit(decl: KFuncDecl<S>): Int {
            val args = decl.argSorts.map { it.serializeSort() }
            val sort = decl.sort.serializeSort()
            return decl.serializeAst {
                writeEnum(AstKind.Decl)
                writeString(decl.name)
                writeAstArray(args)
                writeAst(sort)
            }
        }
    }

    private fun AbstractBuffer.writeAst(idx: Int) {
        writeInt(idx)
    }

    private fun AbstractBuffer.writeAstArray(asts: List<Int>) {
        val indices = asts.toIntArray()
        writeIntArray(indices)
    }

    private fun AbstractBuffer.writeAstArray(asts: Array<Int>) {
        val indices = asts.toIntArray()
        writeIntArray(indices)
    }

    private inline fun KAst.serializeAst(body: AbstractBuffer.() -> Unit): Int {
        val idx = serializationCtx.mkAstIdx(this)
        output.writeInt(idx)
        output.body()
        output.writeInt(AstSerializationCtx.SERIALIZED_AST_ENTRY_END)
        return idx
    }

    private inline fun KExpr<*>.writeExpr(argWriter: AbstractBuffer.() -> Unit): Int = serializeAst {
        val kind = exprKindMapper.getKind(this@writeExpr)
        writeEnum(AstKind.Expr)
        writeEnum(kind)
        argWriter()
    }

    private fun <S : KExpr<*>> S.serialize(): S = transform { writeExpr { } }

    private fun <S : KExpr<*>> S.serialize(
        arg: KExpr<*>
    ): S = transform(arg) { argIdx: Int ->
        writeExpr {
            writeAst(argIdx)
        }
    }

    private fun <S : KExpr<*>> S.serialize(
        arg0: KExpr<*>,
        arg1: KExpr<*>
    ): S = transform(arg0, arg1) { a0: Int, a1: Int ->
        writeExpr {
            writeAst(a0)
            writeAst(a1)
        }
    }

    private fun <S : KExpr<*>> S.serialize(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>
    ): S = transform(arg0, arg1, arg2) { a0: Int, a1: Int, a2: Int ->
        writeExpr {
            writeAst(a0)
            writeAst(a1)
            writeAst(a2)
        }
    }

    override fun <T : KSort> transform(expr: KFunctionApp<T>) = with(expr) {
        transformList(args) { args: Array<Int> ->
            val declIdx = decl.serializeDecl()
            writeExpr {
                writeAst(declIdx)
                writeAstArray(args)
            }
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = with(expr) {
        transform {
            val declIdx = decl.serializeDecl()
            writeExpr {
                writeAst(declIdx)
            }
        }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformList(args) { args: Array<Int> ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformList(args) { args: Array<Int> ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun transform(expr: KNotExpr) = with(expr) { serialize(arg) }

    override fun transform(expr: KImpliesExpr) = with(expr) { serialize(p, q) }

    override fun transform(expr: KXorExpr) = with(expr) { serialize(a, b) }

    override fun transform(expr: KTrue) = with(expr) { serialize() }

    override fun transform(expr: KFalse) = with(expr) { serialize() }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = with(expr) { serialize(lhs, rhs) }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformList(args) { args: Array<Int> ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        serialize(condition, trueBranch, falseBranch)
    }

    override fun transform(expr: KBitVec1Value) = with(expr) {
        transform { writeExpr { writeBoolean(value) } }
    }

    override fun transform(expr: KBitVec8Value) = with(expr) {
        transform { writeExpr { writeByte(numberValue) } }
    }

    override fun transform(expr: KBitVec16Value) = with(expr) {
        transform { writeExpr { writeShort(numberValue) } }
    }

    override fun transform(expr: KBitVec32Value) = with(expr) {
        transform { writeExpr { writeInt(numberValue) } }
    }

    override fun transform(expr: KBitVec64Value) = with(expr) {
        transform { writeExpr { writeLong(numberValue) } }
    }

    override fun transform(expr: KBitVecCustomValue) = with(expr) {
        transform {
            writeExpr {
                writeBigInteger(value)
                writeUInt(sort.sizeBits)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>) =
        with(expr) { serialize(value) }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>) =
        with(expr) { serialize(value) }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>) =
        with(expr) { serialize(value) }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>) =
        with(expr) { serialize(value) }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> =
        with(expr) { serialize(arg0, arg1) }

    override fun transform(expr: KBvExtractExpr) = with(expr) {
        transform(value) { value: Int ->
            writeExpr {
                writeInt(high)
                writeInt(low)
                writeAst(value)
            }
        }
    }

    override fun transform(expr: KBvSignExtensionExpr) = with(expr) {
        transform(value) { value: Int ->
            writeExpr {
                writeInt(extensionSize)
                writeAst(value)
            }
        }
    }

    override fun transform(expr: KBvZeroExtensionExpr) = with(expr) {
        transform(value) { value: Int ->
            writeExpr {
                writeInt(extensionSize)
                writeAst(value)
            }
        }
    }

    override fun transform(expr: KBvRepeatExpr) = with(expr) {
        transform(value) { value: Int ->
            writeExpr {
                writeInt(repeatNumber)
                writeAst(value)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) =
        with(expr) { serialize(arg, shift) }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) =
        with(expr) { serialize(arg, shift) }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) =
        with(expr) { serialize(arg, shift) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) =
        with(expr) { serialize(arg, rotation) }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { value: Int ->
            writeExpr {
                writeInt(rotationNumber)
                writeAst(value)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) =
        with(expr) { serialize(arg, rotation) }

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transform(value) { value: Int ->
            writeExpr {
                writeInt(rotationNumber)
                writeAst(value)
            }
        }
    }

    override fun transform(expr: KBv2IntExpr) = with(expr) {
        transform(value) { value: Int ->
            writeExpr {
                writeAst(value)
                writeBoolean(isSigned)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: Int, a1: Int ->
            writeExpr {
                writeAst(a0)
                writeAst(a1)
                writeBoolean(isSigned)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1) { a0: Int, a1: Int ->
            writeExpr {
                writeAst(a0)
                writeAst(a1)
                writeBoolean(isSigned)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) =
        with(expr) { serialize(value) }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: Int, a1: Int ->
            writeExpr {
                writeAst(a0)
                writeAst(a1)
                writeBoolean(isSigned)
            }
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) =
        with(expr) { serialize(arg0, arg1) }

    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = with(expr) {
        transform { writeExpr { writeFloat(value) } }
    }

    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = with(expr) {
        transform { writeExpr { writeFloat(value) } }
    }

    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = with(expr) {
        transform { writeExpr { writeDouble(value) } }
    }

    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = with(expr) {
        transform(biasedExponent, significand) { exp: Int, significand: Int ->
            writeExpr {
                writeAst(significand)
                writeAst(exp)
                writeBoolean(signBit)
            }
        }
    }

    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = with(expr) {
        transform(biasedExponent, significand) { exp: Int, significand: Int ->
            writeExpr {
                writeUInt(significandSize)
                writeUInt(exponentSize)
                writeAst(significand)
                writeAst(exp)
                writeBoolean(signBit)
            }
        }
    }

    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> = with(expr) {
        transform { writeExpr { writeEnum(value) } }
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = with(expr) {
        serialize(roundingMode, arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = with(expr) {
        serialize(roundingMode, arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = with(expr) {
        serialize(roundingMode, arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = with(expr) {
        serialize(roundingMode, arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> =
        with(expr) {
            transform(roundingMode, arg0, arg1, arg2) { rm: Int, a0: Int, a1: Int, a2: Int ->
                writeExpr {
                    writeAst(rm)
                    writeAst(a0)
                    writeAst(a1)
                    writeAst(a2)
                }
            }
        }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> =
        with(expr) {
            serialize(roundingMode, value)
        }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        with(expr) {
            serialize(roundingMode, value)
        }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(arg0, arg1)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        with(expr) {
            transform(roundingMode, value) { rm: Int, value: Int ->
                writeExpr {
                    writeAst(rm)
                    writeAst(value)
                    writeInt(bvSize)
                    writeBoolean(isSigned)
                }
            }
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = with(expr) {
        serialize(value)
    }

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = with(expr) {
        serialize(sign, biasedExponent, significand)
    }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Int, value: Int ->
            val fpSort = sort.serializeSort()
            writeExpr {
                writeAst(fpSort)
                writeAst(rm)
                writeAst(value)
            }
        }
    }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Int, value: Int ->
            val fpSort = sort.serializeSort()
            writeExpr {
                writeAst(fpSort)
                writeAst(rm)
                writeAst(value)
            }
        }
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Int, value: Int ->
            val fpSort = sort.serializeSort()
            writeExpr {
                writeAst(fpSort)
                writeAst(rm)
                writeAst(value)
                writeBoolean(signed)
            }
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = with(expr) {
        serialize(array, index, value)
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = with(expr) {
        transform(array, index0, index1, value) { a: Int, i0: Int, i1: Int, v: Int ->
            writeExpr {
                writeAst(a)
                writeAst(i0)
                writeAst(i1)
                writeAst(v)
            }
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = with(expr) {
        transformList(listOf(array, index0, index1, index2, value)) { args: Array<Int> ->
            val (a: Int, i0: Int, i1: Int, i2: Int, v: Int) = args
            writeExpr {
                writeAst(a)
                writeAst(i0)
                writeAst(i1)
                writeAst(i2)
                writeAst(v)
            }
        }
    }

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> = with(expr) {
        transformList(indices + listOf(array, value)) { args: Array<Int> ->
            val array = args[args.lastIndex - 1]
            val value = args[args.lastIndex]
            val indices = args.dropLast(2)
            writeExpr {
                writeAst(array)
                writeAstArray(indices)
                writeAst(value)
            }
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        serialize(array, index)
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Select<D0, D1, R>
    ): KExpr<R> = with(expr) {
        serialize(array, index0, index1)
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = with(expr) {
        transform(array, index0, index1, index2) { a: Int, i0: Int, i1: Int, i2: Int ->
            writeExpr {
                writeAst(a)
                writeAst(i0)
                writeAst(i1)
                writeAst(i2)
            }
        }
    }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> = with(expr) {
        transformList(indices + array) { args: Array<Int> ->
            val array = args.last()
            val indices = args.dropLast(1)
            writeExpr {
                writeAst(array)
                writeAstArray(indices)
            }
        }
    }

    override fun <A: KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>) = with(expr) {
        transform(value) { value: Int ->
            val sortIdx = sort.serializeSort()
            writeExpr {
                writeAst(sortIdx)
                writeAst(value)
            }
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = with(expr) {
        transform(body) { body: Int ->
            val serializedIndex = indexVarDecl.serializeDecl()
            writeExpr {
                writeAst(serializedIndex)
                writeAst(body)
            }
        }
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = with(expr) {
        transform(body) { body: Int ->
            val serializedIndex0 = indexVar0Decl.serializeDecl()
            val serializedIndex1 = indexVar1Decl.serializeDecl()
            writeExpr {
                writeAst(serializedIndex0)
                writeAst(serializedIndex1)
                writeAst(body)
            }
        }
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = with(expr) {
        transform(body) { body: Int ->
            val serializedIndex0 = indexVar0Decl.serializeDecl()
            val serializedIndex1 = indexVar1Decl.serializeDecl()
            val serializedIndex2 = indexVar2Decl.serializeDecl()
            writeExpr {
                writeAst(serializedIndex0)
                writeAst(serializedIndex1)
                writeAst(serializedIndex2)
                writeAst(body)
            }
        }
    }

    override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> = with(expr) {
        transform(body) { body: Int ->
            val serializedIndices = indexVarDeclarations.map { it.serializeDecl() }
            writeExpr {
                writeAstArray(serializedIndices)
                writeAst(body)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>) = with(expr) {
        transformList(args) { args: Array<Int> ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>) = with(expr) {
        transformList(args) { args: Array<Int> ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>) = with(expr) {
        transformList(args) { args: Array<Int> ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>) = with(expr) {
        serialize(arg)
    }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>) = with(expr) {
        serialize(lhs, rhs)
    }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>) = with(expr) {
        serialize(lhs, rhs)
    }

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>) = with(expr) { serialize(lhs, rhs) }

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>) = with(expr) { serialize(lhs, rhs) }

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>) = with(expr) { serialize(lhs, rhs) }

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>) = with(expr) { serialize(lhs, rhs) }

    override fun transform(expr: KModIntExpr) = with(expr) { serialize(lhs, rhs) }

    override fun transform(expr: KRemIntExpr) = with(expr) { serialize(lhs, rhs) }

    override fun transform(expr: KToRealIntExpr) = with(expr) { serialize(arg) }

    override fun transform(expr: KInt32NumExpr) = with(expr) { transform { writeExpr { writeInt(value) } } }

    override fun transform(expr: KInt64NumExpr) = with(expr) { transform { writeExpr { writeLong(value) } } }

    override fun transform(expr: KIntBigNumExpr) = with(expr) {
        transform { writeExpr { writeString(value.toString()) } }
    }

    override fun transform(expr: KToIntRealExpr) = with(expr) { serialize(arg) }

    override fun transform(expr: KIsIntRealExpr) = with(expr) { serialize(arg) }

    override fun transform(expr: KRealNumExpr) = with(expr) {
        serialize(numerator, denominator)
    }

    override fun transform(expr: KExistentialQuantifier) = with(expr) {
        transform(body) { body: Int ->
            val bounds = bounds.map { it.serializeDecl() }
            writeExpr {
                writeAstArray(bounds)
                writeAst(body)
            }
        }
    }

    override fun transform(expr: KUniversalQuantifier) = with(expr) {
        transform(body) { body: Int ->
            val bounds = bounds.map { it.serializeDecl() }
            writeExpr {
                writeAstArray(bounds)
                writeAst(body)
            }
        }
    }

    override fun <A: KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>) = with(expr) {
        transform {
            val sortIdx = sort.serializeSort()
            val funcIdx = function.serializeDecl()
            writeExpr {
                writeAst(sortIdx)
                writeAst(funcIdx)
            }
        }
    }
}
