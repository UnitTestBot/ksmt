package io.ksmt.runner.serializer

import com.jetbrains.rd.framework.AbstractBuffer
import com.jetbrains.rd.framework.writeEnum
import io.ksmt.KAst
import io.ksmt.decl.KDecl
import io.ksmt.decl.KDeclVisitor
import io.ksmt.decl.KFuncDecl
import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBv2IntExpr
import io.ksmt.expr.KBvAddExpr
import io.ksmt.expr.KBvAddNoOverflowExpr
import io.ksmt.expr.KBvAddNoUnderflowExpr
import io.ksmt.expr.KBvAndExpr
import io.ksmt.expr.KBvArithShiftRightExpr
import io.ksmt.expr.KBvConcatExpr
import io.ksmt.expr.KBvDivNoOverflowExpr
import io.ksmt.expr.KBvExtractExpr
import io.ksmt.expr.KBvLogicalShiftRightExpr
import io.ksmt.expr.KBvMulExpr
import io.ksmt.expr.KBvMulNoOverflowExpr
import io.ksmt.expr.KBvMulNoUnderflowExpr
import io.ksmt.expr.KBvNAndExpr
import io.ksmt.expr.KBvNegNoOverflowExpr
import io.ksmt.expr.KBvNegationExpr
import io.ksmt.expr.KBvNorExpr
import io.ksmt.expr.KBvNotExpr
import io.ksmt.expr.KBvOrExpr
import io.ksmt.expr.KBvReductionAndExpr
import io.ksmt.expr.KBvReductionOrExpr
import io.ksmt.expr.KBvRepeatExpr
import io.ksmt.expr.KBvRotateLeftExpr
import io.ksmt.expr.KBvRotateLeftIndexedExpr
import io.ksmt.expr.KBvRotateRightExpr
import io.ksmt.expr.KBvRotateRightIndexedExpr
import io.ksmt.expr.KBvShiftLeftExpr
import io.ksmt.expr.KBvSignExtensionExpr
import io.ksmt.expr.KBvSignedDivExpr
import io.ksmt.expr.KBvSignedGreaterExpr
import io.ksmt.expr.KBvSignedGreaterOrEqualExpr
import io.ksmt.expr.KBvSignedLessExpr
import io.ksmt.expr.KBvSignedLessOrEqualExpr
import io.ksmt.expr.KBvSignedModExpr
import io.ksmt.expr.KBvSignedRemExpr
import io.ksmt.expr.KBvSubExpr
import io.ksmt.expr.KBvSubNoOverflowExpr
import io.ksmt.expr.KBvSubNoUnderflowExpr
import io.ksmt.expr.KBvToFpExpr
import io.ksmt.expr.KBvUnsignedDivExpr
import io.ksmt.expr.KBvUnsignedGreaterExpr
import io.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.KBvUnsignedLessExpr
import io.ksmt.expr.KBvUnsignedLessOrEqualExpr
import io.ksmt.expr.KBvUnsignedRemExpr
import io.ksmt.expr.KBvXNorExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KBvZeroExtensionExpr
import io.ksmt.expr.KConst
import io.ksmt.expr.KDistinctExpr
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KFp128Value
import io.ksmt.expr.KFp16Value
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
import io.ksmt.expr.KFpCustomSizeValue
import io.ksmt.expr.KFpDivExpr
import io.ksmt.expr.KFpEqualExpr
import io.ksmt.expr.KFpFromBvExpr
import io.ksmt.expr.KFpFusedMulAddExpr
import io.ksmt.expr.KFpGreaterExpr
import io.ksmt.expr.KFpGreaterOrEqualExpr
import io.ksmt.expr.KFpIsInfiniteExpr
import io.ksmt.expr.KFpIsNaNExpr
import io.ksmt.expr.KFpIsNegativeExpr
import io.ksmt.expr.KFpIsNormalExpr
import io.ksmt.expr.KFpIsPositiveExpr
import io.ksmt.expr.KFpIsSubnormalExpr
import io.ksmt.expr.KFpIsZeroExpr
import io.ksmt.expr.KFpLessExpr
import io.ksmt.expr.KFpLessOrEqualExpr
import io.ksmt.expr.KFpMaxExpr
import io.ksmt.expr.KFpMinExpr
import io.ksmt.expr.KFpMulExpr
import io.ksmt.expr.KFpNegationExpr
import io.ksmt.expr.KFpRemExpr
import io.ksmt.expr.KFpRoundToIntegralExpr
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIsIntRealExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KLeArithExpr
import io.ksmt.expr.KLtArithExpr
import io.ksmt.expr.KModIntExpr
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.KSubArithExpr
import io.ksmt.expr.KToIntRealExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KTrue
import io.ksmt.expr.KUnaryMinusArithExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.KXorExpr
import io.ksmt.solver.util.KExprIntInternalizerBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort

@Suppress("LargeClass")
@OptIn(ExperimentalUnsignedTypes::class)
class AstSerializer(
    private val serializationCtx: AstSerializationCtx,
    private val output: AbstractBuffer
) : KExprIntInternalizerBase() {

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
        if (idx != NOT_INTERNALIZED) return idx
        return declSerializer.serializeDecl(this)
    }

    private fun <T : KSort> T.serializeSort(): Int {
        val idx = serializationCtx.getAstIndex(this)
        if (idx != NOT_INTERNALIZED) return idx
        return sortSerializer.serializeSort(this)
    }

    override fun findInternalizedExpr(expr: KExpr<*>): Int =
        serializationCtx.getAstIndex(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: Int) {
        // Do nothing since expr is already saved into serializationCtx
    }

    private val sortSerializer = SortSerializer()
    private val declSerializer = DeclSerializer()

    @OptIn(ExperimentalUnsignedTypes::class)
    private inner class SortSerializer : KSortVisitor<Unit> {
        private var serializedSort: Int = NOT_INTERNALIZED

        override fun visit(sort: KBoolSort) {
            serializeSort(sort, SortKind.Bool) {}
        }

        override fun visit(sort: KIntSort) {
            serializeSort(sort, SortKind.Int) {}
        }

        override fun visit(sort: KRealSort) {
            serializeSort(sort, SortKind.Real) {}
        }

        override fun <S : KBvSort> visit(sort: S) {
            serializeSort(sort, SortKind.Bv) {
                writeUInt(sort.sizeBits)
            }
        }

        override fun <S : KFpSort> visit(sort: S) {
            serializeSort(sort, SortKind.Fp) {
                writeUInt(sort.exponentBits)
                writeUInt(sort.significandBits)
            }
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            val domain = sort.domain.serializeSort()
            val range = sort.range.serializeSort()
            serializeSort(sort, SortKind.Array) {
                writeAst(domain)
                writeAst(range)
            }
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
            val domain0 = sort.domain0.serializeSort()
            val domain1 = sort.domain1.serializeSort()
            val range = sort.range.serializeSort()
            serializeSort(sort, SortKind.Array2) {
                writeAst(domain0)
                writeAst(domain1)
                writeAst(range)
            }
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
            val domain0 = sort.domain0.serializeSort()
            val domain1 = sort.domain1.serializeSort()
            val domain2 = sort.domain2.serializeSort()
            val range = sort.range.serializeSort()
            serializeSort(sort, SortKind.Array3) {
                writeAst(domain0)
                writeAst(domain1)
                writeAst(domain2)
                writeAst(range)
            }
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>) {
            val domain = sort.domainSorts.let { sorts ->
                IntArray(sorts.size) { sorts[it].serializeSort() }
            }
            val range = sort.range.serializeSort()
            serializeSort(sort, SortKind.ArrayN) {
                writeAstArray(domain)
                writeAst(range)
            }
        }

        override fun visit(sort: KFpRoundingModeSort) {
            serializeSort(sort, SortKind.FpRM) {}
        }

        override fun visit(sort: KUninterpretedSort) {
            serializeSort(sort, SortKind.Uninterpreted) {
                writeString(sort.name)
            }
        }

        private inline fun serializeSort(sort: KSort, kind: SortKind, sortArgs: AbstractBuffer.() -> Unit) {
            serializedSort = sort.serializeAst {
                output.writeEnum(AstKind.Sort)
                output.writeEnum(kind)
                sortArgs()
            }
        }

        fun serializeSort(sort: KSort): Int {
            sort.accept(this)
            return serializedSort
        }
    }

    private inner class DeclSerializer : KDeclVisitor<Unit> {
        private var serializedDecl: Int = NOT_INTERNALIZED

        override fun <S : KSort> visit(decl: KFuncDecl<S>) {
            val args = decl.argSorts.let { argSorts ->
                IntArray(argSorts.size) { argSorts[it].serializeSort() }
            }
            val sort = decl.sort.serializeSort()

            serializedDecl = decl.serializeAst {
                writeEnum(AstKind.Decl)
                writeString(decl.name)
                writeAstArray(args)
                writeAst(sort)
            }
        }

        fun serializeDecl(decl: KDecl<*>): Int {
            decl.accept(this)
            return serializedDecl
        }
    }

    private fun AbstractBuffer.writeAst(idx: Int) {
        writeInt(idx)
    }

    private fun AbstractBuffer.writeAstArray(asts: List<Int>) {
        val indices = asts.toIntArray()
        writeIntArray(indices)
    }

    private fun AbstractBuffer.writeAstArray(asts: IntArray) {
        writeIntArray(asts)
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
        transformList(args) { args: IntArray ->
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
        transformList(args) { args: IntArray ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun transform(expr: KAndBinaryExpr) = with(expr) { serialize(lhs, rhs) }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformList(args) { args: IntArray ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun transform(expr: KOrBinaryExpr) = with(expr) { serialize(lhs, rhs) }

    override fun transform(expr: KNotExpr) = with(expr) { serialize(arg) }

    override fun transform(expr: KImpliesExpr) = with(expr) { serialize(p, q) }

    override fun transform(expr: KXorExpr) = with(expr) { serialize(a, b) }

    override fun transform(expr: KTrue) = with(expr) { serialize() }

    override fun transform(expr: KFalse) = with(expr) { serialize() }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = with(expr) { serialize(lhs, rhs) }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformList(args) { args: IntArray ->
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
        transform { writeExpr { writeByte(byteValue) } }
    }

    override fun transform(expr: KBitVec16Value) = with(expr) {
        transform { writeExpr { writeShort(shortValue) } }
    }

    override fun transform(expr: KBitVec32Value) = with(expr) {
        transform { writeExpr { writeInt(intValue) } }
    }

    override fun transform(expr: KBitVec64Value) = with(expr) {
        transform { writeExpr { writeLong(longValue) } }
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
        transformList(listOf(array, index0, index1, index2, value)) { args: IntArray ->
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
        transformList(indices + listOf(array, value)) { args: IntArray ->
            val array = args[args.lastIndex - 1]
            val value = args[args.lastIndex]
            val indices = args.copyOf(args.size - 2)
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
        transformList(indices + array) { args: IntArray ->
            val array = args.last()
            val indices = args.copyOf(args.size - 1)
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
        transformList(args) { args: IntArray ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>) = with(expr) {
        transformList(args) { args: IntArray ->
            writeExpr {
                writeAstArray(args)
            }
        }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>) = with(expr) {
        transformList(args) { args: IntArray ->
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

    override fun transform(expr: KUninterpretedSortValue) = with(expr) {
        transform {
            val sortIdx = sort.serializeSort()
            writeExpr {
                writeAst(sortIdx)
                writeAst(valueIdx)
            }
        }
    }
}
