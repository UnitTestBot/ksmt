package org.ksmt.solver.z3

import com.microsoft.z3.ArithExpr
import com.microsoft.z3.ArrayExpr
import com.microsoft.z3.BitVecExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.IntExpr
import com.microsoft.z3.RealExpr
import org.ksmt.decl.KAndDecl
import org.ksmt.decl.KArithAddDecl
import org.ksmt.decl.KArithDivDecl
import org.ksmt.decl.KArithGeDecl
import org.ksmt.decl.KArithGtDecl
import org.ksmt.decl.KArithLeDecl
import org.ksmt.decl.KArithLtDecl
import org.ksmt.decl.KArithMulDecl
import org.ksmt.decl.KArithPowerDecl
import org.ksmt.decl.KArithSubDecl
import org.ksmt.decl.KArithUnaryMinusDecl
import org.ksmt.decl.KArraySelectDecl
import org.ksmt.decl.KArrayStoreDecl
import org.ksmt.decl.KBitVec16ValueDecl
import org.ksmt.decl.KBitVec1ValueDecl
import org.ksmt.decl.KBitVec32ValueDecl
import org.ksmt.decl.KBitVec64ValueDecl
import org.ksmt.decl.KBitVec8ValueDecl
import org.ksmt.decl.KBitVecCustomSizeValueDecl
import org.ksmt.decl.KBitVecValueDecl
import org.ksmt.decl.KBv2IntDecl
import org.ksmt.decl.KBvAddDecl
import org.ksmt.decl.KBvAddNoOverflowDecl
import org.ksmt.decl.KBvAddNoUnderflowDecl
import org.ksmt.decl.KBvAndDecl
import org.ksmt.decl.KBvArithShiftRightDecl
import org.ksmt.decl.KBvDivNoOverflowDecl
import org.ksmt.decl.KBvLogicalShiftRightDecl
import org.ksmt.decl.KBvMulDecl
import org.ksmt.decl.KBvMulNoOverflowDecl
import org.ksmt.decl.KBvMulNoUnderflowDecl
import org.ksmt.decl.KBvNAndDecl
import org.ksmt.decl.KBvNegNoOverflowDecl
import org.ksmt.decl.KBvNegationDecl
import org.ksmt.decl.KBvNorDecl
import org.ksmt.decl.KBvNotDecl
import org.ksmt.decl.KBvOrDecl
import org.ksmt.decl.KBvReductionAndDecl
import org.ksmt.decl.KBvReductionOrDecl
import org.ksmt.decl.KBvRotateLeftDecl
import org.ksmt.decl.KBvRotateLeftIndexedDecl
import org.ksmt.decl.KBvRotateRightDecl
import org.ksmt.decl.KBvRotateRightIndexedDecl
import org.ksmt.decl.KBvShiftLeftDecl
import org.ksmt.decl.KBvSignedDivDecl
import org.ksmt.decl.KBvSignedGreaterDecl
import org.ksmt.decl.KBvSignedGreaterOrEqualDecl
import org.ksmt.decl.KBvSignedLessDecl
import org.ksmt.decl.KBvSignedLessOrEqualDecl
import org.ksmt.decl.KBvSignedModDecl
import org.ksmt.decl.KBvSignedRemDecl
import org.ksmt.decl.KBvSubDecl
import org.ksmt.decl.KBvSubNoOverflowDecl
import org.ksmt.decl.KBvSubNoUnderflowDecl
import org.ksmt.decl.KBvUnsignedDivDecl
import org.ksmt.decl.KBvUnsignedGreaterDecl
import org.ksmt.decl.KBvUnsignedGreaterOrEqualDecl
import org.ksmt.decl.KBvUnsignedLessDecl
import org.ksmt.decl.KBvUnsignedLessOrEqualDecl
import org.ksmt.decl.KBvUnsignedRemDecl
import org.ksmt.decl.KBvXNorDecl
import org.ksmt.decl.KBvXorDecl
import org.ksmt.decl.KBvConcatDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KEqDecl
import org.ksmt.decl.KBvExtractDecl
import org.ksmt.decl.KFalseDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.decl.KIntModDecl
import org.ksmt.decl.KIntNumDecl
import org.ksmt.decl.KIntRemDecl
import org.ksmt.decl.KIntToRealDecl
import org.ksmt.decl.KIteDecl
import org.ksmt.decl.KNotDecl
import org.ksmt.decl.KOrDecl
import org.ksmt.decl.KRealIsIntDecl
import org.ksmt.decl.KRealNumDecl
import org.ksmt.decl.KRealToIntDecl
import org.ksmt.decl.KBvRepeatDecl
import org.ksmt.decl.KSignExtDecl
import org.ksmt.decl.KTrueDecl
import org.ksmt.decl.KZeroExtDecl
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort
import org.ksmt.utils.cast

@Suppress("TooManyFunctions")
open class KZ3DeclInternalizer(
    private val z3Ctx: Context,
    private val z3InternCtx: KZ3InternalizationContext,
    private val sortInternalizer: KZ3SortInternalizer
) : KDeclVisitor<FuncDecl> {
    override fun <S : KSort> visit(decl: KFuncDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        val argSorts = decl.argSorts.map { it.accept(sortInternalizer) }.toTypedArray()
        z3Ctx.mkFuncDecl(decl.name, argSorts, decl.sort.accept(sortInternalizer))
    }

    override fun <S : KSort> visit(decl: KConstDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkConstDecl(decl.name, decl.sort.accept(sortInternalizer))
    }

    @Suppress("ForbiddenComment")
    /* TODO: there is no way in Z3 api to create FuncDecl for builtin functions.
     *  To overcome this we create a sample expression.
     */
    override fun visit(decl: KAndDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkAnd(z3Ctx.mkTrue()).funcDecl
    }

    override fun visit(decl: KOrDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkOr(z3Ctx.mkTrue()).funcDecl
    }

    override fun visit(decl: KNotDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkNot(z3Ctx.mkTrue()).funcDecl
    }

    override fun visit(decl: KFalseDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkFalse().funcDecl
    }

    override fun visit(decl: KTrueDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkTrue().funcDecl
    }

    override fun <S : KSort> visit(decl: KEqDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        val expr = decl.arg0Sort.sample()
        z3Ctx.mkEq(expr, expr).funcDecl
    }

    override fun <S : KSort> visit(decl: KIteDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        val expr = decl.arg0Sort.sample()
        z3Ctx.mkITE(z3Ctx.mkTrue(), expr, expr).funcDecl
    }

    override fun <D : KSort, R : KSort> visit(decl: KArraySelectDecl<D, R>): FuncDecl =
        z3InternCtx.internalizeDecl(decl) {
            z3Ctx.mkSelect(decl.arg0Sort.sample() as ArrayExpr, decl.arg1Sort.sample()).funcDecl
        }

    override fun <D : KSort, R : KSort> visit(decl: KArrayStoreDecl<D, R>): FuncDecl =
        z3InternCtx.internalizeDecl(decl) {
            z3Ctx.mkStore(decl.arg0Sort.sample() as ArrayExpr, decl.arg1Sort.sample(), decl.arg2Sort.sample()).funcDecl
        }

    override fun <S : KArithSort<S>> visit(decl: KArithAddDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkAdd(decl.argSort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithSubDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkSub(decl.argSort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithMulDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkMul(decl.argSort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithUnaryMinusDecl<S>): FuncDecl =
        z3InternCtx.internalizeDecl(decl) {
            z3Ctx.mkUnaryMinus(decl.argSort.sample() as ArithExpr).funcDecl
        }

    override fun <S : KArithSort<S>> visit(decl: KArithPowerDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkPower(decl.arg0Sort.sample() as ArithExpr, decl.arg1Sort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithDivDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkDiv(decl.arg0Sort.sample() as ArithExpr, decl.arg1Sort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithGeDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkGe(decl.arg0Sort.sample() as ArithExpr, decl.arg1Sort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithGtDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkGt(decl.arg0Sort.sample() as ArithExpr, decl.arg1Sort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithLeDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkLe(decl.arg0Sort.sample() as ArithExpr, decl.arg1Sort.sample() as ArithExpr).funcDecl
    }

    override fun <S : KArithSort<S>> visit(decl: KArithLtDecl<S>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkLt(decl.arg0Sort.sample() as ArithExpr, decl.arg1Sort.sample() as ArithExpr).funcDecl
    }

    override fun visit(decl: KIntModDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkMod(decl.arg0Sort.sample() as IntExpr, decl.arg1Sort.sample() as IntExpr).funcDecl
    }

    override fun visit(decl: KIntRemDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkRem(decl.arg0Sort.sample() as IntExpr, decl.arg1Sort.sample() as IntExpr).funcDecl
    }

    override fun visit(decl: KIntToRealDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkInt2Real(decl.argSort.sample() as IntExpr).funcDecl
    }

    override fun visit(decl: KIntNumDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkInt(decl.value).funcDecl
    }

    override fun visit(decl: KRealIsIntDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkIsInteger(decl.argSort.sample() as RealExpr).funcDecl
    }

    override fun visit(decl: KRealToIntDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkReal2Int(decl.argSort.sample() as RealExpr).funcDecl
    }

    override fun visit(decl: KRealNumDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkReal(decl.value).funcDecl
    }

    override fun visit(decl: KBitVec1ValueDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBV(decl.value, decl.sort.sizeBits.toInt()).funcDecl
    }

    override fun visit(decl: KBitVec8ValueDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBV(decl.byteValue.toInt(), decl.sort.sizeBits.toInt()).funcDecl
    }

    override fun visit(decl: KBitVec16ValueDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBV(decl.shortValue.toInt(), decl.sort.sizeBits.toInt()).funcDecl
    }

    override fun visit(decl: KBitVec32ValueDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBV(decl.intValue, decl.sort.sizeBits.toInt()).funcDecl
    }

    override fun visit(decl: KBitVec64ValueDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBV(decl.longValue, decl.sort.sizeBits.toInt()).funcDecl
    }

    override fun visit(decl: KBitVecCustomSizeValueDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBV(decl.value, decl.sort.sizeBits.toInt()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvNotDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVNot(decl.argSort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvReductionAndDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVRedAND(decl.argSort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvReductionOrDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVRedOR(decl.argSort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvAndDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVAND(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvOrDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVOR(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvXorDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVXOR(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvNAndDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVNAND(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvNorDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVNOR(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvXNorDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVXNOR(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvNegationDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVNeg(decl.argSort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvAddDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVAdd(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSubDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSub(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvMulDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVMul(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvUnsignedDivDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVUDiv(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSignedDivDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSDiv(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvUnsignedRemDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVURem(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSignedRemDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSRem(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSignedModDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSMod(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvUnsignedLessDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVULT(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSignedLessDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSLT(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSignedLessOrEqualDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSLE(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvUnsignedLessOrEqualDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVULE(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvUnsignedGreaterOrEqualDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVUGE(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSignedGreaterOrEqualDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSGE(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvUnsignedGreaterDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVUGT(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl

    }

    override fun <T: KBvSort> visit(decl: KBvSignedGreaterDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSGT(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun visit(decl: KBvConcatDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkConcat(decl.arg0Sort.sample() as BitVecExpr, decl.arg1Sort.sample() as BitVecExpr).funcDecl
    }

    override fun visit(decl: KBvExtractDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkExtract(decl.high, decl.low, decl.argSort.sample().cast()).funcDecl
    }

    override fun visit(decl: KSignExtDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkSignExt(decl.i, decl.argSort.sample().cast()).funcDecl
    }

    override fun visit(decl: KZeroExtDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkZeroExt(decl.i, decl.argSort.sample().cast()).funcDecl
    }

    override fun visit(decl: KBvRepeatDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkRepeat(decl.i, decl.argSort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvShiftLeftDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSHL(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvLogicalShiftRightDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVLSHR(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvArithShiftRightDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVASHR(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvRotateLeftDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVRotateLeft(decl.arg0Sort.sample() as BitVecExpr, decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvRotateLeftIndexedDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVRotateLeft(decl.i, decl.argSort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvRotateRightDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVRotateRight(decl.arg0Sort.sample() as BitVecExpr, decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvRotateRightIndexedDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVRotateRight(decl.i, decl.argSort.sample().cast()).funcDecl
    }

    override fun visit(decl: KBv2IntDecl): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBV2Int(decl.argSort.sample().cast(), decl.isSigned).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvAddNoOverflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVAddNoOverflow(
            decl.arg0Sort.sample().cast(),
            decl.arg1Sort.sample().cast(),
            decl.isSigned
        ).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvAddNoUnderflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVAddNoUnderflow(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSubNoOverflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSubNoOverflow(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvSubNoUnderflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSubNoUnderflow(
            decl.arg0Sort.sample().cast(),
            decl.arg1Sort.sample().cast(),
            decl.isSigned
        ).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvDivNoOverflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVSDivNoOverflow(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvNegNoOverflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVNegNoOverflow(decl.argSort.sample().cast()).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvMulNoOverflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVMulNoOverflow(
            decl.arg0Sort.sample().cast(),
            decl.arg1Sort.sample().cast(),
            decl.isSigned
        ).funcDecl
    }

    override fun <T: KBvSort> visit(decl: KBvMulNoUnderflowDecl<T>): FuncDecl = z3InternCtx.internalizeDecl(decl) {
        z3Ctx.mkBVMulNoUnderflow(decl.arg0Sort.sample().cast(), decl.arg1Sort.sample().cast()).funcDecl
    }

    // collect this consts
    private fun KSort.sample(): Expr = z3Ctx.mkConst("e", accept(sortInternalizer))
}
