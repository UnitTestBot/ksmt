package org.ksmt.solver.z3

import com.microsoft.z3.ArithExpr
import com.microsoft.z3.ArrayExpr
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
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDeclVisitor
import org.ksmt.decl.KEqDecl
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
import org.ksmt.decl.KTrueDecl
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KSort

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

    private fun KSort.sample(): Expr = z3Ctx.mkConst("e", accept(sortInternalizer))
}
