package org.ksmt.solver.cvc5

import io.github.cvc5.Kind
import io.github.cvc5.RoundingMode
import io.github.cvc5.Solver
import io.github.cvc5.Sort
import io.github.cvc5.Term
import io.github.cvc5.mkQuantifier
import org.ksmt.decl.KDecl
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndBinaryExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArraySelectBase
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KArrayStoreBase
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBitVecNumberValue
import org.ksmt.expr.KBitVecValue
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
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrBinaryExpr
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
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import org.ksmt.expr.rewrite.simplify.rewriteBvAddNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.rewriteBvAddNoUnderflowExpr
import org.ksmt.expr.rewrite.simplify.rewriteBvDivNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.rewriteBvMulNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.rewriteBvMulNoUnderflowExpr
import org.ksmt.expr.rewrite.simplify.rewriteBvNegNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.rewriteBvSubNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.rewriteBvSubNoUnderflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvRotateLeftExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvRotateRightExpr
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.util.KExprInternalizerBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.utils.powerOfTwo
import java.math.BigInteger

@Suppress("LargeClass")
class KCvc5ExprInternalizer(
    private val cvc5Ctx: KCvc5Context
) : KExprInternalizerBase<Term>() {

    private val sortInternalizer = KCvc5SortInternalizer(cvc5Ctx)
    private val declInternalizer = KCvc5DeclInternalizer(cvc5Ctx, sortInternalizer)

    override fun findInternalizedExpr(expr: KExpr<*>): Term? = cvc5Ctx.findInternalizedExpr(expr)

    override fun saveInternalizedExpr(expr: KExpr<*>, internalized: Term) {
        cvc5Ctx.saveInternalizedExpr(expr, internalized)
    }

    private val nsolver: Solver
        get() = cvc5Ctx.nativeSolver

    private val zeroIntValueTerm: Term by lazy { nsolver.mkInteger(0L) }

    fun <T : KDecl<*>> T.internalizeDecl(): Term = accept(declInternalizer)

    fun <T : KSort> T.internalizeSort(): Sort = accept(sortInternalizer)

    fun KExpr<KBoolSort>.internalizeAssertion(): Term {
        val (term, axiom) = internalizeWithAxiom(this)
        return if (axiom == null) {
            term
        } else {
            joinWithAxiom(term, axiom)
        }
    }

    fun <T : KSort> KExpr<T>.internalizeWithNoAxiomsAllowed(): Term {
        val (term, axiom) = internalizeWithAxiom(this)
        if (axiom != null) {
            throw KSolverUnsupportedFeatureException("Axioms are not allowed")
        }
        return term
    }

    fun internalizeWithAxiom(expr: KExpr<*>): Pair<Term, KCvc5Context.ExprAxiom?> = try {
        val term = expr.internalizeExpr()
        val axiom = cvc5Ctx.findExpressionAxiom(expr)
        term to axiom
    } finally {
        exprStack.clear()
    }

    override fun <T : KSort> transform(expr: KFunctionApp<T>) = with(expr) {
        transformListWithAxioms(args) { args: Array<Term> ->
            // args[0] is a function declaration
            val decl = decl.internalizeDecl()

            if (decl.hasOp()) {
                nsolver.mkTerm(decl.op, args)
            } else {
                nsolver.mkTerm(Kind.APPLY_UF, arrayOf(decl) + args)
            }
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = with(expr) {
        transformWithAxioms {
            decl.internalizeDecl()
        }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformListWithAxioms(args) { args: Array<Term> -> nsolver.mkTerm(Kind.AND, args) }
    }

    override fun transform(expr: KAndBinaryExpr) = with(expr) {
        transformWithAxioms(lhs, rhs) { l: Term, r: Term -> nsolver.mkTerm(Kind.AND, l, r) }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformListWithAxioms(args) { args: Array<Term> -> nsolver.mkTerm(Kind.OR, args) }
    }

    override fun transform(expr: KOrBinaryExpr) = with(expr) {
        transformWithAxioms(lhs, rhs) { l: Term, r: Term -> nsolver.mkTerm(Kind.OR, l, r) }
    }

    override fun transform(expr: KNotExpr) =
        with(expr) { transformWithAxioms(arg) { arg: Term -> nsolver.mkTerm(Kind.NOT, arg) } }

    override fun transform(expr: KImpliesExpr) = with(expr) {
        transformWithAxioms(p, q) { p: Term, q: Term ->
            nsolver.mkTerm(Kind.IMPLIES, p, q)
        }
    }

    override fun transform(expr: KXorExpr) = with(expr) {
        transformWithAxioms(a, b) { a: Term, b: Term ->
            nsolver.mkTerm(Kind.XOR, a, b)
        }
    }

    override fun transform(expr: KTrue) = expr.transformWithAxioms { nsolver.mkTrue() }

    override fun transform(expr: KFalse) = expr.transformWithAxioms { nsolver.mkFalse() }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.EQUAL, lhs, rhs)
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformListWithAxioms(args) { args: Array<Term> -> nsolver.mkTerm(Kind.DISTINCT, args) }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        transformWithAxioms(
            condition, trueBranch, falseBranch
        ) { condition: Term, trueBranch: Term, falseBranch: Term ->
            condition.iteTerm(trueBranch, falseBranch)
        }
    }

    fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>) = expr.transformWithAxioms {
        val size = expr.decl.sort.sizeBits.toInt()
        when {
            expr is KBitVec1Value -> nsolver.mkBitVector(size, if (expr.value) 1L else 0L)
            // cvc5 can't create bitvector from negatives
            expr is KBitVecNumberValue<*, *> && expr.numberValue.toLong() >= 0 -> {
                nsolver.mkBitVector(size, expr.numberValue.toLong())
            }

            else -> nsolver.mkBitVector(size, expr.stringValue, 2)
        }
    }

    override fun transform(expr: KBitVec1Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec8Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec16Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec32Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVec64Value) = transformBitVecValue(expr)

    override fun transform(expr: KBitVecCustomValue) = transformBitVecValue(expr)

    override fun <T : KBvSort> transform(expr: KBvNotExpr<T>) = with(expr) {
        transformWithAxioms(value) { value: Term -> nsolver.mkTerm(Kind.BITVECTOR_NOT, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>) = with(expr) {
        transformWithAxioms(value) { value: Term -> nsolver.mkTerm(Kind.BITVECTOR_REDAND, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>) = with(expr) {
        transformWithAxioms(value) { value: Term -> nsolver.mkTerm(Kind.BITVECTOR_REDOR, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_AND, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_OR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_XOR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_NAND, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_NOR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_XNOR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_NEG, value)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ADD, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SUB, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_MULT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UDIV, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SDIV, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UREM, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SREM, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SMOD, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ULT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SLT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ULE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SLE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UGE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SGE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UGT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>) = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SGT, arg0, arg1)
        }
    }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_CONCAT, arg0, arg1)
        }
    }

    override fun transform(expr: KBvExtractExpr) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            val extractOp = nsolver.mkOp(Kind.BITVECTOR_EXTRACT, high, low)
            nsolver.mkTerm(extractOp, value)
        }
    }

    override fun transform(expr: KBvSignExtensionExpr) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            val extensionOp = nsolver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, extensionSize)
            nsolver.mkTerm(extensionOp, value)
        }
    }

    override fun transform(expr: KBvZeroExtensionExpr) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            val extensionOp = nsolver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, extensionSize)
            nsolver.mkTerm(extensionOp, value)
        }
    }

    override fun transform(expr: KBvRepeatExpr) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            val repeatOp = nsolver.mkOp(Kind.BITVECTOR_REPEAT, repeatNumber)
            nsolver.mkTerm(repeatOp, value)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) = with(expr) {
        transformWithAxioms(arg, shift) { arg: Term, shift: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SHL, arg, shift)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) = with(expr) {
        transformWithAxioms(arg, shift) { arg: Term, shift: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_LSHR, arg, shift)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) = with(expr) {
        transformWithAxioms(arg, shift) { arg: Term, shift: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ASHR, arg, shift)
        }
    }

    /*
     * we can internalize rotate expr as concat expr due to simplification,
     * otherwise we can't process rotate expr on expression
     */
    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.simplifyBvRotateLeftExpr(arg, rotation)

        if (simplifiedExpr is KBvRotateLeftExpr<*>)
            throw KSolverUnsupportedFeatureException("Rotate expr with expression argument is not supported by cvc5")

        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    /*
     * @see transform(expr: KBvRotateLeftExpr<T>)
     */
    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.simplifyBvRotateRightExpr(arg, rotation)

        if (simplifiedExpr is KBvRotateRightExpr<*>)
            throw KSolverUnsupportedFeatureException("Rotate expr with expression argument is not supported by cvc5")

        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            val rotationOp = nsolver.mkOp(Kind.BITVECTOR_ROTATE_LEFT, rotationNumber)
            nsolver.mkTerm(rotationOp, value)
        }
    }


    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            val rotationOp = nsolver.mkOp(Kind.BITVECTOR_ROTATE_RIGHT, rotationNumber)
            nsolver.mkTerm(rotationOp, value)
        }
    }

    // custom implementation
    @Suppress("MagicNumber")
    override fun transform(expr: KBv2IntExpr) = with(expr) {
        transformWithAxioms(value) { value: Term ->
            // by default, it is unsigned in cvc5
            val intTerm = nsolver.mkTerm(Kind.BITVECTOR_TO_NAT, value)

            if (isSigned) {
                val size = this.value.sort.sizeBits.toInt()
                val modulo = powerOfTwo(size.toUInt())
                val maxInt = (powerOfTwo((size - 1).toUInt())) - BigInteger.ONE

                val moduloTerm = nsolver.mkInteger(modulo.toString(10))
                val maxIntTerm = nsolver.mkInteger(maxInt.toString(10))

                val gtTerm = nsolver.mkTerm(Kind.GT, intTerm, maxIntTerm)
                val subTerm = nsolver.mkTerm(Kind.SUB, intTerm, moduloTerm)

                nsolver.mkTerm(Kind.ITE, gtTerm, subTerm, intTerm)
            } else intTerm
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvAddNoOverflowExpr(expr.arg0, expr.arg1, isSigned)
        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvAddNoUnderflowExpr(expr.arg0, expr.arg1)
        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvSubNoOverflowExpr(expr.arg0, expr.arg1)
        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvSubNoUnderflowExpr(expr.arg0, expr.arg1, isSigned)
        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvDivNoOverflowExpr(expr.arg0, expr.arg1)
        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvNegNoOverflowExpr(expr.value)
        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvMulNoOverflowExpr(expr.arg0, expr.arg1, isSigned)

        if (simplifiedExpr is KBvMulNoOverflowExpr<*>) // can't rewrite
            throw KSolverUnsupportedFeatureException("no direct support in cvc5")

        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) = with(expr) {
        val simplifiedExpr = ctx.rewriteBvMulNoUnderflowExpr(expr.arg0, expr.arg1)

        if (simplifiedExpr is KBvMulNoUnderflowExpr<*>) // can't rewrite
            throw KSolverUnsupportedFeatureException("no direct support in cvc5")

        simplifiedExpr.accept(this@KCvc5ExprInternalizer)
    }

    private fun fpToBvTerm(signBit: Boolean, biasedExp: KBitVecValue<*>, significand: KBitVecValue<*>): Term {
        val signString = if (signBit) "1" else "0"
        val expString = biasedExp.stringValue
        val significandString = significand.stringValue

        val bvString = signString + expString + significandString

        return nsolver.mkBitVector(bvString.length, bvString, 2)
    }

    private fun <T : KFpSort> KFpValue<T>.toBitvectorTerm(): Term = fpToBvTerm(signBit, biasedExponent, significand)

    private fun <T : KFpValue<*>> transformFpValue(expr: T): T = with(expr) {
        transformWithAxioms {
            val bv = expr.toBitvectorTerm()
            // created from IEEE-754 bit-vector representation of the floating-point value
            nsolver.mkFloatingPoint(sort.exponentBits.toInt(), sort.significandBits.toInt(), bv)
        }
    }

    override fun transform(expr: KFp16Value): KExpr<KFp16Sort> = transformFpValue(expr)

    override fun transform(expr: KFp32Value): KExpr<KFp32Sort> = transformFpValue(expr)

    override fun transform(expr: KFp64Value): KExpr<KFp64Sort> = transformFpValue(expr)

    override fun transform(expr: KFp128Value): KExpr<KFp128Sort> = transformFpValue(expr)

    override fun transform(expr: KFpCustomSizeValue): KExpr<KFpSort> = transformFpValue(expr)

    override fun transform(expr: KFpRoundingModeExpr): KExpr<KFpRoundingModeSort> = with(expr) {
        transformWithAxioms {
            val rmMode = when (value) {
                KFpRoundingMode.RoundNearestTiesToEven -> RoundingMode.ROUND_NEAREST_TIES_TO_EVEN
                KFpRoundingMode.RoundNearestTiesToAway -> RoundingMode.ROUND_NEAREST_TIES_TO_AWAY
                KFpRoundingMode.RoundTowardPositive -> RoundingMode.ROUND_TOWARD_POSITIVE
                KFpRoundingMode.RoundTowardNegative -> RoundingMode.ROUND_TOWARD_NEGATIVE
                KFpRoundingMode.RoundTowardZero -> RoundingMode.ROUND_TOWARD_ZERO
            }

            nsolver.mkRoundingMode(rmMode)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpAbsExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_ABS, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_NEG, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_ADD,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_SUB,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_MULT,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_DIV,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, arg0, arg1, arg2) { rm: Term, arg0: Term, arg1: Term, arg2: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_FMA,
                arrayOf(rm, arg0, arg1, arg2)
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, value) { roundingMode: Term, value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_SQRT, roundingMode, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_REM, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        with(expr) {
            transformWithAxioms(roundingMode, value) { roundingMode: Term, value: Term ->
                nsolver.mkTerm(Kind.FLOATINGPOINT_RTI, roundingMode, value)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_MIN, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_MAX, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_LEQ, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_LT, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_GEQ, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_GT, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_EQ, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_NORMAL, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_SUBNORMAL, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_ZERO, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_INF, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_NAN, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_NEG, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_POS, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        with(expr) {
            transformWithAxioms(roundingMode, value) { rm: Term, value: Term ->
                val opKind = if (isSigned) Kind.FLOATINGPOINT_TO_SBV else Kind.FLOATINGPOINT_TO_UBV
                val op = nsolver.mkOp(opKind, bvSize)
                nsolver.mkTerm(op, rm, value)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = with(expr) {
        transformWithAxioms(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_TO_REAL, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> =
        throw KSolverUnsupportedFeatureException("no direct support for $expr")

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(sign, biasedExponent, significand) { sign: Term, biasedExp: Term, significand: Term ->
            val bvTerm = nsolver.mkTerm(
                Kind.BITVECTOR_CONCAT,
                nsolver.mkTerm(Kind.BITVECTOR_CONCAT, sign, biasedExp),
                significand
            )

            val toFpOp = nsolver.mkOp(
                Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )

            nsolver.mkTerm(toFpOp, bvTerm)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpToFpExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, value) { rm: Term, value: Term ->
            val op = nsolver.mkOp(
                Kind.FLOATINGPOINT_TO_FP_FROM_FP,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, value) { rm: Term, value: Term ->
            val op = nsolver.mkOp(
                Kind.FLOATINGPOINT_TO_FP_FROM_REAL,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = with(expr) {
        transformWithAxioms(roundingMode, value) { rm: Term, value: Term ->
            val opKind = if (signed) Kind.FLOATINGPOINT_TO_FP_FROM_SBV else Kind.FLOATINGPOINT_TO_FP_FROM_UBV
            val op = nsolver.mkOp(
                opKind,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = with(expr) {
        transformWithAxioms(array, index, value) { array: Term, index: Term, value: Term ->
            nsolver.mkTerm(Kind.STORE, arrayOf(array, index, value))
        }
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = expr.transformStore()

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = expr.transformStore()

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> =
        expr.transformStore()

    private fun <E : KArrayStoreBase<*, *>> E.transformStore(): E =
        transformListWithAxioms(listOf(array, value) + indices) { transformedArgs: Array<Term> ->
            val (array, value) = transformedArgs.take(2)
            val indices = transformedArgs.drop(2)
            val storeIdx = mkArrayOperationIndex(indices)
            nsolver.mkTerm(Kind.STORE, arrayOf(array, storeIdx, value))
        }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        transformWithAxioms(array, index) { array: Term, index: Term ->
            nsolver.mkTerm(Kind.SELECT, array, index)
        }
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D0, D1, R>) =
        expr.transformSelect()

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = expr.transformSelect()

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> =
        expr.transformSelect()

    private fun <E : KArraySelectBase<*, *>> E.transformSelect(): E =
        transformListWithAxioms(listOf(array) + indices) { transformedArgs: Array<Term> ->
            val array = transformedArgs.first()
            val indices = transformedArgs.drop(1)
            mkArraySelectTerm(array, indices)
        }

    private fun mkArraySelectTerm(array: Term, indices: List<Term>): Term {
        val selectIdx = mkArrayOperationIndex(indices)
        return nsolver.mkTerm(Kind.SELECT, array, selectIdx)
    }

    private fun mkArrayOperationIndex(indices: List<Term>): Term =
        if (indices.size == 1) {
            indices.single()
        } else {
            nsolver.mkTuple(indices.map { it.sort }.toTypedArray(), indices.toTypedArray())
        }

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>) = with(expr) {
        if (value is KInterpretedValue<*> || value is KArrayConst<*, *>) {
            transformWithAxioms(value) { value: Term ->
                // const array base must be a value or a constant array
                nsolver.mkConstArray(sort.internalizeSort(), value)
            }
        } else {
            transformExprWithQuantifierAxiom(
                body = value,
                bounds = sort.domainSorts.map { ctx.mkFreshConstDecl("i", it) },
                mkResultTerm = { nsolver.mkConst(sort.internalizeSort()) },
                mkQuantifiedAxiomBody = { bounds, result, body ->
                    val resultValue = mkArraySelectTerm(result, bounds)
                    nsolver.mkTerm(Kind.EQUAL, resultValue, body)
                }
            )
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) =
        expr.transformLambda()

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        expr.transformLambda()

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        expr.transformLambda()

    override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> =
        expr.transformLambda()

    private fun <E : KArrayLambdaBase<*, *>> E.transformLambda(): E =
        transformExprWithQuantifierAxiom(
            bounds = indexVarDeclarations,
            body = body,
            mkResultTerm = { nsolver.mkConst(sort.internalizeSort()) },
            mkQuantifiedAxiomBody = { bounds, result, body ->
                val resultValue = mkArraySelectTerm(result, bounds)
                nsolver.mkTerm(Kind.EQUAL, resultValue, body)
            }
        )

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>) = with(expr) {
        transformListWithAxioms(args) { args -> nsolver.mkTerm(Kind.ADD, args) }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>) = with(expr) {
        transformListWithAxioms(args) { args -> nsolver.mkTerm(Kind.SUB, args) }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>) = with(expr) {
        transformListWithAxioms(args) { args -> nsolver.mkTerm(Kind.MULT, args) }
    }

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>) = with(expr) {
        transformWithAxioms(arg) { arg: Term -> nsolver.mkTerm(Kind.NEG, arg) }
    }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term -> nsolver.mkTerm(Kind.DIVISION, lhs, rhs) }
    }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term -> nsolver.mkTerm(Kind.POW, lhs, rhs) }
    }

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.LT, lhs, rhs)
        }
    }

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.LEQ, lhs, rhs)
        }
    }

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.GT, lhs, rhs)
        }
    }

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.GEQ, lhs, rhs)
        }
    }

    override fun transform(expr: KModIntExpr) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.INTS_MODULUS, lhs, rhs)
        }
    }

    // custom implementation
    override fun transform(expr: KRemIntExpr) = with(expr) {
        transformWithAxioms(lhs, rhs) { lhs: Term, rhs: Term ->
            // there is no ints remainder in cvc5
            val remSign = nsolver.mkTerm(Kind.GEQ, lhs, zeroIntValueTerm)
                .xorTerm(nsolver.mkTerm(Kind.GEQ, rhs, zeroIntValueTerm))
            val modTerm = nsolver.mkTerm(Kind.INTS_MODULUS, lhs, rhs)

            remSign.iteTerm(nsolver.mkTerm(Kind.NEG, modTerm), modTerm)
        }
    }

    override fun transform(expr: KToRealIntExpr) = with(expr) {
        transformWithAxioms(arg) { arg: Term ->
            nsolver.mkTerm(Kind.TO_REAL, arg)
        }
    }

    override fun transform(expr: KInt32NumExpr) = with(expr) {
        transformWithAxioms { nsolver.mkInteger(expr.value.toLong()) }
    }

    override fun transform(expr: KInt64NumExpr) = with(expr) {
        // We need to pass String value here because on Windows it might be cut to 32 bit int value
        transformWithAxioms { nsolver.mkInteger(expr.value.toString()) }
    }

    override fun transform(expr: KIntBigNumExpr) = with(expr) {
        transformWithAxioms { nsolver.mkInteger(expr.value.toString()) }
    }

    override fun transform(expr: KToIntRealExpr) = with(expr) {
        transformWithAxioms(arg) { arg: Term ->
            nsolver.mkTerm(Kind.TO_INTEGER, arg)
        }
    }

    override fun transform(expr: KIsIntRealExpr) = with(expr) {
        transformWithAxioms(arg) { arg: Term ->
            nsolver.mkTerm(Kind.IS_INTEGER, arg)
        }
    }

    override fun transform(expr: KRealNumExpr) = with(expr) {
        transformWithAxioms(numerator, denominator) { num: Term, denom: Term ->
            val numAsReal = nsolver.mkTerm(Kind.TO_REAL, num)
            val denomAsReal = nsolver.mkTerm(Kind.TO_REAL, denom)

            nsolver.mkTerm(Kind.DIVISION, numAsReal, denomAsReal)
        }
    }

    override fun transform(expr: KExistentialQuantifier) =
        expr.transformQuantifiedExpression(expr.bounds, expr.body, isUniversal = false)

    override fun transform(expr: KUniversalQuantifier) =
        expr.transformQuantifiedExpression(expr.bounds, expr.body, isUniversal = true)

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> {
        throw KSolverUnsupportedFeatureException(
            "No direct impl in cvc5 (as-array is CONST_ARRAY term with base array element)"
        )
    }

    /**
     * Transform expression into term specified by [mkResultTerm]
     * guarded with an axiom of the form:
     * (forall (bounds) [mkQuantifiedAxiomBody])
     * */
    private inline fun <E : KExpr<*>> E.transformExprWithQuantifierAxiom(
        bounds: List<KDecl<*>>,
        body: KExpr<*>,
        mkResultTerm: () -> Term,
        mkQuantifiedAxiomBody: (List<Term>, Term, Term) -> Term,
    ): E = transform(body) { transformedBody: Term ->
        val boundConsts = bounds.map { it.internalizeDecl() }
        val resultTerm = mkResultTerm()

        val axiomBody = mkQuantifiedAxiomBody(boundConsts, resultTerm, transformedBody)

        val (axiomTerm, nestedAxiom) = mkQuantifierWithAxiomsResolved(
            isUniversal = true, bounds, boundConsts, axiomBody, body
        )
        val axiom = KCvc5Context.ExprAxiomInstance(this, resultTerm, axiomTerm)

        mergeAxioms(listOfNotNull(axiom, nestedAxiom))?.let { mergedAxiom ->
            cvc5Ctx.storeExpressionAxiom(this, mergedAxiom)
        }

        resultTerm
    }

    private fun <E : KExpr<*>> E.transformQuantifiedExpression(
        bounds: List<KDecl<*>>,
        body: KExpr<*>,
        isUniversal: Boolean
    ): E = transform(body) { transformedBody: Term ->
        val boundConsts = bounds.map { it.internalizeDecl() }

        val (term, nestedAxiom) = mkQuantifierWithAxiomsResolved(
            isUniversal, bounds, boundConsts, transformedBody, body
        )

        if (nestedAxiom != null) {
            cvc5Ctx.storeExpressionAxiom(this, nestedAxiom)
        }

        term
    }

    private fun mkQuantifierWithAxiomsResolved(
        isUniversal: Boolean,
        bounds: List<KDecl<*>>,
        boundTerms: List<Term>,
        body: Term,
        bodyExpr: KExpr<*>
    ): Pair<Term, KCvc5Context.ExprAxiom?> {
        val bodyAxiom = cvc5Ctx.findExpressionAxiom(bodyExpr)
        val (bodyWithAxiom, axiomToPull) = if (bodyAxiom == null) {
            body to null
        } else {
            resolveQuantifiedAxiom(bounds, body, bodyAxiom)
        }

        return mkQuantifierTerm(isUniversal, boundTerms, bodyWithAxiom) to axiomToPull
    }

    /**
     * Resolve quantifier body axiom according to the following rules:
     * 1. If an axiom does not depend on a bound variables we can pull it to the upper level.
     * 2. Otherwise, the axiom must be a part of the quantifier body. Therefore, we must existentially
     * quantify axiom free variables.
     * */
    private fun resolveQuantifiedAxiom(
        bounds: List<KDecl<*>>,
        body: Term,
        axiom: KCvc5Context.ExprAxiom
    ): Pair<Term, KCvc5Context.ExprAxiom?> {
        val boundDecls = bounds.toSet()
        val (quantifiedAxioms, independentAxioms) = axiom.flatten().partition { ax ->
            val axiomExprDecls = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(ax.expr)
            axiomExprDecls.any { it in boundDecls }
        }

        val bodyWithoutQuantifiedAxioms = if (quantifiedAxioms.isEmpty()) {
            body
        } else {
            val axiomVars = quantifiedAxioms.map { it.variable }
            val axiomTerms = quantifiedAxioms.map { it.axiom }
            mkQuantifierTerm(isUniversal = false, axiomVars, joinWithAxiom(body, axiomTerms))
        }

        val independentAxiom = mergeAxioms(independentAxioms)

        return bodyWithoutQuantifiedAxioms to independentAxiom
    }

    private fun mkQuantifierTerm(isUniversal: Boolean, bounds: List<Term>, body: Term): Term {
        val boundVars = bounds.map {
            if (it.hasSymbol()) {
                nsolver.mkVar(it.sort, it.symbol)
            } else {
                nsolver.mkVar(it.sort)
            }
        }.toTypedArray()

        val bodyWithVars = body.substitute(bounds.toTypedArray(), boundVars)
        return nsolver.mkQuantifier(isUniversal, boundVars, bodyWithVars)
    }

    private inline fun <S : KExpr<*>> S.transformWithAxioms(
        operation: () -> Term
    ): S = transform {
        operation()
    }

    private inline fun <S : KExpr<*>> S.transformWithAxioms(
        arg: KExpr<*>,
        operation: (Term) -> Term
    ): S = transform(arg) { tArg: Term ->
        propagateAxioms(listOf(arg)) {
            operation(tArg)
        }
    }

    private inline fun <S : KExpr<*>> S.transformWithAxioms(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        operation: (Term, Term) -> Term
    ): S = transform(arg0, arg1) { a0: Term, a1: Term ->
        propagateAxioms(listOf(arg0, arg1)) {
            operation(a0, a1)
        }
    }

    private inline fun <S : KExpr<*>> S.transformWithAxioms(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        operation: (Term, Term, Term) -> Term
    ): S = transform(arg0, arg1, arg2) { a0: Term, a1: Term, a2: Term ->
        propagateAxioms(listOf(arg0, arg1, arg2)) {
            operation(a0, a1, a2)
        }
    }

    private inline fun <S : KExpr<*>> S.transformWithAxioms(
        arg0: KExpr<*>,
        arg1: KExpr<*>,
        arg2: KExpr<*>,
        arg3: KExpr<*>,
        operation: (Term, Term, Term, Term) -> Term
    ): S = transform(arg0, arg1, arg2, arg3) { a0: Term, a1: Term, a2: Term, a3: Term ->
        propagateAxioms(listOf(arg0, arg1, arg2, arg3)) {
            operation(a0, a1, a2, a3)
        }
    }

    private inline fun <S : KExpr<*>> S.transformListWithAxioms(
        args: List<KExpr<*>>,
        operation: (Array<Term>) -> Term
    ): S = transformList(args) { tArgs: Array<Term> ->
        propagateAxioms(args) {
            operation(tArgs)
        }
    }

    /**
     * Default axioms propagation.
     *
     * Propagation rules:
     * 1. If all expression arguments have no axiom then expression also have no axiom.
     * 2. If any of expression arguments has an axiom then expression axiom is a conjunction of arguments axioms.
     * */
    private inline fun KExpr<*>.propagateAxioms(
        args: List<KExpr<*>>,
        op: () -> Term
    ): Term {
        val result = op()

        val argumentsAxioms = args.mapNotNull { cvc5Ctx.findExpressionAxiom(it) }
        if (argumentsAxioms.isEmpty()) {
            return result
        }

        // axioms are already handled by the expression transformer
        if (cvc5Ctx.expressionHasAxiom(this)) {
            return result
        }

        mergeAxioms(argumentsAxioms)?.let { mergedAxiom ->
            cvc5Ctx.storeExpressionAxiom(this, mergedAxiom)
        }

        return result
    }

    private fun joinWithAxiom(term: Term, axiom: KCvc5Context.ExprAxiom): Term =
        joinWithAxiom(term, axiom.flatten().map { it.axiom })

    private fun joinWithAxiom(term: Term, axioms: List<Term>): Term {
        val termsToMerge = axioms + listOf(term)
        return nsolver.mkTerm(Kind.AND, termsToMerge.toTypedArray())
    }

    private fun mergeAxioms(axioms: List<KCvc5Context.ExprAxiom>): KCvc5Context.ExprAxiom? = when (axioms.size) {
        0 -> null
        1 -> axioms.single()
        else -> KCvc5Context.ExprAxiomMerge(axioms)
    }
}
