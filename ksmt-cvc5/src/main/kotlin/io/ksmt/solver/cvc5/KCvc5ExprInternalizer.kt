package io.ksmt.solver.cvc5

import io.github.cvc5.Kind
import io.github.cvc5.RoundingMode
import io.github.cvc5.Solver
import io.github.cvc5.Sort
import io.github.cvc5.Term
import io.github.cvc5.mkLambda
import io.github.cvc5.mkQuantifier
import io.ksmt.decl.KDecl
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
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArraySelectBase
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KArrayStoreBase
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBitVecNumberValue
import io.ksmt.expr.KBitVecValue
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
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KInterpretedValue
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
import io.ksmt.expr.rewrite.simplify.rewriteBvAddNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvAddNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvDivNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvNegNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvSubNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvSubNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvMulNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.rewriteBvMulNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvRotateLeftExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvRotateRightExpr
import io.ksmt.solver.KSolverUnsupportedFeatureException
import io.ksmt.solver.util.KExprInternalizerBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.powerOfTwo
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

    override fun <T : KSort> transform(expr: KFunctionApp<T>) = with(expr) {
        transformArray(args) { args: Array<Term> ->
            cvc5Ctx.addDeclaration(decl)

            // args[0] is a function declaration
            val decl = decl.internalizeDecl()

            if (decl.hasOp()) {
                nsolver.mkTerm(decl.op, args)
            } else {
                decl.mkFunctionApp(args.asList())
            }
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = with(expr) {
        transform {
            cvc5Ctx.addDeclaration(decl)

            decl.internalizeDecl()
        }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformArray(args) { args: Array<Term> -> mkAndTerm(args.asList()) }
    }

    override fun transform(expr: KAndBinaryExpr) = with(expr) {
        transform(lhs, rhs) { l: Term, r: Term -> nsolver.mkTerm(Kind.AND, l, r) }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformArray(args) { args: Array<Term> -> mkOrTerm(args.asList()) }
    }

    override fun transform(expr: KOrBinaryExpr) = with(expr) {
        transform(lhs, rhs) { l: Term, r: Term -> nsolver.mkTerm(Kind.OR, l, r) }
    }

    override fun transform(expr: KNotExpr) =
        with(expr) { transform(arg) { arg: Term -> nsolver.mkTerm(Kind.NOT, arg) } }

    override fun transform(expr: KImpliesExpr) = with(expr) {
        transform(p, q) { p: Term, q: Term ->
            nsolver.mkTerm(Kind.IMPLIES, p, q)
        }
    }

    override fun transform(expr: KXorExpr) = with(expr) {
        transform(a, b) { a: Term, b: Term ->
            nsolver.mkTerm(Kind.XOR, a, b)
        }
    }

    override fun transform(expr: KTrue) = expr.transform { nsolver.mkTrue() }

    override fun transform(expr: KFalse) = expr.transform { nsolver.mkFalse() }

    override fun <T : KSort> transform(expr: KEqExpr<T>) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            mkArraySpecificTerm(
                sort = expr.lhs.sort,
                arrayTerm = { mkArrayEqTerm(it, lhs, rhs) },
                default = { lhs.eqTerm(rhs) }
            )
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformArray(args) { args: Array<Term> ->
            mkArraySpecificTerm(
                sort = expr.args.first().sort,
                arrayTerm = { mkArrayDistinctTerm(it, args.asList()) },
                default = { nsolver.mkTerm(Kind.DISTINCT, args) }
            )
        }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        transform(
            condition, trueBranch, falseBranch
        ) { condition: Term, trueBranch: Term, falseBranch: Term ->
            mkArraySpecificTerm(
                sort = expr.trueBranch.sort,
                arrayTerm = { mkArrayIteTerm(it, condition, trueBranch, falseBranch) },
                default = { condition.iteTerm(trueBranch, falseBranch) }
            )
        }
    }

    fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>) = expr.transform {
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
        transform(value) { value: Term -> nsolver.mkTerm(Kind.BITVECTOR_NOT, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvReductionAndExpr<T>) = with(expr) {
        transform(value) { value: Term -> nsolver.mkTerm(Kind.BITVECTOR_REDAND, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvReductionOrExpr<T>) = with(expr) {
        transform(value) { value: Term -> nsolver.mkTerm(Kind.BITVECTOR_REDOR, value) }
    }

    override fun <T : KBvSort> transform(expr: KBvAndExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_AND, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvOrExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_OR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvXorExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_XOR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNAndExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_NAND, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNorExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_NOR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvXNorExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_XNOR, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNegationExpr<T>) = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_NEG, value)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ADD, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SUB, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_MULT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedDivExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UDIV, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedDivExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SDIV, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedRemExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UREM, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedRemExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SREM, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedModExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SMOD, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ULT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SLT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedLessOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ULE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedLessOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SLE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UGE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterOrEqualExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SGE, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvUnsignedGreaterExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_UGT, arg0, arg1)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSignedGreaterExpr<T>) = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SGT, arg0, arg1)
        }
    }

    override fun transform(expr: KBvConcatExpr): KExpr<KBvSort> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_CONCAT, arg0, arg1)
        }
    }

    override fun transform(expr: KBvExtractExpr) = with(expr) {
        transform(value) { value: Term ->
            val extractOp = nsolver.mkOp(Kind.BITVECTOR_EXTRACT, high, low)
            nsolver.mkTerm(extractOp, value)
        }
    }

    override fun transform(expr: KBvSignExtensionExpr) = with(expr) {
        transform(value) { value: Term ->
            val extensionOp = nsolver.mkOp(Kind.BITVECTOR_SIGN_EXTEND, extensionSize)
            nsolver.mkTerm(extensionOp, value)
        }
    }

    override fun transform(expr: KBvZeroExtensionExpr) = with(expr) {
        transform(value) { value: Term ->
            val extensionOp = nsolver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, extensionSize)
            nsolver.mkTerm(extensionOp, value)
        }
    }

    override fun transform(expr: KBvRepeatExpr) = with(expr) {
        transform(value) { value: Term ->
            val repeatOp = nsolver.mkOp(Kind.BITVECTOR_REPEAT, repeatNumber)
            nsolver.mkTerm(repeatOp, value)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvShiftLeftExpr<T>) = with(expr) {
        transform(arg, shift) { arg: Term, shift: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_SHL, arg, shift)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvLogicalShiftRightExpr<T>) = with(expr) {
        transform(arg, shift) { arg: Term, shift: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_LSHR, arg, shift)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvArithShiftRightExpr<T>) = with(expr) {
        transform(arg, shift) { arg: Term, shift: Term ->
            nsolver.mkTerm(Kind.BITVECTOR_ASHR, arg, shift)
        }
    }

    /*
     * we can internalize rotate expr as concat expr due to simplification,
     * otherwise we can't process rotate expr on expression
     */
    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) = with(expr) {
        transform {
            val simplifiedExpr = ctx.simplifyBvRotateLeftExpr(arg, rotation)

            if (simplifiedExpr is KBvRotateLeftExpr<*>) {
                throw KSolverUnsupportedFeatureException(
                    "Rotate expr with expression argument is not supported by cvc5"
                )
            }

            simplifiedExpr.internalizeExpr()
        }
    }

    /*
     * @see transform(expr: KBvRotateLeftExpr<T>)
     */
    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) = with(expr) {
        transform {
            val simplifiedExpr = ctx.simplifyBvRotateRightExpr(arg, rotation)

            if (simplifiedExpr is KBvRotateRightExpr<*>) {
                throw KSolverUnsupportedFeatureException(
                    "Rotate expr with expression argument is not supported by cvc5"
                )
            }

            simplifiedExpr.internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { value: Term ->
            val rotationOp = nsolver.mkOp(Kind.BITVECTOR_ROTATE_LEFT, rotationNumber)
            nsolver.mkTerm(rotationOp, value)
        }
    }


    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transform(value) { value: Term ->
            val rotationOp = nsolver.mkOp(Kind.BITVECTOR_ROTATE_RIGHT, rotationNumber)
            nsolver.mkTerm(rotationOp, value)
        }
    }

    // custom implementation
    @Suppress("MagicNumber")
    override fun transform(expr: KBv2IntExpr) = with(expr) {
        transform(value) { value: Term ->
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
        transform {
            ctx.rewriteBvAddNoOverflowExpr(arg0, arg1, isSigned).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvAddNoUnderflowExpr(arg0, arg1).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvSubNoOverflowExpr(arg0, arg1).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvSubNoUnderflowExpr(arg0, arg1, isSigned).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvDivNoOverflowExpr(arg0, arg1).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvNegNoOverflowExpr(value).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvMulNoOverflowExpr(arg0, arg1, isSigned).internalizeExpr()
        }
    }

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) = with(expr) {
        transform {
            ctx.rewriteBvMulNoUnderflowExpr(arg0, arg1).internalizeExpr()
        }
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
        transform {
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
        transform {
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
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_ABS, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpNegationExpr<T>): KExpr<T> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_NEG, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpAddExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_ADD,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpSubExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_SUB,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpMulExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_MULT,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpDivExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1) { roundingMode: Term, arg0: Term, arg1: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_DIV,
                roundingMode,
                arg0,
                arg1
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpFusedMulAddExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, arg0, arg1, arg2) { rm: Term, arg0: Term, arg1: Term, arg2: Term ->
            nsolver.mkTerm(
                Kind.FLOATINGPOINT_FMA,
                arrayOf(rm, arg0, arg1, arg2)
            )
        }
    }

    override fun <T : KFpSort> transform(expr: KFpSqrtExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { roundingMode: Term, value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_SQRT, roundingMode, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpRemExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_REM, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpRoundToIntegralExpr<T>): KExpr<T> =
        with(expr) {
            transform(roundingMode, value) { roundingMode: Term, value: Term ->
                nsolver.mkTerm(Kind.FLOATINGPOINT_RTI, roundingMode, value)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpMinExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_MIN, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpMaxExpr<T>): KExpr<T> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_MAX, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpLessOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_LEQ, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpLessExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_LT, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterOrEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_GEQ, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpGreaterExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_GT, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpEqualExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(arg0, arg1) { arg0: Term, arg1: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_EQ, arg0, arg1)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsNormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_NORMAL, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsSubnormalExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_SUBNORMAL, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsZeroExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_ZERO, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsInfiniteExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_INF, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsNaNExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_NAN, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsNegativeExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_NEG, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpIsPositiveExpr<T>): KExpr<KBoolSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_IS_POS, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        with(expr) {
            transform(roundingMode, value) { rm: Term, value: Term ->
                val opKind = if (isSigned) Kind.FLOATINGPOINT_TO_SBV else Kind.FLOATINGPOINT_TO_UBV
                val op = nsolver.mkOp(opKind, bvSize)
                nsolver.mkTerm(op, rm, value)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_TO_REAL, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> =
        throw KSolverUnsupportedFeatureException("no direct support for $expr")

    override fun <T : KFpSort> transform(expr: KFpFromBvExpr<T>): KExpr<T> = with(expr) {
        transform(sign, biasedExponent, significand) { sign: Term, biasedExp: Term, significand: Term ->
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
        transform(roundingMode, value) { rm: Term, value: Term ->
            val op = nsolver.mkOp(
                Kind.FLOATINGPOINT_TO_FP_FROM_FP,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Term, value: Term ->
            val op = nsolver.mkOp(
                Kind.FLOATINGPOINT_TO_FP_FROM_REAL,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KBvToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Term, value: Term ->
            val opKind = if (signed) Kind.FLOATINGPOINT_TO_FP_FROM_SBV else Kind.FLOATINGPOINT_TO_FP_FROM_UBV
            val op = nsolver.mkOp(
                opKind,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt()
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) =
        expr.transformStore()

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = expr.transformStore()

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = expr.transformStore()

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> =
        expr.transformStore()

    private fun <E : KArrayStoreBase<*, *>> E.transformStore(): E =
        transformArray(listOf(array, value) + indices) { transformedArgs: Array<Term> ->
            val (array, value) = transformedArgs.take(2)
            val indices = transformedArgs.drop(2)
            mkArrayStoreTerm(array, indices, value)
        }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) =
        expr.transformSelect()

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D0, D1, R>) =
        expr.transformSelect()

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = expr.transformSelect()

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> =
        expr.transformSelect()

    private fun <E : KArraySelectBase<*, *>> E.transformSelect(): E =
        transformArray(listOf(array) + indices) { transformedArgs: Array<Term> ->
            val array = transformedArgs.first()
            val indices = transformedArgs.drop(1)
            mkArraySelectTerm(array, indices)
        }

    override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KArrayConst<A, R>) = with(expr) {
        transform(value) { valueTerm: Term ->
            if (value is KInterpretedValue<*> || value is KArrayConst<*, *>) {
                // const array base must be a value or a constant array
                nsolver.mkConstArray(sort.internalizeSort(), valueTerm)
            } else {
                val bounds = sort.domainSorts.map { nsolver.mkConst(it.internalizeSort()) }
                mkLambdaTerm(bounds, valueTerm)
            }
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

    private fun <E : KArrayLambdaBase<*, *>> E.transformLambda(): E = transform(body) { bodyTerm: Term ->
        val bounds = indexVarDeclarations.map { it.internalizeDecl() }
        mkLambdaTerm(bounds, bodyTerm)
    }

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>) = with(expr) {
        transformArray(args) { args -> nsolver.mkTerm(Kind.ADD, args) }
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>) = with(expr) {
        transformArray(args) { args -> nsolver.mkTerm(Kind.SUB, args) }
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>) = with(expr) {
        transformArray(args) { args -> nsolver.mkTerm(Kind.MULT, args) }
    }

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>) = with(expr) {
        transform(arg) { arg: Term -> nsolver.mkTerm(Kind.NEG, arg) }
    }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            arithDivide(sort, lhs, rhs)
        }
    }

    private fun arithDivide(sort: KArithSort, lhs: Term, rhs: Term): Term = with(sort.ctx) {
        when (sort) {
            realSort -> nsolver.mkTerm(Kind.DIVISION, lhs, rhs)
            intSort -> nsolver.mkTerm(Kind.INTS_DIVISION, lhs, rhs)
            else -> throw KSolverUnsupportedFeatureException("Arith sort $sort is unsupported")
        }
    }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term -> nsolver.mkTerm(Kind.POW, lhs, rhs) }
    }

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.LT, lhs, rhs)
        }
    }

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.LEQ, lhs, rhs)
        }
    }

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.GT, lhs, rhs)
        }
    }

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.GEQ, lhs, rhs)
        }
    }

    override fun transform(expr: KModIntExpr) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            nsolver.mkTerm(Kind.INTS_MODULUS, lhs, rhs)
        }
    }

    // custom implementation
    override fun transform(expr: KRemIntExpr) = with(expr) {
        transform(lhs, rhs) { lhs: Term, rhs: Term ->
            // there is no ints remainder in cvc5
            val remSign = nsolver.mkTerm(Kind.GEQ, lhs, zeroIntValueTerm)
                .xorTerm(nsolver.mkTerm(Kind.GEQ, rhs, zeroIntValueTerm))
            val modTerm = nsolver.mkTerm(Kind.INTS_MODULUS, lhs, rhs)

            remSign.iteTerm(nsolver.mkTerm(Kind.NEG, modTerm), modTerm)
        }
    }

    override fun transform(expr: KToRealIntExpr) = with(expr) {
        transform(arg) { arg: Term ->
            nsolver.mkTerm(Kind.TO_REAL, arg)
        }
    }

    override fun transform(expr: KInt32NumExpr) = with(expr) {
        transform { nsolver.mkInteger(expr.value.toLong()) }
    }

    override fun transform(expr: KInt64NumExpr) = with(expr) {
        // We need to pass String value here because on Windows it might be cut to 32 bit int value
        transform { nsolver.mkInteger(expr.value.toString()) }
    }

    override fun transform(expr: KIntBigNumExpr) = with(expr) {
        transform { nsolver.mkInteger(expr.value.toString()) }
    }

    override fun transform(expr: KToIntRealExpr) = with(expr) {
        transform(arg) { arg: Term ->
            nsolver.mkTerm(Kind.TO_INTEGER, arg)
        }
    }

    override fun transform(expr: KIsIntRealExpr) = with(expr) {
        transform(arg) { arg: Term ->
            nsolver.mkTerm(Kind.IS_INTEGER, arg)
        }
    }

    override fun transform(expr: KRealNumExpr) = with(expr) {
        transform(numerator, denominator) { num: Term, denom: Term ->
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
     * There is no way in cvc5 API to mark uninterpreted constant as value.
     *
     * To overcome this we apply the following scheme:
     * 1. Internalize value `x` of a sort T as normal constant.
     * 2. Associate unique interpreted value `i` with this constant.
     * Currently, we use integer values.
     * 3. Introduce `interpreter` function `F` of type T -> Int.
     * We introduce one function for each uninterpreted sort.
     * 4. Assert expression `(= i (F x))` to the solver.
     * Since all Int values are known to be distinct, this
     * assertion forces that all values of T are also distinct.
     * */
    override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> = with(expr) {
        transform(ctx.mkIntNum(expr.valueIdx)) { intValueExpr: Term ->
            val exprSort = sort.internalizeSort()
            cvc5Ctx.saveUninterpretedSortValue(nsolver.mkConst(exprSort), expr).also {
                cvc5Ctx.registerUninterpretedSortValue(expr, intValueExpr, it) {
                    val descriptorSort = ctx.intSort.internalizeSort()
                    nsolver.declareFun("${sort.name}!interpreter", arrayOf(exprSort), descriptorSort)
                }
            }
        }
    }

    private fun <E : KExpr<*>> E.transformQuantifiedExpression(
        bounds: List<KDecl<*>>,
        body: KExpr<*>,
        isUniversal: Boolean
    ): E = transform(body) { transformedBody: Term ->
        val boundConsts = bounds.map { it.internalizeDecl() }
        mkQuantifierTerm(isUniversal, boundConsts, transformedBody)
    }

    private fun mkQuantifierTerm(isUniversal: Boolean, bounds: List<Term>, body: Term): Term =
        resolveBoundVars(bounds, body) { boundVars, bodyWithVars ->
            nsolver.mkQuantifier(isUniversal, boundVars, bodyWithVars)
        }

    private fun mkLambdaTerm(bounds: List<Term>, body: Term): Term =
        resolveBoundVars(bounds, body) { boundVars, bodyWithVars ->
            nsolver.mkLambda(boundVars, bodyWithVars)
        }

    private inline fun resolveBoundVars(bounds: List<Term>, body: Term, mkTerm: (Array<Term>, Term) -> Term): Term {
        val boundVars = bounds.map {
            if (it.hasSymbol()) {
                nsolver.mkVar(it.sort, it.symbol)
            } else {
                nsolver.mkVar(it.sort)
            }
        }.toTypedArray()

        val bodyWithVars = body.substitute(bounds.toTypedArray(), boundVars)
        return mkTerm(boundVars, bodyWithVars)
    }

    private fun Term.mkFunctionApp(args: List<Term>): Term =
        nsolver.mkTerm(Kind.APPLY_UF, arrayOf(this) + args)

    private fun mkAndTerm(args: List<Term>): Term =
        if (args.size == 1) args.single() else nsolver.mkTerm(Kind.AND, args.toTypedArray())

    private fun mkOrTerm(args: List<Term>): Term =
        if (args.size == 1) args.single() else nsolver.mkTerm(Kind.OR, args.toTypedArray())

    private fun mkArraySelectTerm(array: Term, indices: List<Term>): Term =
        if (array.sort.isArray) {
            val selectIdx = mkArrayOperationIndex(indices)
            nsolver.mkTerm(Kind.SELECT, array, selectIdx)
        } else {
            nsolver.mkTerm(Kind.APPLY_UF, arrayOf(array) + indices)
        }

    private fun mkArrayStoreTerm(array: Term, indices: List<Term>, value: Term): Term =
        if (array.sort.isArray) {
            val storeIdx = mkArrayOperationIndex(indices)
            nsolver.mkTerm(Kind.STORE, arrayOf(array, storeIdx, value))
        } else {
            val boundVars = indices.map { nsolver.mkVar(it.sort) }

            val condition = indices.zip(boundVars) { idx, boundVar ->
                idx.eqTerm(boundVar)
            }.let { mkAndTerm(it) }

            val arrayValue = array.mkFunctionApp(boundVars)
            val body = condition.iteTerm(value, arrayValue)

            nsolver.mkLambda(boundVars.toTypedArray(), body)
        }

    private fun mkArrayOperationIndex(indices: List<Term>): Term =
        if (indices.size == 1) {
            indices.single()
        } else {
            nsolver.mkTuple(indices.map { it.sort }.toTypedArray(), indices.toTypedArray())
        }

    private fun mkArrayEqTerm(arraySort: KArraySortBase<*>, lhs: Term, rhs: Term): Term {
        if (lhs.sort == rhs.sort) {
            return lhs.eqTerm(rhs)
        }

        val leftLambda = ensureArrayLambda(arraySort, lhs)
        val rightLambda = ensureArrayLambda(arraySort, rhs)
        return leftLambda.eqTerm(rightLambda)
    }

    /**
     * Ensure that an array expression has a function type.
     *
     * Since arrays can be represented as `SMT arrays` or as functions (lambda expressions)
     * we must coerce them to the same type.
     * We coerce `SMT arrays` to functions.
     * */
    private fun ensureArrayLambda(arraySort: KArraySortBase<*>, term: Term): Term =
        if (!term.sort.isArray) {
            term
        } else {
            val boundVars = arraySort.domainSorts.map { nsolver.mkVar(it.internalizeSort()) }
            val body = mkArraySelectTerm(term, boundVars)
            nsolver.mkLambda(boundVars.toTypedArray(), body)
        }

    private fun mkArrayIteTerm(arraySort: KArraySortBase<*>, cond: Term, trueBranch: Term, falseBranch: Term): Term {
        if (trueBranch.sort.isArray && falseBranch.sort.isArray) {
            return cond.iteTerm(trueBranch, falseBranch)
        }

        val boundVars = arraySort.domainSorts.map { nsolver.mkVar(it.internalizeSort()) }
        val trueValue = mkArraySelectTerm(trueBranch, boundVars)
        val falseValue = mkArraySelectTerm(falseBranch, boundVars)

        val body = cond.iteTerm(trueValue, falseValue)

        return nsolver.mkLambda(boundVars.toTypedArray(), body)
    }

    private fun mkArrayDistinctTerm(arraySort: KArraySortBase<*>, args: List<Term>): Term {
        val inequalities = mutableListOf<Term>()
        for (i in args.indices) {
            for (j in (i + 1) until args.size) {
                val equality = mkArrayEqTerm(arraySort, args[i], args[j])
                inequalities.add(equality.notTerm())
            }
        }

        return mkAndTerm(inequalities)
    }

    private inline fun mkArraySpecificTerm(
        sort: KSort,
        arrayTerm: (KArraySortBase<*>) -> Term,
        default: () -> Term
    ): Term = if (sort is KArraySortBase<*>) {
        arrayTerm(sort)
    } else {
        default()
    }

    inline fun <S : KExpr<*>> S.transformArray(
        args: List<KExpr<*>>,
        operation: (Array<Term>) -> Term
    ): S = transformList(args) { a: Array<Term> -> operation(a) }
}
