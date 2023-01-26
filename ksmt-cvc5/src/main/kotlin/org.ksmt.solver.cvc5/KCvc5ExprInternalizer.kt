package org.ksmt.solver.cvc5

import io.github.cvc5.*
import org.ksmt.decl.KDecl
import org.ksmt.expr.*
import org.ksmt.solver.util.KExprInternalizerBase
import org.ksmt.sort.*
import java.math.BigInteger

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
            // args[0] is a function declaration
            val decl = decl.internalizeDecl()

            if (decl.hasOp()) {
                nsolver.mkTerm(decl.op, args)
            } else {
                nsolver.mkTerm(Kind.APPLY_UF, arrayOf(decl, *args))
            }
        }
    }

    override fun <T : KSort> transform(expr: KConst<T>) = with(expr) {
        transform {
            decl.internalizeDecl()
        }
    }

    override fun transform(expr: KAndExpr) = with(expr) {
        transformArray(args) { args: Array<Term> -> nsolver.mkTerm(Kind.AND, args) }
    }

    override fun transform(expr: KOrExpr) = with(expr) {
        transformArray(args) { args: Array<Term> -> nsolver.mkTerm(Kind.OR, args) }
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
            nsolver.mkTerm(Kind.EQUAL, lhs, rhs)
        }
    }

    override fun <T : KSort> transform(expr: KDistinctExpr<T>) = with(expr) {
        transformArray(args) { args: Array<Term> -> nsolver.mkTerm(Kind.DISTINCT, args) }
    }

    override fun <T : KSort> transform(expr: KIteExpr<T>) = with(expr) {
        transform(condition, trueBranch, falseBranch) { condition: Term, trueBranch: Term, falseBranch: Term ->
            condition.iteTerm(trueBranch, falseBranch)
        }
    }

    // TODO: maybe for 1/8/16/32/64 int/long value set would be more effective than toString() conversion
    fun <T : KBvSort> transformBitVecValue(expr: KBitVecValue<T>) = expr.transform {
        when (expr) {
            is KBitVec1Value, is KBitVec8Value, is KBitVec16Value, is KBitVec32Value, is KBitVec64Value, is KBitVecCustomValue -> {
                nsolver.mkBitVector(expr.decl.sort.sizeBits.toInt(), expr.stringValue, 2)
            }

            else -> error("Unknown bv expression class ${expr::class} in transformation method: $expr")
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


    // we can check that expr.rotation is KBitvecValue and translate to KBvRotateLeftIndexedExpr
    override fun <T : KBvSort> transform(expr: KBvRotateLeftExpr<T>) = TODO("No direct support for cvc5")

    override fun <T : KBvSort> transform(expr: KBvRotateLeftIndexedExpr<T>) = with(expr) {
        transform(value) { value: Term ->
            val rotationOp = nsolver.mkOp(Kind.BITVECTOR_ROTATE_LEFT, rotationNumber)
            nsolver.mkTerm(rotationOp, value)
        }
    }

    // the same with KBvRotateLeftExpr
    override fun <T : KBvSort> transform(expr: KBvRotateRightExpr<T>) = TODO("No direct support for cvc5")

    override fun <T : KBvSort> transform(expr: KBvRotateRightIndexedExpr<T>) = with(expr) {
        transform(value) { value: Term ->
            val rotationOp = nsolver.mkOp(Kind.BITVECTOR_ROTATE_RIGHT, rotationNumber)
            nsolver.mkTerm(rotationOp, value)
        }
    }

    // custom implementation
    override fun transform(expr: KBv2IntExpr) = with(expr) {
        transform(value) { value: Term ->
            // by default, it is unsigned in cvc5
            val intTerm = nsolver.mkTerm(Kind.BITVECTOR_TO_NAT, value)

            if (isSigned) {
                val size = this.value.sort.sizeBits.toInt()
                val modulo = BigInteger.ONE shl size
                val maxInt = (BigInteger.ONE shl (size - 1)) - BigInteger.ONE

                val moduloTerm = nsolver.mkInteger(modulo.toString(10))
                val maxIntTerm = nsolver.mkInteger(maxInt.toString(10))

                val gtTerm = nsolver.mkTerm(Kind.GT, intTerm, maxIntTerm)
                val subTerm = nsolver.mkTerm(Kind.SUB, intTerm, moduloTerm)

                nsolver.mkTerm(Kind.ITE, gtTerm, subTerm, intTerm)
            } else intTerm
        }
    }

    // custom implementation
    override fun <T : KBvSort> transform(expr: KBvAddNoOverflowExpr<T>) = with(expr) {
        transform(arg0, arg1) { a0: Term, a1: Term ->

            val zExtOp = nsolver.mkOp(Kind.BITVECTOR_ZERO_EXTEND, 1)

            val a0ExtTerm = nsolver.mkTerm(zExtOp, a0)
            val a1ExtTerm = nsolver.mkTerm(zExtOp, a1)

            val additionTerm = nsolver.mkTerm(Kind.BITVECTOR_ADD, a0ExtTerm, a1ExtTerm)

            val signBitPos = arg0.sort.sizeBits.toInt()
            val extractOp = nsolver.mkOp(Kind.BITVECTOR_EXTRACT, signBitPos, signBitPos)

            val extractionTerm = nsolver.mkTerm(extractOp, additionTerm)

            val zeroBvValueTerm = nsolver.mkBitVector(1, 0L)

            nsolver.mkTerm(Kind.EQUAL, extractionTerm, zeroBvValueTerm)
        }
    }

    override fun <T : KBvSort> transform(expr: KBvAddNoUnderflowExpr<T>) = TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvSubNoOverflowExpr<T>) = TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvSubNoUnderflowExpr<T>) = TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvDivNoOverflowExpr<T>) = TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvNegNoOverflowExpr<T>) = TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvMulNoOverflowExpr<T>) = TODO("no direct support for $expr")

    override fun <T : KBvSort> transform(expr: KBvMulNoUnderflowExpr<T>) = TODO("no direct support for $expr")


    // TODO: maybe bitvector should contain unbiased exponent?
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
                // TODO: by cvc5 (1.0.2) documentation - args are terms of floating-point Sort (sorts must match)
                // that's why it shouldn't work
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


    @Suppress("UNUSED_ANONYMOUS_PARAMETER")
    override fun <T : KFpSort> transform(expr: KFpToBvExpr<T>): KExpr<KBvSort> =
        with(expr) {
            transform(roundingMode, value) { rm: Term, value: Term ->
                val opKind = if (isSigned) Kind.FLOATINGPOINT_TO_SBV else Kind.FLOATINGPOINT_TO_UBV
                val op = nsolver.mkOp(opKind, bvSize)

                // TODO: shouldn't work, by cvc5 (1.0.2) documentation there is only 1 arg - value (no rounding mode)
                nsolver.mkTerm(op, value)
            }
        }

    override fun <T : KFpSort> transform(expr: KFpToRealExpr<T>): KExpr<KRealSort> = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkTerm(Kind.FLOATINGPOINT_TO_REAL, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KFpToIEEEBvExpr<T>): KExpr<KBvSort> = TODO("no direct support for $expr")


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
                sort.significandBits.toInt() + 1
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <T : KFpSort> transform(expr: KRealToFpExpr<T>): KExpr<T> = with(expr) {
        transform(roundingMode, value) { rm: Term, value: Term ->
            val op = nsolver.mkOp(
                Kind.FLOATINGPOINT_TO_FP_FROM_REAL,
                sort.exponentBits.toInt(),
                sort.significandBits.toInt() + 1
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
                sort.significandBits.toInt() + 1
            )

            nsolver.mkTerm(op, rm, value)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>) = with(expr) {
        transform(array, index, value) { array: Term, index: Term, value: Term ->
            nsolver.mkTerm(Kind.STORE, arrayOf(array, index, value))
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>) = with(expr) {
        transform(array, index) { array: Term, index: Term ->
            nsolver.mkTerm(Kind.SELECT, array, index)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayConst<D, R>) = with(expr) {
        transform(value) { value: Term ->
            nsolver.mkConstArray(sort.domain.internalizeSort(), value)
        }
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>) = TODO("no direct impl for $expr")

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
        transform(lhs, rhs) { lhs: Term, rhs: Term -> nsolver.mkTerm(Kind.DIVISION, lhs, rhs) }
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
    override fun transform(expr: KRemIntExpr) = with(expr) { transform(lhs, rhs) { lhs: Term, rhs: Term ->
        // there is no ints remainder in cvc5
        val remSign = nsolver.mkTerm(Kind.GEQ, lhs, zeroIntValueTerm)
            .xorTerm(
                nsolver.mkTerm(Kind.GEQ, rhs, zeroIntValueTerm)
            )
        val modTerm = nsolver.mkTerm(Kind.INTS_MODULUS, lhs, rhs)

        remSign.iteTerm(nsolver.mkTerm(Kind.NEG, modTerm), modTerm)
    } }

    override fun transform(expr: KToRealIntExpr) = with(expr) { transform(arg) { arg: Term ->
        nsolver.mkTerm(Kind.TO_REAL, arg)
    } }

    override fun transform(expr: KInt32NumExpr) = with(expr) {
        transform { nsolver.mkInteger(expr.value.toLong()) }
    }

    override fun transform(expr: KInt64NumExpr) = with(expr) {
        transform { nsolver.mkInteger(expr.value) }
    }

    override fun transform(expr: KIntBigNumExpr) = with(expr) {
        transform { nsolver.mkInteger(expr.value.toString()) }
    }

    override fun transform(expr: KToIntRealExpr) = with(expr) { transform(arg) { arg: Term ->
        nsolver.mkTerm(Kind.TO_INTEGER, arg)
    } }

    override fun transform(expr: KIsIntRealExpr) = with(expr) { transform(arg) { arg: Term ->
        nsolver.mkTerm(Kind.IS_INTEGER, arg)
    } }

    override fun transform(expr: KRealNumExpr) = with(expr) {
        transform(ctx.mkIntToReal(numerator), ctx.mkIntToReal(denominator)) { num: Term, denum: Term ->
            nsolver.mkTerm(Kind.DIVISION, num, denum)
        }
    }

    private fun transformQuantifier(expr: KQuantifier, isUniversal: Boolean) = with(expr) {
        transformArray(bounds.map { ctx.mkConstApp(it) } + body) { args ->
            val cvc5Consts = args.copyOfRange(0, args.size - 1)
            val cvc5Body = args.last()

            val cvc5Vars = cvc5Consts.map { nsolver.mkVar(it.sort, it.symbol) }.toTypedArray()
            val cvc5BodySubstituted = cvc5Body.substitute(cvc5Consts, cvc5Vars)

            nsolver.mkQuantifier(isUniversal, cvc5Vars, cvc5BodySubstituted)
        }
    }

    override fun transform(expr: KExistentialQuantifier) = transformQuantifier(expr, isUniversal = false)

    override fun transform(expr: KUniversalQuantifier) = transformQuantifier(expr, isUniversal = true)

    override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
        TODO("No direct impl in cvc5 (as-array is CONST_ARRAY term with base array element)")
    }

    @Suppress("ArrayPrimitive")
    inline fun <S : KExpr<*>> S.transformArray(
        args: List<KExpr<*>>,
        operation: (Array<Term>) -> Term
    ): S = transformList(args) { a: Array<Term> -> operation(a) }

}