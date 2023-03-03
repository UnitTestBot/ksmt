package org.ksmt.solver.cvc5

import io.github.cvc5.Kind
import io.github.cvc5.RoundingMode
import io.github.cvc5.Sort
import io.github.cvc5.Term
import io.github.cvc5.bvLowerExtractionBitIndex
import io.github.cvc5.bvRepeatTimes
import io.github.cvc5.bvRotateBitsCount
import io.github.cvc5.bvSignExtensionSize
import io.github.cvc5.bvSizeToConvertTo
import io.github.cvc5.bvUpperExtractionBitIndex
import io.github.cvc5.bvZeroExtensionSize
import io.github.cvc5.intDivisibleArg
import io.github.cvc5.toFpExponentSize
import io.github.cvc5.toFpSignificandSize
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.util.KExprConverterBase
import org.ksmt.solver.util.ExprConversionResult
import org.ksmt.solver.util.KExprConverterUtils.argumentsConversionRequired
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.uncheckedCast
import java.util.*

open class KCvc5ExprConverter(
    private val ctx: KContext,
    private val cvc5Ctx: KCvc5Context
) : KExprConverterBase<Term>() {

    private val internalizer = KCvc5ExprInternalizer(cvc5Ctx)

    override fun findConvertedNative(expr: Term): KExpr<*>? = cvc5Ctx.findConvertedExpr(expr)

    override fun saveConvertedNative(native: Term, converted: KExpr<*>) {
        cvc5Ctx.saveConvertedExpr(native, converted)
    }

    @Suppress("LongMethod", "ComplexMethod")
    override fun convertNativeExpr(expr: Term): ExprConversionResult = with(ctx) {
        when (expr.kind) {
            // const
            Kind.CONST_BOOLEAN -> convertNativeBoolConstExpr(expr)
            Kind.CONST_FLOATINGPOINT -> convertNativeFloatingPointConstExpr(expr)
            Kind.CONST_ROUNDINGMODE -> convertNativeRoundingModeConstExpr(expr)
            Kind.CONST_BITVECTOR -> convertNativeBitvectorConstExpr(expr)
            Kind.CONST_ARRAY -> convertNativeConstArrayExpr<KSort, KSort>(expr)
            Kind.CONST_INTEGER -> convertNativeConstIntegerExpr(expr)
            Kind.CONST_RATIONAL -> convertNativeConstRealExpr(expr)
            Kind.CONSTANT -> convert { mkConst(expr.symbol, expr.sort.convertSort()) }
            Kind.UNINTERPRETED_SORT_VALUE -> convert { mkConst(expr.uninterpretedSortValue, expr.sort.convertSort()) }

            // equality, cmp, ite
            Kind.EQUAL -> expr.convert(::mkEq)
            Kind.DISTINCT -> expr.convertList(::mkDistinct)
            Kind.ITE -> expr.convert(::mkIte)
            Kind.LEQ -> expr.convertChainable(::mkArithLe, ::mkAnd)
            Kind.GEQ -> expr.convertChainable(::mkArithGe, ::mkAnd)
            Kind.LT -> expr.convertChainable(::mkArithLt, ::mkAnd)
            Kind.GT -> expr.convertChainable(::mkArithGt, ::mkAnd)

            // bool
            Kind.NOT -> expr.convert(::mkNot)
            Kind.AND -> expr.convertList(::mkAnd)
            Kind.OR -> expr.convertList(::mkOr)
            Kind.XOR -> expr.convertReduced(::mkXor)
            Kind.IMPLIES -> expr.convert(::mkImplies)

            // quantifiers
            Kind.FORALL -> convertNativeQuantifierExpr(expr)
            Kind.EXISTS -> convertNativeQuantifierExpr(expr)

            Kind.VARIABLE_LIST -> error("variable list should not be handled here")
            Kind.VARIABLE -> error("variable should not be handled here")
            Kind.INST_PATTERN,
            Kind.INST_PATTERN_LIST,
            Kind.INST_NO_PATTERN -> throw KSolverUnsupportedFeatureException("ksmt has no impl of patterns")

            Kind.INST_POOL,
            Kind.INST_ADD_TO_POOL,
            Kind.SKOLEM_ADD_TO_POOL -> throw KSolverUnsupportedFeatureException(
                "annotations of instantiations are not supported in ksmt now"
            )
            Kind.INST_ATTRIBUTE -> throw KSolverUnsupportedFeatureException(
                "attributes for a quantifier are not supported in ksmt now"
            )

            Kind.HO_APPLY -> throw KSolverUnsupportedFeatureException("no direct mapping in ksmt")
            Kind.CARDINALITY_CONSTRAINT -> throw KSolverUnsupportedFeatureException(
                "Operation is not supported in ksmt now"
            )
            Kind.APPLY_UF -> expr.getChildren().let { children ->
                expr.convertList(children.copyOfRange(1, children.size)) { args: List<KExpr<KSort>> ->
                    val fTerm = children.first().convertDecl<KSort>()
                    mkApp(fTerm, args)
                }
            }

            Kind.LAMBDA -> error("lambdas used only in interpretations, and they handled separately")
            Kind.SEXPR -> throw KSolverUnsupportedFeatureException("Operation is not supported in ksmt now")

            // arith
            Kind.ABS -> expr.convert { arithExpr: KExpr<KArithSort> ->
                val geExpr = if (arithExpr.sort is KIntSort) arithExpr.asExpr(intSort) ge 0.expr
                else arithExpr.asExpr(realSort) ge mkRealNum(0)
                mkIte(geExpr, arithExpr, -arithExpr)
            }
            Kind.ADD -> expr.convertList(::mkArithAdd)
            Kind.SUB -> expr.convertList(::mkArithSub)
            Kind.MULT -> expr.convertList(::mkArithMul)
            Kind.NEG -> expr.convert(::mkArithUnaryMinus)
            Kind.SQRT -> throw KSolverUnsupportedFeatureException("No direct mapping of sqrt on real in ksmt")
            Kind.POW -> convertNativePowExpr(expr)
            Kind.POW2 -> convertNativePowExpr(expr)
            Kind.INTS_MODULUS -> expr.convert(::mkIntMod)
            Kind.INTS_DIVISION -> expr.convert(::mkArithDiv)
            Kind.DIVISION -> expr.convert(::mkArithDiv)
            Kind.ARCCOTANGENT,
            Kind.ARCSECANT,
            Kind.ARCCOSECANT,
            Kind.ARCTANGENT,
            Kind.ARCCOSINE,
            Kind.ARCSINE,
            Kind.COTANGENT,
            Kind.SECANT,
            Kind.COSECANT,
            Kind.TANGENT,
            Kind.COSINE,
            Kind.SINE,
            Kind.PI,
            Kind.EXPONENTIAL -> throw KSolverUnsupportedFeatureException("No direct mapping in ksmt")

            Kind.TO_REAL -> expr.convert(::mkIntToReal)
            Kind.TO_INTEGER -> expr.convert(::mkRealToInt)
            Kind.IS_INTEGER -> expr.convert(::mkRealIsInt)
            Kind.DIVISIBLE -> expr.convert { arExpr: KExpr<KIntSort> ->
                mkIntMod(arExpr, expr.intDivisibleArg.expr) eq 0.expr
            }
            Kind.IAND -> throw KSolverUnsupportedFeatureException(
                "No direct mapping in ksmt. btw int to bv is not supported in ksmt"
            )

            // bitvectors
            Kind.BITVECTOR_NOT -> expr.convert(::mkBvNotExpr)
            Kind.BITVECTOR_AND -> expr.convertCascadeBinaryArityOpOrArg(::mkBvAndExpr)
            Kind.BITVECTOR_NAND -> expr.convert(::mkBvNAndExpr)
            Kind.BITVECTOR_OR -> expr.convertCascadeBinaryArityOpOrArg(::mkBvOrExpr)
            Kind.BITVECTOR_NOR -> expr.convert(::mkBvNorExpr)
            Kind.BITVECTOR_XNOR -> expr.convert(::mkBvXNorExpr)
            Kind.BITVECTOR_XOR -> expr.convert(::mkBvXorExpr)

            Kind.BITVECTOR_SHL -> expr.convert(::mkBvShiftLeftExpr)
            Kind.BITVECTOR_LSHR -> expr.convert(::mkBvLogicalShiftRightExpr)
            Kind.BITVECTOR_ASHR -> expr.convert(::mkBvArithShiftRightExpr)

            Kind.BITVECTOR_ULT -> expr.convert(::mkBvUnsignedLessExpr)
            Kind.BITVECTOR_ULE -> expr.convert(::mkBvUnsignedLessOrEqualExpr)
            Kind.BITVECTOR_UGT -> expr.convert(::mkBvUnsignedGreaterExpr)
            Kind.BITVECTOR_UGE -> expr.convert(::mkBvUnsignedGreaterOrEqualExpr)
            Kind.BITVECTOR_SLT -> expr.convert(::mkBvSignedLessExpr)
            Kind.BITVECTOR_SLE -> expr.convert(::mkBvSignedLessOrEqualExpr)
            Kind.BITVECTOR_SGT -> expr.convert(::mkBvSignedGreaterExpr)
            Kind.BITVECTOR_SGE -> expr.convert(::mkBvSignedGreaterOrEqualExpr)
            Kind.BITVECTOR_ULTBV -> throw KSolverUnsupportedFeatureException(
                "No direct mapping of ${Kind.BITVECTOR_ULTBV} in ksmt"
            )

            Kind.BITVECTOR_SLTBV -> throw KSolverUnsupportedFeatureException(
                "No direct mapping of ${Kind.BITVECTOR_SLTBV} in ksmt"
            )

            Kind.BITVECTOR_ITE -> throw KSolverUnsupportedFeatureException(
                "No direct mapping of ${Kind.BITVECTOR_ITE} in ksmt"
            )

            Kind.BITVECTOR_REDOR -> expr.convert(::mkBvReductionOrExpr)
            Kind.BITVECTOR_REDAND -> expr.convert(::mkBvReductionAndExpr)

            Kind.BITVECTOR_EXTRACT -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBvExtractExpr(expr.bvUpperExtractionBitIndex, expr.bvLowerExtractionBitIndex, bvExpr)
            }
            Kind.BITVECTOR_REPEAT -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBvRepeatExpr(expr.bvRepeatTimes, bvExpr)
            }
            Kind.BITVECTOR_ZERO_EXTEND -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBvZeroExtensionExpr(expr.bvZeroExtensionSize, bvExpr)
            }
            Kind.BITVECTOR_SIGN_EXTEND -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBvSignExtensionExpr(expr.bvSignExtensionSize, bvExpr)
            }
            Kind.BITVECTOR_ROTATE_LEFT -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBvRotateLeftIndexedExpr(expr.bvRotateBitsCount, bvExpr)
            }
            Kind.BITVECTOR_ROTATE_RIGHT -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBvRotateRightIndexedExpr(expr.bvRotateBitsCount, bvExpr)
            }

            Kind.BITVECTOR_NEG -> expr.convert(::mkBvNegationExpr)
            Kind.BITVECTOR_ADD -> expr.convertReduced(::mkBvAddExpr)
            Kind.BITVECTOR_SUB -> expr.convertReduced(::mkBvSubExpr)
            Kind.BITVECTOR_MULT -> expr.convertReduced(::mkBvMulExpr)
            Kind.BITVECTOR_SDIV -> expr.convert(::mkBvSignedDivExpr)
            Kind.BITVECTOR_UDIV -> expr.convert(::mkBvUnsignedDivExpr)
            Kind.BITVECTOR_SREM -> expr.convert(::mkBvSignedRemExpr)
            Kind.BITVECTOR_UREM -> expr.convert(::mkBvUnsignedRemExpr)
            Kind.BITVECTOR_SMOD -> expr.convert(::mkBvSignedModExpr)
            Kind.BITVECTOR_CONCAT -> expr.convertReduced(::mkBvConcatExpr)

            Kind.INT_TO_BITVECTOR -> throw KSolverUnsupportedFeatureException(
                "No direct mapping of ${Kind.INT_TO_BITVECTOR} in ksmt"
            )
            // bitvector -> non-negative integer
            Kind.BITVECTOR_TO_NAT -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBv2IntExpr(bvExpr, false)
            }
            Kind.BITVECTOR_COMP -> throw KSolverUnsupportedFeatureException(
                "No direct mapping of ${Kind.BITVECTOR_COMP} in ksmt"
            )

            // fp
            Kind.FLOATINGPOINT_FP -> {
                expr.convert { bvSign: KExpr<KBv1Sort>, bvExponent: KExpr<KBvSort>, bvSignificand: KExpr<KBvSort> ->
                    bvSign as KBitVecValue<KBv1Sort>
                    bvExponent as KBitVecValue<*>
                    bvSignificand as KBitVecValue<*>
                    mkFpCustomSizeBiased(
                        bvSignificand.sort.sizeBits,
                        bvExponent.sort.sizeBits,
                        bvSignificand,
                        bvExponent,
                        bvSign.stringValue[0] == '1'
                    )
                }
            }
            Kind.FLOATINGPOINT_EQ -> expr.convert(::mkFpEqualExpr)
            Kind.FLOATINGPOINT_ABS -> expr.convert(::mkFpAbsExpr)
            Kind.FLOATINGPOINT_NEG -> expr.convert(::mkFpNegationExpr)
            Kind.FLOATINGPOINT_ADD -> expr.convert(::mkFpAddExpr)
            Kind.FLOATINGPOINT_SUB -> expr.convert(::mkFpSubExpr)
            Kind.FLOATINGPOINT_MULT -> expr.convert(::mkFpMulExpr)
            Kind.FLOATINGPOINT_DIV -> expr.convert(::mkFpDivExpr)
            Kind.FLOATINGPOINT_FMA -> expr.convert(::mkFpFusedMulAddExpr)
            Kind.FLOATINGPOINT_SQRT -> expr.convert(::mkFpSqrtExpr)
            Kind.FLOATINGPOINT_REM -> expr.convert(::mkFpRemExpr)
            Kind.FLOATINGPOINT_RTI -> expr.convert(::mkFpRoundToIntegralExpr)
            Kind.FLOATINGPOINT_MIN -> expr.convert(::mkFpMinExpr)
            Kind.FLOATINGPOINT_MAX -> expr.convert(::mkFpMaxExpr)
            Kind.FLOATINGPOINT_LEQ -> expr.convert(::mkFpLessOrEqualExpr)
            Kind.FLOATINGPOINT_LT -> expr.convert(::mkFpLessExpr)
            Kind.FLOATINGPOINT_GEQ -> expr.convert(::mkFpGreaterOrEqualExpr)
            Kind.FLOATINGPOINT_GT -> expr.convert(::mkFpGreaterExpr)
            Kind.FLOATINGPOINT_IS_NORMAL -> expr.convert(::mkFpIsNormalExpr)
            Kind.FLOATINGPOINT_IS_SUBNORMAL -> expr.convert(::mkFpIsSubnormalExpr)
            Kind.FLOATINGPOINT_IS_ZERO -> expr.convert(::mkFpIsZeroExpr)
            Kind.FLOATINGPOINT_IS_INF -> expr.convert(::mkFpIsInfiniteExpr)
            Kind.FLOATINGPOINT_IS_NAN -> expr.convert(::mkFpIsNaNExpr)
            Kind.FLOATINGPOINT_IS_NEG -> expr.convert(::mkFpIsNegativeExpr)
            Kind.FLOATINGPOINT_IS_POS -> expr.convert(::mkFpIsPositiveExpr)
            Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV -> expr.convert { bv: KExpr<KBvSort> ->
                val expSize = expr.toFpExponentSize
                val significandSize = expr.toFpSignificandSize

                val signPos = expSize + significandSize - 1

                @Suppress("UNCHECKED_CAST")
                val bvSign = mkBvExtractExpr(signPos, signPos, bv) as KExpr<KBv1Sort>
                val bvExp = mkBvExtractExpr(signPos - 1, signPos - 1 - expSize + 1, bv)
                val bvSignificand = mkBvExtractExpr(significandSize - 1 - 1, 0, bv) // without sign

                mkFpFromBvExpr(bvSign, bvExp, bvSignificand)
            }
            Kind.FLOATINGPOINT_TO_FP_FROM_FP -> expr.convert { rm: KExpr<KFpRoundingModeSort>, fpExpr: KExpr<KFpSort> ->
                mkFpToFpExpr(
                    mkFpSort(expr.toFpExponentSize.toUInt(), expr.toFpSignificandSize.toUInt()),
                    rm,
                    fpExpr
                )
            }
            Kind.FLOATINGPOINT_TO_FP_FROM_REAL -> {
                expr.convert { roundingMode: KExpr<KFpRoundingModeSort>, realExpr: KExpr<KRealSort> ->
                    mkRealToFpExpr(
                        mkFpSort(expr.toFpExponentSize.toUInt(), expr.toFpSignificandSize.toUInt()),
                        roundingMode,
                        realExpr
                    )
                }
            }
            Kind.FLOATINGPOINT_TO_FP_FROM_SBV -> convertNativeBvToFpExpr(expr, true)
            Kind.FLOATINGPOINT_TO_FP_FROM_UBV -> convertNativeBvToFpExpr(expr, false)
            Kind.FLOATINGPOINT_TO_UBV -> convertNativeFpToBvExpr(expr, false)
            Kind.FLOATINGPOINT_TO_SBV -> convertNativeFpToBvExpr(expr, true)
            Kind.FLOATINGPOINT_TO_REAL -> expr.convert { fpExpr: KExpr<KFpSort> -> mkFpToRealExpr(fpExpr) }

            // array
            Kind.SELECT -> expr.convert(::mkArray1Select)
            Kind.STORE -> expr.convert(::mkArray1Store)

            Kind.EQ_RANGE -> throw KSolverUnsupportedFeatureException("EQ_RANGE is not supported")

            Kind.APPLY_CONSTRUCTOR,
            Kind.APPLY_SELECTOR,
            Kind.APPLY_TESTER,
            Kind.APPLY_UPDATER -> throw KSolverUnsupportedFeatureException("currently ksmt does not support datatypes")

            Kind.MATCH,
            Kind.MATCH_CASE,
            Kind.MATCH_BIND_CASE,
            Kind.TUPLE_PROJECT -> throw KSolverUnsupportedFeatureException("currently ksmt does not support tuples")

            Kind.SEP_NIL,
            Kind.SEP_EMP,
            Kind.SEP_PTO,
            Kind.SEP_STAR,
            Kind.SEP_WAND -> throw KSolverUnsupportedFeatureException(
                "currently ksmt does not support separation logic"
            )

            Kind.SET_EMPTY,
            Kind.SET_UNION,
            Kind.SET_INTER,
            Kind.SET_MINUS,
            Kind.SET_SUBSET,
            Kind.SET_MEMBER,
            Kind.SET_SINGLETON,
            Kind.SET_INSERT,
            Kind.SET_CARD,
            Kind.SET_COMPLEMENT,
            Kind.SET_UNIVERSE,
            Kind.SET_COMPREHENSION,
            Kind.SET_CHOOSE,
            Kind.SET_IS_SINGLETON,
            Kind.SET_MAP,
            Kind.SET_FILTER,
            Kind.SET_FOLD -> throw KSolverUnsupportedFeatureException("currently ksmt does not support sets")

            Kind.RELATION_JOIN,
            Kind.RELATION_PRODUCT,
            Kind.RELATION_TRANSPOSE,
            Kind.RELATION_TCLOSURE,
            Kind.RELATION_JOIN_IMAGE,
            Kind.RELATION_IDEN,
            Kind.RELATION_GROUP,
            Kind.RELATION_AGGREGATE,
            Kind.RELATION_PROJECT -> throw KSolverUnsupportedFeatureException(
                "currently ksmt does not support relations"
            )

            Kind.BAG_EMPTY,
            Kind.BAG_UNION_MAX,
            Kind.BAG_UNION_DISJOINT,
            Kind.BAG_INTER_MIN,
            Kind.BAG_DIFFERENCE_SUBTRACT,
            Kind.BAG_DIFFERENCE_REMOVE,
            Kind.BAG_SUBBAG,
            Kind.BAG_COUNT,
            Kind.BAG_MEMBER,
            Kind.BAG_DUPLICATE_REMOVAL,
            Kind.BAG_MAKE,
            Kind.BAG_CARD,
            Kind.BAG_CHOOSE,
            Kind.BAG_IS_SINGLETON,
            Kind.BAG_FROM_SET,
            Kind.BAG_TO_SET,
            Kind.BAG_MAP,
            Kind.BAG_FILTER,
            Kind.BAG_FOLD,
            Kind.BAG_PARTITION -> throw KSolverUnsupportedFeatureException("currently ksmt does not support bags")

            Kind.TABLE_PRODUCT,
            Kind.TABLE_PROJECT,
            Kind.TABLE_AGGREGATE,
            Kind.TABLE_JOIN,
            Kind.TABLE_GROUP -> throw KSolverUnsupportedFeatureException("currently ksmt does not support tables")

            Kind.STRING_CONCAT,
            Kind.STRING_IN_REGEXP,
            Kind.STRING_LENGTH,
            Kind.STRING_SUBSTR,
            Kind.STRING_UPDATE,
            Kind.STRING_CHARAT,
            Kind.STRING_CONTAINS,
            Kind.STRING_INDEXOF,
            Kind.STRING_INDEXOF_RE,
            Kind.STRING_REPLACE,
            Kind.STRING_REPLACE_ALL,
            Kind.STRING_REPLACE_RE,
            Kind.STRING_REPLACE_RE_ALL,
            Kind.STRING_TO_LOWER,
            Kind.STRING_TO_UPPER,
            Kind.STRING_REV,
            Kind.STRING_TO_CODE,
            Kind.STRING_FROM_CODE,
            Kind.STRING_LT,
            Kind.STRING_LEQ,
            Kind.STRING_PREFIX,
            Kind.STRING_SUFFIX,
            Kind.STRING_IS_DIGIT,
            Kind.STRING_FROM_INT,
            Kind.STRING_TO_INT,
            Kind.CONST_STRING,
            Kind.STRING_TO_REGEXP -> throw KSolverUnsupportedFeatureException("currently ksmt does not support strings")

            Kind.REGEXP_CONCAT,
            Kind.REGEXP_UNION,
            Kind.REGEXP_INTER,
            Kind.REGEXP_DIFF,
            Kind.REGEXP_STAR,
            Kind.REGEXP_PLUS,
            Kind.REGEXP_OPT,
            Kind.REGEXP_RANGE,
            Kind.REGEXP_REPEAT,
            Kind.REGEXP_LOOP,
            Kind.REGEXP_NONE,
            Kind.REGEXP_ALL,
            Kind.REGEXP_ALLCHAR,
            Kind.REGEXP_COMPLEMENT -> throw KSolverUnsupportedFeatureException(
                "currently ksmt does not support regular expressions"
            )

            Kind.SEQ_CONCAT,
            Kind.SEQ_LENGTH,
            Kind.SEQ_EXTRACT,
            Kind.SEQ_UPDATE,
            Kind.SEQ_AT,
            Kind.SEQ_CONTAINS,
            Kind.SEQ_INDEXOF,
            Kind.SEQ_REPLACE,
            Kind.SEQ_REPLACE_ALL,
            Kind.SEQ_REV,
            Kind.SEQ_PREFIX,
            Kind.SEQ_SUFFIX,
            Kind.CONST_SEQUENCE,
            Kind.SEQ_UNIT,
            Kind.SEQ_NTH -> throw KSolverUnsupportedFeatureException("currently ksmt does not support sequences")

            Kind.WITNESS -> error("no direct mapping in ksmt")
            Kind.LAST_KIND -> error("should not be here. Marks the upper-bound of this enumeration, not op kind")
            Kind.INTERNAL_KIND -> error("should not be here. Not exposed via the API")
            Kind.UNDEFINED_KIND -> error("should not be here. Not exposed via the API")
            Kind.NULL_TERM -> error("no support in ksmt")
            null -> error("kind can't be null")
        }

    }

    // Kind.[FORALL/EXISTS] to Array of Kind.VARIABLE
    private val qfVarsAsConsts = TreeMap<Term, Array<Term>>()

    @Suppress("UNCHECKED_CAST")
    private fun convertNativeQuantifierExpr(expr: Term) = with(ctx) {
        val mkQf = when (expr.kind) {
            Kind.FORALL -> ::mkUniversalQuantifier
            Kind.EXISTS -> ::mkExistentialQuantifier
            else -> error("Unknown term of quantifier. Kind of term: ${expr.kind}")
        }

        val cvc5VarList = expr.getChild(0)
        val cvc5BodyWithVars = expr.getChild(1)

        val cvc5SubstitutedConsts = qfVarsAsConsts[expr]
        if (cvc5SubstitutedConsts == null) {

            // fresh bounds
            val bounds = cvc5VarList.getChildren().map { mkFreshConstDecl(it.symbol, it.sort.convertSort()) }

            val cvc5ConstBounds = bounds
                .map { mkConstApp(it) }
                .map { with(internalizer) { it.internalizeExpr() } }
                .toTypedArray()

            val cvc5PreparedBody = cvc5BodyWithVars.substitute(cvc5VarList.getChildren(), cvc5ConstBounds)

            val body = findConvertedNative(cvc5PreparedBody)
            if (body == null) {
                exprStack.add(expr)
                exprStack.add(cvc5PreparedBody)

                qfVarsAsConsts[expr] = cvc5ConstBounds
                return@with argumentsConversionRequired
            }

            @Suppress("UNCHECKED_CAST")
            body as? KExpr<KBoolSort> ?: error("Body is not properly converted")
            return@with ExprConversionResult(mkQf(body, bounds))
        }

        val bounds = cvc5SubstitutedConsts.map { mkConstDecl(it.symbol, it.sort.convertSort()) }
        val cvc5PreparedBody = cvc5BodyWithVars.substitute(cvc5VarList.getChildren(), cvc5SubstitutedConsts)
        val body = findConvertedNative(cvc5PreparedBody) ?: error("Body must be converted here")

        @Suppress("UNCHECKED_CAST")
        body as? KExpr<KBoolSort> ?: error("Body is not properly converted")

        qfVarsAsConsts.remove(expr)
        ExprConversionResult(mkQf(body, bounds))
    }

    private fun <D : KSort, R : KSort> convertNativeConstArrayExpr(expr: Term): ExprConversionResult = with(ctx) {
        expr.convert(arrayOf(expr.constArrayBase)) { arrayBase: KExpr<R> ->
            mkArrayConst(expr.sort.convertSort() as KArraySort<D, R>, arrayBase)
        }
    }

    private fun mkArray1Select(
        array: KExpr<KArraySort<KSort, KSort>>,
        index: KExpr<KSort>
    ) = ctx.mkArraySelect(array, index)

    private fun mkArray1Store(
        array: KExpr<KArraySort<KSort, KSort>>,
        index: KExpr<KSort>,
        value: KExpr<KSort>
    ) = ctx.mkArrayStore(array, index, value)

    private fun convertNativeConstIntegerExpr(expr: Term): ExprConversionResult = with(ctx) {
        convert { mkIntNum(expr.integerValue) }
    }

    private fun convertNativeConstRealExpr(expr: Term): ExprConversionResult = with(ctx) {
        convert {
            val numDenum = expr.realValue
            mkRealNum(mkIntNum(numDenum.first), mkIntNum(numDenum.second))
        }
    }

    private fun convertNativeBitvectorConstExpr(expr: Term): ExprConversionResult = with(ctx) {
        convert { mkBv(expr.bitVectorValue, expr.bitVectorValue.length.toUInt()) }
    }

    private fun convertNativeRoundingModeConstExpr(expr: Term): ExprConversionResult = with(ctx) {
        convert { mkFpRoundingModeExpr(expr.roundingModeValue.convert()) }
    }

    fun RoundingMode.convert(): KFpRoundingMode = when (this) {
        RoundingMode.ROUND_TOWARD_ZERO -> KFpRoundingMode.RoundTowardZero
        RoundingMode.ROUND_TOWARD_NEGATIVE -> KFpRoundingMode.RoundTowardNegative
        RoundingMode.ROUND_TOWARD_POSITIVE -> KFpRoundingMode.RoundTowardPositive
        RoundingMode.ROUND_NEAREST_TIES_TO_AWAY -> KFpRoundingMode.RoundNearestTiesToAway
        RoundingMode.ROUND_NEAREST_TIES_TO_EVEN -> KFpRoundingMode.RoundNearestTiesToEven
    }

    private fun convertNativeFpToBvExpr(expr: Term, signed: Boolean): ExprConversionResult = with(ctx) {
        expr.convert { roundingMode: KExpr<KFpRoundingModeSort>, fpExpr: KExpr<KFpSort> ->
            mkFpToBvExpr(roundingMode, fpExpr, expr.bvSizeToConvertTo , signed)
        }
    }

    private fun convertNativeBvToFpExpr(expr: Term, signed: Boolean) = with(ctx) {
        expr.convert { roundingMode: KExpr<KFpRoundingModeSort>, bvExpr: KExpr<KBvSort> ->
            mkBvToFpExpr(
                mkFpSort(expr.toFpExponentSize.toUInt(), expr.toFpSignificandSize.toUInt()),
                roundingMode,
                bvExpr,
                signed = signed
            )
        }
    }

    private fun convertNativeFloatingPointConstExpr(expr: Term): ExprConversionResult = with(ctx) {
        // Kind.CONST_FLOATINGPOINT - created from IEEE-754 bit-vector representation
        val fpTriplet = expr.floatingPointValue // (exponent, significand, bvValue)
        val significandSize = fpTriplet.second.toInt()
        val exponentSize = fpTriplet.first.toInt()
        val bvValue = fpTriplet.third.bitVectorValue
        val fpSignBit = bvValue[0] == '1'
        val fpExponent = mkBv(bvValue.substring(1..exponentSize), exponentSize.toUInt())
        val fpSignificand = mkBv(bvValue.substring(exponentSize + 1), significandSize.toUInt() - 1u)
        convert {
            mkFpCustomSizeBiased(
                significandSize = significandSize.toUInt(),
                exponentSize = exponentSize.toUInt(),
                significand = fpSignificand,
                biasedExponent = fpExponent,
                signBit = fpSignBit
            )
        }
    }

    private fun convertNativeBoolConstExpr(expr: Term): ExprConversionResult = with(ctx) {
        when (expr.booleanValue) {
            true -> convert { trueExpr }
            false -> convert { falseExpr }
        }
    }

    private fun convertNativePowExpr(expr: Term): ExprConversionResult = with(ctx) {
        when (expr.kind) {
            Kind.POW -> expr.convert(::mkArithPower)
            Kind.POW2 -> expr.convert { baseExpr: KExpr<KArithSort> ->
                mkArithMul(baseExpr, baseExpr)
            }
            else -> error("expected power expr term, but was $expr")
        }
    }

    fun <T : KSort> Term.convertExpr(): KExpr<T> = convertFromNative()

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Sort.convertSort(): T = cvc5Ctx.convertSort(this) {
        convertNativeSort(this)
    } as? T ?: error("sort is not properly converted")

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> Term.convertDecl(): KDecl<T> = cvc5Ctx.convertDecl(this) {
        convertNativeDecl(this)
    } as? KDecl<T> ?: error("decl is not properly converted")


    open fun convertNativeDecl(decl: Term): KDecl<*> = with(ctx) {
        when {
            decl.sort.isFunction -> {
                val range = decl.sort.functionCodomainSort.convertSort<KSort>()
                val domain = decl.sort.functionDomainSorts.map { it.convertSort<KSort>() }
                val name = decl.symbol
                return mkFuncDecl(name, range, domain)
            }

            decl.kind == Kind.CONSTANT -> {
                return mkConstDecl(decl.symbol, decl.sort.convertSort())
            }

            else -> error("Unexpected term: $decl")
        }
    }

    open fun convertNativeSort(sort: Sort): KSort = with(ctx) {
        when {
            sort.isBoolean -> boolSort
            sort.isBitVector -> mkBvSort(sort.bitVectorSize.toUInt())
            sort.isInteger -> intSort
            sort.isReal -> realSort
            sort.isArray -> mkArraySort(sort.arrayIndexSort.convertSort(), sort.arrayElementSort.convertSort())
            sort.isFloatingPoint -> mkFpSort(
                sort.floatingPointExponentSize.toUInt(),
                sort.floatingPointSignificandSize.toUInt()
            )
            sort.isUninterpretedSort -> mkUninterpretedSort(sort.symbol)
            sort.isRoundingMode -> mkFpRoundingModeSort()

            else -> throw KSolverUnsupportedFeatureException("Sort $sort is not supported now")
        }
    }

    inline fun <T : KSort> Term.convertCascadeBinaryArityOpOrArg(crossinline op: (KExpr<T>, KExpr<T>) -> KExpr<T>) =
        getChildren().let { children ->
            when {
                children.isEmpty() -> error("Arguments of term [this] must not be empty, but was")
                children.size == 1 -> children.first().convert<T, T> { it }
                children.size == 2 -> convert(children, op)
                else -> {
                    val accumulate = { a: KExpr<*>, b: KExpr<*> -> op(a.uncheckedCast(), b.uncheckedCast()) }

                    ensureArgsConvertedAndConvert(this, children, expectedSize = children.size) { convertedArgs ->
                        convertedArgs.subList(2, convertedArgs.size).fold(
                            initial = accumulate(convertedArgs[0], convertedArgs[1])
                        ) { acc, curr ->
                            accumulate(acc, curr)
                        }
                    }
                }
            }
        }

    inline fun <T : KSort, A0 : KSort> Term.convert(op: (KExpr<A0>) -> KExpr<T>) =
        convert(getChildren(), op)

    inline fun <T : KSort, A0 : KSort, A1 : KSort> Term.convert(op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>) =
        convert(getChildren(), op)

    inline fun <S : KSort> Term.convertReduced(op: (KExpr<S>, KExpr<S>) -> KExpr<S>) =
        convertReduced(getChildren(), op)

    inline fun <S : KSort> Term.convertChainable(
        op: (KExpr<S>, KExpr<S>) -> KExpr<KBoolSort>,
        chainOp: (List<KExpr<KBoolSort>>) -> KExpr<KBoolSort>
    ) = when {
        numChildren == 2 -> convert(getChildren(), op)
        numChildren > 2 -> convertList { args: List<KExpr<S>> ->
            chainOp(args.zip(args.subList(1, args.size), op))
        }
        else -> error("Children count must me >= 2, but was $numChildren")
    }

    @Suppress("MaxLineLength")
    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> Term.convert(op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>) =
        convert(getChildren(), op)

    @Suppress("MaxLineLength")
    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort> Term.convert(op: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>) =
        convert(getChildren(), op)

    inline fun <T : KSort, A0 : KSort> Term.convertList(op: (List<KExpr<A0>>) -> KExpr<T>) =
        convertList(getChildren(), op)

    fun Term.getChildren(): Array<Term> = Array(numChildren) { i -> getChild(i) }
}
