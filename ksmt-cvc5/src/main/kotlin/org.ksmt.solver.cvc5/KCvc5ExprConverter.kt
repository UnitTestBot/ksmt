package org.ksmt.solver.cvc5

import io.github.cvc5.*
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.*
import org.ksmt.solver.util.KExprConverterBase
import org.ksmt.sort.*
import org.ksmt.utils.asExpr

open class KCvc5ExprConverter(
    private val ctx: KContext,
    private val cvc5Ctx: KCvc5Context
) : KExprConverterBase<Term>() {

    private val internalizer = KCvc5ExprInternalizer(cvc5Ctx)

    override fun findConvertedNative(expr: Term): KExpr<*>? = cvc5Ctx.findConvertedExpr(expr)

    override fun saveConvertedNative(native: Term, converted: KExpr<*>) {
        cvc5Ctx.saveConvertedExpr(native, converted)
    }

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
            Kind.LEQ -> expr.convertChainable<KArithSort>(::mkArithLe, ::mkAnd)
            Kind.GEQ -> expr.convertChainable<KArithSort>(::mkArithGe, ::mkAnd)
            Kind.LT -> expr.convertChainable<KArithSort>(::mkArithLt, ::mkAnd)
            Kind.GT -> expr.convertChainable<KArithSort>(::mkArithGt, ::mkAnd)

            // bool
            Kind.NOT -> expr.convert(::mkNot)
            Kind.AND -> expr.convertList(::mkAnd)
            Kind.OR -> expr.convertList(::mkOr)
            Kind.XOR -> expr.convertReduced(::mkXor)
            Kind.IMPLIES -> expr.convert(::mkImplies)

            // quantifiers
            Kind.FORALL -> convertNativeQuantifierExpr(expr)
            Kind.EXISTS -> convertNativeQuantifierExpr(expr)

            Kind.VARIABLE_LIST -> error("cvc5 Variable list should not be handled here. It maps to List<KDecl<*>>")
            Kind.VARIABLE -> error("cvc5 Variable should not be handled here. It maps to KDecl<*>")
            Kind.INST_PATTERN -> TODO("patterns should not be handled here. ksmt has no impl of patterns")
            Kind.INST_PATTERN_LIST -> TODO("patterns should not be handled here. ksmt has no impl of patterns")
            Kind.INST_NO_PATTERN -> TODO("patterns should not be handled here. ksmt has no impl of patterns")
            Kind.INST_POOL -> TODO("annotations of instantiations are not supported in ksmt now (experimental in cvc5 1.0.2)")
            Kind.INST_ADD_TO_POOL -> TODO("annotations of instantiations are not supported in ksmt now (experimental in cvc5 1.0.2)")
            Kind.SKOLEM_ADD_TO_POOL -> TODO("annotations of instantiations are not supported in ksmt now (experimental in cvc5 1.0.2)")
            Kind.INST_ATTRIBUTE -> TODO("attributes for a quantifier are not supported in ksmt now")

            Kind.HO_APPLY -> TODO("no direct mapping in ksmt")
            Kind.CARDINALITY_CONSTRAINT -> TODO("experimental in cvc5 1.0.2")
            Kind.APPLY_UF -> expr.getChildren().let { children ->
                expr.convertList(children.copyOfRange(1, children.size)) { args: List<KExpr<KSort>> ->
                    val fTerm = children.first().convertDecl<KSort>()
                    mkApp(fTerm, args)
                }
            }

            Kind.LAMBDA -> error("lambdas used only in interpretations, and they handled separately")
            Kind.SEXPR -> TODO("no direct impl in ksmt now (experimental in cvc5 1.0.2)")

            // arith
            Kind.ABS -> expr.convert { arithExpr: KExpr<KArithSort> ->
                val geExpr = if (arithExpr.sort is KIntSort) arithExpr.asExpr(intSort) ge 0.expr
                else arithExpr.asExpr(realSort) ge mkRealNum(0)
                mkIte(geExpr, arithExpr, -arithExpr)
            }
            Kind.ADD -> expr.convertList<KArithSort, KArithSort>(::mkArithAdd)
            Kind.SUB -> expr.convertList<KArithSort, KArithSort>(::mkArithSub)
            Kind.MULT -> expr.convertList<KArithSort, KArithSort>(::mkArithMul)
            Kind.NEG -> expr.convert<KArithSort, KArithSort>(::mkArithUnaryMinus)
            Kind.SQRT -> TODO("No direct mapping of sqrt on real in ksmt")
            Kind.POW -> convertNativePowExpr(expr)
            Kind.POW2 -> convertNativePowExpr(expr)
            Kind.INTS_MODULUS -> expr.convert(::mkIntMod)
            Kind.INTS_DIVISION -> expr.convert<KArithSort, KArithSort, KArithSort>(::mkArithDiv)
            Kind.DIVISION -> expr.convert<KArithSort, KArithSort, KArithSort>(::mkArithDiv)
            Kind.ARCCOTANGENT -> TODO("No direct mapping in ksmt")
            Kind.ARCSECANT -> TODO("No direct mapping in ksmt")
            Kind.ARCCOSECANT -> TODO("No direct mapping in ksmt")
            Kind.ARCTANGENT -> TODO("No direct mapping in ksmt")
            Kind.ARCCOSINE -> TODO("No direct mapping in ksmt")
            Kind.ARCSINE -> TODO("No direct mapping in ksmt")
            Kind.COTANGENT -> TODO("No direct mapping in ksmt")
            Kind.SECANT -> TODO("No direct mapping in ksmt")
            Kind.COSECANT -> TODO("No direct mapping in ksmt")
            Kind.TANGENT -> TODO("No direct mapping in ksmt")
            Kind.COSINE -> TODO("No direct mapping in ksmt")
            Kind.SINE -> TODO("No direct mapping in ksmt")
            Kind.EXPONENTIAL -> TODO("No direct mapping in ksmt")

            Kind.TO_REAL -> expr.convert(::mkIntToReal)
            Kind.TO_INTEGER -> expr.convert(::mkRealToInt)
            Kind.IS_INTEGER -> expr.convert(::mkRealIsInt)
            Kind.PI -> TODO("PI real const has no mapping in ksmt")
            Kind.DIVISIBLE -> expr.convert { arExpr: KExpr<KIntSort> -> mkIntMod(arExpr, expr.intDivisibleArg.expr) eq 0.expr }
            Kind.IAND -> TODO("No direct mapping in ksmt. btw int to bv is not supported in ksmt")
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
            Kind.BITVECTOR_ULTBV -> TODO("No direct mapping of ${Kind.BITVECTOR_ULTBV} in ksmt")
            Kind.BITVECTOR_SLTBV -> TODO("No direct mapping of ${Kind.BITVECTOR_SLTBV} in ksmt")
            Kind.BITVECTOR_ITE -> TODO("No direct mapping of ${Kind.BITVECTOR_ITE} in ksmt")

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


            Kind.INT_TO_BITVECTOR -> TODO("No direct mapping of ${Kind.INT_TO_BITVECTOR} in ksmt")
            // bitvector -> non-negative integer
            Kind.BITVECTOR_TO_NAT -> expr.convert { bvExpr: KExpr<KBvSort> ->
                mkBv2IntExpr(bvExpr, false)
            }
            Kind.BITVECTOR_COMP -> TODO("No direct mapping of ${Kind.BITVECTOR_COMP} in ksmt")


            // fp
            Kind.FLOATINGPOINT_FP -> expr.convert { bvSign: KExpr<KBv1Sort>, bvExponent: KExpr<KBvSort>, bvSignificand: KExpr<KBvSort> ->
                // Kind.FLOATINGPOINT_FP contains: significand term of bit-vector Sort of significand size - 1 (significand without hidden bit)
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
            Kind.FLOATINGPOINT_TO_FP_FROM_FP -> expr.convert { roundingMode: KExpr<KFpRoundingModeSort>, fpExpr: KExpr<KFpSort> ->
                mkFpToFpExpr(
                    mkFpSort(expr.toFpExponentSize.toUInt(), expr.toFpSignificandSize.toUInt()),
                    roundingMode,
                    fpExpr
                )
            }
            Kind.FLOATINGPOINT_TO_FP_FROM_REAL -> expr.convert { roundingMode: KExpr<KFpRoundingModeSort>, realExpr: KExpr<KRealSort> ->
                mkRealToFpExpr(
                    mkFpSort(expr.toFpExponentSize.toUInt(), expr.toFpSignificandSize.toUInt()),
                    roundingMode,
                    realExpr
                )
            }
            Kind.FLOATINGPOINT_TO_FP_FROM_SBV -> convertNativeBvToFpExpr(expr, true)
            Kind.FLOATINGPOINT_TO_FP_FROM_UBV -> convertNativeBvToFpExpr(expr, false)
            Kind.FLOATINGPOINT_TO_UBV -> convertNativeFpToBvExpr(expr, false)
            Kind.FLOATINGPOINT_TO_SBV -> convertNativeFpToBvExpr(expr, true)
            Kind.FLOATINGPOINT_TO_REAL -> expr.convert { fpExpr: KExpr<KFpSort> ->  mkFpToRealExpr(fpExpr) }

            // array
            Kind.SELECT -> expr.convert(::mkArraySelect)
            Kind.STORE -> expr.convert(::mkArrayStore)

            Kind.EQ_RANGE -> TODO("experimental feature in cvc5 1.0.2")

            Kind.APPLY_CONSTRUCTOR -> TODO("currently ksmt does not support datatypes")
            Kind.APPLY_SELECTOR -> TODO("currently ksmt does not support datatypes")
            Kind.APPLY_TESTER -> TODO("currently ksmt does not support datatypes")
            Kind.APPLY_UPDATER -> TODO("currently ksmt does not support datatypes")

            Kind.MATCH -> TODO("currently ksmt does not support grammars")
            Kind.MATCH_CASE -> TODO("currently ksmt does not support grammars")
            Kind.MATCH_BIND_CASE -> TODO("currently ksmt does not support grammars")
            Kind.TUPLE_PROJECT -> TODO("currently ksmt does not support tuples")

            Kind.SEP_NIL -> TODO("currently ksmt does not support separation logic")
            Kind.SEP_EMP -> TODO("currently ksmt does not support separation logic")
            Kind.SEP_PTO -> TODO("currently ksmt does not support separation logic")
            Kind.SEP_STAR -> TODO("currently ksmt does not support separation logic")
            Kind.SEP_WAND -> TODO("currently ksmt does not support separation logic")

            Kind.SET_EMPTY -> TODO("currently ksmt does not support sets")
            Kind.SET_UNION -> TODO("currently ksmt does not support sets")
            Kind.SET_INTER -> TODO("currently ksmt does not support sets")
            Kind.SET_MINUS -> TODO("currently ksmt does not support sets")
            Kind.SET_SUBSET -> TODO("currently ksmt does not support sets")
            Kind.SET_MEMBER -> TODO("currently ksmt does not support sets")
            Kind.SET_SINGLETON -> TODO("currently ksmt does not support sets")
            Kind.SET_INSERT -> TODO("currently ksmt does not support sets")
            Kind.SET_CARD -> TODO("currently ksmt does not support sets")
            Kind.SET_COMPLEMENT -> TODO("currently ksmt does not support sets")
            Kind.SET_UNIVERSE -> TODO("currently ksmt does not support sets")
            Kind.SET_COMPREHENSION -> TODO("currently ksmt does not support sets")
            Kind.SET_CHOOSE -> TODO("currently ksmt does not support sets")
            Kind.SET_IS_SINGLETON -> TODO("currently ksmt does not support sets")
            Kind.SET_MAP -> TODO("currently ksmt does not support sets")
            Kind.SET_FILTER -> TODO("currently ksmt does not support sets")
            Kind.SET_FOLD -> TODO("currently ksmt does not support sets")

            Kind.RELATION_JOIN -> TODO("currently ksmt does not support relations")
            Kind.RELATION_PRODUCT -> TODO("currently ksmt does not support relations")
            Kind.RELATION_TRANSPOSE -> TODO("currently ksmt does not support relations")
            Kind.RELATION_TCLOSURE -> TODO("currently ksmt does not support relations")
            Kind.RELATION_JOIN_IMAGE -> TODO("currently ksmt does not support relations")
            Kind.RELATION_IDEN -> TODO("currently ksmt does not support relations")
            Kind.RELATION_GROUP -> TODO("currently ksmt does not support relations")
            Kind.RELATION_AGGREGATE -> TODO("currently ksmt does not support relations")
            Kind.RELATION_PROJECT -> TODO("currently ksmt does not support relations")

            Kind.BAG_EMPTY -> TODO("currently ksmt does not support bags")
            Kind.BAG_UNION_MAX -> TODO("currently ksmt does not support bags")
            Kind.BAG_UNION_DISJOINT -> TODO("currently ksmt does not support bags")
            Kind.BAG_INTER_MIN -> TODO("currently ksmt does not support bags")
            Kind.BAG_DIFFERENCE_SUBTRACT -> TODO("currently ksmt does not support bags")
            Kind.BAG_DIFFERENCE_REMOVE -> TODO("currently ksmt does not support bags")
            Kind.BAG_SUBBAG -> TODO("currently ksmt does not support bags")
            Kind.BAG_COUNT -> TODO("currently ksmt does not support bags")
            Kind.BAG_MEMBER -> TODO("currently ksmt does not support bags")
            Kind.BAG_DUPLICATE_REMOVAL -> TODO("currently ksmt does not support bags")
            Kind.BAG_MAKE -> TODO("currently ksmt does not support bags")
            Kind.BAG_CARD -> TODO("currently ksmt does not support bags")
            Kind.BAG_CHOOSE -> TODO("currently ksmt does not support bags")
            Kind.BAG_IS_SINGLETON -> TODO("currently ksmt does not support bags")
            Kind.BAG_FROM_SET -> TODO("currently ksmt does not support bags")
            Kind.BAG_TO_SET -> TODO("currently ksmt does not support bags")
            Kind.BAG_MAP -> TODO("currently ksmt does not support bags")
            Kind.BAG_FILTER -> TODO("currently ksmt does not support bags")
            Kind.BAG_FOLD -> TODO("currently ksmt does not support bags")
            Kind.BAG_PARTITION -> TODO("currently ksmt does not support bags")

            Kind.TABLE_PRODUCT -> TODO("currently ksmt does not support tables")
            Kind.TABLE_PROJECT -> TODO("currently ksmt does not support tables")
            Kind.TABLE_AGGREGATE -> TODO("currently ksmt does not support tables")
            Kind.TABLE_JOIN -> TODO("currently ksmt does not support tables")
            Kind.TABLE_GROUP -> TODO("currently ksmt does not support tables")

            Kind.STRING_CONCAT -> TODO("currently ksmt does not support strings")
            Kind.STRING_IN_REGEXP -> TODO("currently ksmt does not support strings")
            Kind.STRING_LENGTH -> TODO("currently ksmt does not support strings")
            Kind.STRING_SUBSTR -> TODO("currently ksmt does not support strings")
            Kind.STRING_UPDATE -> TODO("currently ksmt does not support strings")
            Kind.STRING_CHARAT -> TODO("currently ksmt does not support strings")
            Kind.STRING_CONTAINS -> TODO("currently ksmt does not support strings")
            Kind.STRING_INDEXOF -> TODO("currently ksmt does not support strings")
            Kind.STRING_INDEXOF_RE -> TODO("currently ksmt does not support strings")
            Kind.STRING_REPLACE -> TODO("currently ksmt does not support strings")
            Kind.STRING_REPLACE_ALL -> TODO("currently ksmt does not support strings")
            Kind.STRING_REPLACE_RE -> TODO("currently ksmt does not support strings")
            Kind.STRING_REPLACE_RE_ALL -> TODO("currently ksmt does not support strings")
            Kind.STRING_TO_LOWER -> TODO("currently ksmt does not support strings")
            Kind.STRING_TO_UPPER -> TODO("currently ksmt does not support strings")
            Kind.STRING_REV -> TODO("currently ksmt does not support strings")
            Kind.STRING_TO_CODE -> TODO("currently ksmt does not support strings")
            Kind.STRING_FROM_CODE -> TODO("currently ksmt does not support strings")
            Kind.STRING_LT -> TODO("currently ksmt does not support strings")
            Kind.STRING_LEQ -> TODO("currently ksmt does not support strings")
            Kind.STRING_PREFIX -> TODO("currently ksmt does not support strings")
            Kind.STRING_SUFFIX -> TODO("currently ksmt does not support strings")
            Kind.STRING_IS_DIGIT -> TODO("currently ksmt does not support strings")
            Kind.STRING_FROM_INT -> TODO("currently ksmt does not support strings")
            Kind.STRING_TO_INT -> TODO("currently ksmt does not support strings")
            Kind.CONST_STRING -> TODO("currently ksmt does not support strings")
            Kind.STRING_TO_REGEXP -> TODO("currently ksmt does not support strings")

            Kind.REGEXP_CONCAT -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_UNION -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_INTER -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_DIFF -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_STAR -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_PLUS -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_OPT -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_RANGE -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_REPEAT -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_LOOP -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_NONE -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_ALL -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_ALLCHAR -> TODO("currently ksmt does not support regular expressions")
            Kind.REGEXP_COMPLEMENT -> TODO("currently ksmt does not support regular expressions")

            Kind.SEQ_CONCAT -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_LENGTH -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_EXTRACT -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_UPDATE -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_AT -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_CONTAINS -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_INDEXOF -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_REPLACE -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_REPLACE_ALL -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_REV -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_PREFIX -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_SUFFIX -> TODO("currently ksmt does not support sequences")
            Kind.CONST_SEQUENCE -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_UNIT -> TODO("currently ksmt does not support sequences")
            Kind.SEQ_NTH -> TODO("currently ksmt does not support sequences")

            Kind.WITNESS -> error("no direct mapping in ksmt")
            Kind.LAST_KIND -> error("Marks the upper-bound of this enumeration, not op kind")
            Kind.INTERNAL_KIND -> TODO()
            Kind.UNDEFINED_KIND -> TODO()
            Kind.NULL_TERM -> TODO()
            null -> error("kind can't be null")
        }

    }

    @Suppress("UNCHECKED_CAST")
    private fun convertNativeQuantifierExpr(expr: Term) = with(ctx) {
        val mkQf = when (expr.kind) {
            Kind.FORALL -> ::mkUniversalQuantifier
            Kind.EXISTS -> ::mkExistentialQuantifier
            else -> error("Unknown term of quantifier. Kind of term: ${expr.kind}")
        }

        val cvc5VarList = expr.getChild(0)
        val cvc5BodyWithVars = expr.getChild(1)

        // fresh bounds
        val bounds = convertNativeVariableListExpr(cvc5VarList)
        val cvc5ConstBounds = bounds.map { mkConstApp(it) }
            .map { with(internalizer) { it.internalizeExpr() } }

        val cvc5PreparedBody = cvc5BodyWithVars.substitute(cvc5VarList.getChildren(), cvc5ConstBounds.toTypedArray())

        expr.convertList((cvc5ConstBounds + cvc5PreparedBody).toTypedArray()) { args: List<KExpr<KSort>> ->
            val body = args.last() as KExpr<KBoolSort>
            val boundConsts = args.subList(0, args.lastIndex) as List<KConst<*>>

            require(bounds.toSet() == boundConsts.toSet())
            mkQf(body, boundConsts.map { it.decl })
        }
    }

    private fun convertNativeVariableExpr(expr: Term): KDecl<*> = with(ctx) {
        mkFreshConstDecl(expr.symbol, expr.sort.convertSort())
    }

    private fun convertNativeVariableListExpr(expr: Term): List<KDecl<*>> = expr.getChildren()
        .map(::convertNativeVariableExpr)

    private fun <D: KSort, R: KSort> convertNativeConstArrayExpr(expr: Term): ExprConversionResult = with(ctx) {
        expr.convert(arrayOf(expr.constArrayBase)) { arrayBase: KExpr<R> ->
            mkArrayConst(expr.sort.convertSort() as KArraySort<D, R>, arrayBase)
        }
    }

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

    fun RoundingMode.convert(): KFpRoundingMode = when(this) {
        RoundingMode.ROUND_TOWARD_ZERO -> KFpRoundingMode.RoundTowardZero
        RoundingMode.ROUND_TOWARD_NEGATIVE -> KFpRoundingMode.RoundTowardNegative
        RoundingMode.ROUND_TOWARD_POSITIVE -> KFpRoundingMode.RoundTowardPositive
        RoundingMode.ROUND_NEAREST_TIES_TO_AWAY -> KFpRoundingMode.RoundNearestTiesToAway
        RoundingMode.ROUND_NEAREST_TIES_TO_EVEN -> KFpRoundingMode.RoundNearestTiesToEven
    }


    @Suppress("UNUSED_PARAMETER")
    private fun convertNativeFpToBvExpr(expr: Term, signedBv: Boolean): ExprConversionResult = with(ctx) {
        TODO("ksmt mkFpToBvExpr requires rounding mode, but cvc5 does not provide it")
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
        val fpTriplet = expr.floatingPointValue // (Exponent, Significand, BvValue)
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
            Kind.POW -> expr.convert<KArithSort, KArithSort, KArithSort>(::mkArithPower)
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
            sort.isFloatingPoint -> mkFpSort(sort.floatingPointExponentSize.toUInt(), sort.floatingPointSignificandSize.toUInt())
            sort.isUninterpretedSort -> mkUninterpretedSort(sort.symbol)
            sort.isRoundingMode -> mkFpRoundingModeSort()

            else -> TODO("Sort $sort is not supported now")
        }
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <T : KSort> Term.convertCascadeBinaryArityOpOrArg(crossinline op: (KExpr<T>, KExpr<T>) -> KExpr<T>) = getChildren().let { children ->
        when {
            children.isEmpty() -> error("Arguments of term [this] must not be empty, but was")
            children.size == 1 -> children.first().convert<T, T> { it }
            children.size == 2 -> convert(children, op)
            else -> {
                val accumulate = { a: KExpr<*>, b: KExpr<*> -> op(a as KExpr<T>, b as KExpr<T>) }

                ensureArgsConvertedAndConvert(this, children, expectedSize = children.size) { convertedArgs ->
                    convertedArgs.subList(2, convertedArgs.size).fold(
                        initial = accumulate(convertedArgs[0], convertedArgs[1])) { acc, curr ->
                            accumulate(acc, curr)
                        }
                }
            }
        }
    }

    inline fun <T : KSort, A0 : KSort> Term.convert(op: (KExpr<A0>) -> KExpr<T>) =
        convert(getChildren(), op)

    inline fun <T : KSort, A0 : KSort, A1: KSort> Term.convert(op: (KExpr<A0>, KExpr<A1>) -> KExpr<T>) =
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

    inline fun <T : KSort, A0 : KSort, A1: KSort, A2: KSort> Term.convert(op: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>) =
        convert(getChildren(), op)

    inline fun <T : KSort, A0 : KSort, A1: KSort, A2: KSort, A3: KSort> Term.convert(op: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>) =
        convert(getChildren(), op)

    inline fun <T : KSort, A0 : KSort> Term.convertList(op: (List<KExpr<A0>>) -> KExpr<T>) =
        convertList(getChildren(), op)

    fun Term.getChildren(): Array<Term> = Array(numChildren) { i -> getChild(i) }
}