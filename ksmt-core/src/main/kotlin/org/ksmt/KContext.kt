package org.ksmt

import java.lang.Double.longBitsToDouble
import java.lang.Float.intBitsToFloat
import org.ksmt.cache.Cache0
import org.ksmt.cache.Cache1
import org.ksmt.cache.Cache2
import org.ksmt.cache.Cache3
import org.ksmt.cache.Cache4
import org.ksmt.cache.Cache5
import org.ksmt.cache.mkCache
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
import org.ksmt.decl.KArrayConstDecl
import org.ksmt.decl.KArraySelectDecl
import org.ksmt.decl.KArrayStoreDecl
import org.ksmt.decl.KBvAddDecl
import org.ksmt.decl.KBvAndDecl
import org.ksmt.decl.KBvMulDecl
import org.ksmt.decl.KBvNAndDecl
import org.ksmt.decl.KBvNorDecl
import org.ksmt.decl.KBvNotDecl
import org.ksmt.decl.KBvOrDecl
import org.ksmt.decl.KBvReductionAndDecl
import org.ksmt.decl.KBvReductionOrDecl
import org.ksmt.decl.KBvSignedDivDecl
import org.ksmt.decl.KBvSubDecl
import org.ksmt.decl.KBvUnsignedDivDecl
import org.ksmt.decl.KBvXNorDecl
import org.ksmt.decl.KBvXorDecl
import org.ksmt.decl.KBitVec16ValueDecl
import org.ksmt.decl.KBitVec32ValueDecl
import org.ksmt.decl.KBitVec64ValueDecl
import org.ksmt.decl.KBitVec8ValueDecl
import org.ksmt.decl.KBitVecCustomSizeValueDecl
import org.ksmt.decl.KBv2IntDecl
import org.ksmt.decl.KBvArithShiftRightDecl
import org.ksmt.decl.KBvAddNoOverflowDecl
import org.ksmt.decl.KBvAddNoUnderflowDecl
import org.ksmt.decl.KBvDivNoOverflowDecl
import org.ksmt.decl.KBvLogicalShiftRightDecl
import org.ksmt.decl.KBvMulNoOverflowDecl
import org.ksmt.decl.KBvMulNoUnderflowDecl
import org.ksmt.decl.KBvNegationDecl
import org.ksmt.decl.KBvNegNoOverflowDecl
import org.ksmt.decl.KBvRotateLeftDecl
import org.ksmt.decl.KBvRotateRightDecl
import org.ksmt.decl.KBvSignedGreaterOrEqualDecl
import org.ksmt.decl.KBvSignedGreaterDecl
import org.ksmt.decl.KBvShiftLeftDecl
import org.ksmt.decl.KBvSignedLessOrEqualDecl
import org.ksmt.decl.KBvSignedLessDecl
import org.ksmt.decl.KBitVec1ValueDecl
import org.ksmt.decl.KBvSignedModDecl
import org.ksmt.decl.KBvSignedRemDecl
import org.ksmt.decl.KBvSubNoOverflowDecl
import org.ksmt.decl.KBvUnsignedGreaterOrEqualDecl
import org.ksmt.decl.KBvUnsignedGreaterDecl
import org.ksmt.decl.KBvUnsignedLessOrEqualDecl
import org.ksmt.decl.KBvUnsignedLessDecl
import org.ksmt.decl.KBvUnsignedRemDecl
import org.ksmt.decl.KBvConcatDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDistinctDecl
import org.ksmt.decl.KEqDecl
import org.ksmt.decl.KBvExtractDecl
import org.ksmt.decl.KFalseDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.decl.KImpliesDecl
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
import org.ksmt.decl.KXorDecl
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KBv2IntExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
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
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvSignedDivExpr
import org.ksmt.expr.KBvSignedGreaterOrEqualExpr
import org.ksmt.expr.KBvSignedGreaterExpr
import org.ksmt.expr.KBvShiftLeftExpr
import org.ksmt.expr.KBvSignedLessOrEqualExpr
import org.ksmt.expr.KBvSignedLessExpr
import org.ksmt.expr.KBvSignedModExpr
import org.ksmt.expr.KBvSignedRemExpr
import org.ksmt.expr.KBvSubExpr
import org.ksmt.expr.KBvSubNoOverflowExpr
import org.ksmt.expr.KBvUnsignedDivExpr
import org.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.KBvUnsignedGreaterExpr
import org.ksmt.expr.KBvUnsignedLessOrEqualExpr
import org.ksmt.expr.KBvUnsignedLessExpr
import org.ksmt.expr.KBvUnsignedRemExpr
import org.ksmt.expr.KBvXNorExpr
import org.ksmt.expr.KBvXorExpr
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KBvExtractExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIntNumExpr
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
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KBvRepeatExpr
import org.ksmt.expr.KBvSignExtensionExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvCustomSizeSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpCustomSizeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import java.math.BigInteger
import org.ksmt.decl.KBvRotateLeftIndexedDecl
import org.ksmt.decl.KBvRotateRightIndexedDecl
import org.ksmt.decl.KBvSubNoUnderflowDecl
import org.ksmt.decl.KBvToFpDecl
import org.ksmt.decl.KFp128Decl
import org.ksmt.decl.KFp16Decl
import org.ksmt.decl.KFp32Decl
import org.ksmt.decl.KFp64Decl
import org.ksmt.decl.KFpAbsDecl
import org.ksmt.decl.KFpAddDecl
import org.ksmt.decl.KFpCustomSizeDecl
import org.ksmt.decl.KFpDecl
import org.ksmt.decl.KFpDivDecl
import org.ksmt.decl.KFpEqualDecl
import org.ksmt.decl.KFpFromBvDecl
import org.ksmt.decl.KFpFusedMulAddDecl
import org.ksmt.decl.KFpGreaterDecl
import org.ksmt.decl.KFpGreaterOrEqualDecl
import org.ksmt.decl.KFpIsInfiniteDecl
import org.ksmt.decl.KFpIsNaNDecl
import org.ksmt.decl.KFpIsNegativeDecl
import org.ksmt.decl.KFpIsNormalDecl
import org.ksmt.decl.KFpIsPositiveDecl
import org.ksmt.decl.KFpIsSubnormalDecl
import org.ksmt.decl.KFpIsZeroDecl
import org.ksmt.decl.KFpLessDecl
import org.ksmt.decl.KFpLessOrEqualDecl
import org.ksmt.decl.KFpMaxDecl
import org.ksmt.decl.KFpMinDecl
import org.ksmt.decl.KFpMulDecl
import org.ksmt.decl.KFpNegationDecl
import org.ksmt.decl.KFpRemDecl
import org.ksmt.decl.KFpRoundToIntegralDecl
import org.ksmt.decl.KFpRoundingModeDecl
import org.ksmt.decl.KFpSqrtDecl
import org.ksmt.decl.KFpSubDecl
import org.ksmt.decl.KFpToBvDecl
import org.ksmt.decl.KFpToFpDecl
import org.ksmt.decl.KFpToIEEEBvDecl
import org.ksmt.decl.KFpToRealDecl
import org.ksmt.decl.KRealToFpDecl
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvSubNoUnderflowExpr
import org.ksmt.expr.KBvToFpExpr
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
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.utils.booleanSignBit
import org.ksmt.utils.cast
import org.ksmt.utils.extendWithLeadingZeros
import org.ksmt.utils.extractExponent
import org.ksmt.utils.extractSignificand
import org.ksmt.utils.getExponent
import org.ksmt.utils.getHalfPrecisionExponent
import org.ksmt.utils.toBinary
import org.ksmt.utils.uncheckedCast

@Suppress("TooManyFunctions", "LargeClass", "unused")
open class KContext : AutoCloseable {

    /**
     * KContext and all created expressions are only valid as long as
     * the context is active (not closed).
     * @see ensureContextActive
     * */
    var isActive: Boolean = true
        private set

    private val closableResources = mutableListOf<AutoCloseable>()

    override fun close() {
        isActive = false
        closableResources.forEach { it.close() }
        closableResources.clear()
    }

    /*
    * sorts
    * */
    private val boolSortCache = mkClosableCache<KBoolSort> { KBoolSort(this) }
    fun mkBoolSort(): KBoolSort = boolSortCache.createIfContextActive()

    private val arraySortCache = mkContextCheckingCache { domain: KSort, range: KSort ->
        KArraySort(this, domain, range)
    }

    fun <D : KSort, R : KSort> mkArraySort(domain: D, range: R): KArraySort<D, R> =
        arraySortCache.createIfContextActive(domain, range).cast()

    private val intSortCache = mkClosableCache<KIntSort> { KIntSort(this) }
    fun mkIntSort(): KIntSort = intSortCache.createIfContextActive()

    // bit-vec
    private val bvSortCache = mkClosableCache { sizeBits: UInt ->
        when (sizeBits.toInt()) {
            1 -> KBv1Sort(this)
            Byte.SIZE_BITS -> KBv8Sort(this)
            Short.SIZE_BITS -> KBv16Sort(this)
            Int.SIZE_BITS -> KBv32Sort(this)
            Long.SIZE_BITS -> KBv64Sort(this)
            else -> KBvCustomSizeSort(this, sizeBits)
        }
    }

    fun mkBvSort(sizeBits: UInt): KBvSort = bvSortCache.createIfContextActive(sizeBits)
    fun mkBv1Sort(): KBv1Sort = mkBvSort(sizeBits = 1u).cast()
    fun mkBv8Sort(): KBv8Sort = mkBvSort(Byte.SIZE_BITS.toUInt()).cast()
    fun mkBv16Sort(): KBv16Sort = mkBvSort(Short.SIZE_BITS.toUInt()).cast()
    fun mkBv32Sort(): KBv32Sort = mkBvSort(Int.SIZE_BITS.toUInt()).cast()
    fun mkBv64Sort(): KBv64Sort = mkBvSort(Long.SIZE_BITS.toUInt()).cast()

    private val realSortCache = mkClosableCache<KRealSort> { KRealSort(this) }
    fun mkRealSort(): KRealSort = realSortCache.createIfContextActive()

    private val uninterpretedSortCache = mkClosableCache { name: String -> KUninterpretedSort(name, this) }
    fun mkUninterpretedSort(name: String): KUninterpretedSort = uninterpretedSortCache.createIfContextActive(name)

    // floating point
    private val fpSortCache = mkClosableCache { exponentBits: UInt, significandBits: UInt ->
        when {
            exponentBits == KFp16Sort.exponentBits && significandBits == KFp16Sort.significandBits -> KFp16Sort(this)
            exponentBits == KFp32Sort.exponentBits && significandBits == KFp32Sort.significandBits -> KFp32Sort(this)
            exponentBits == KFp64Sort.exponentBits && significandBits == KFp64Sort.significandBits -> KFp64Sort(this)
            exponentBits == KFp128Sort.exponentBits && significandBits == KFp128Sort.significandBits -> KFp128Sort(this)
            else -> KFpCustomSizeSort(this, exponentBits, significandBits)
        }
    }

    fun mkFp16Sort(): KFp16Sort = fpSortCache.createIfContextActive(
        KFp16Sort.exponentBits, KFp16Sort.significandBits
    ).cast()

    fun mkFp32Sort(): KFp32Sort = fpSortCache.createIfContextActive(
        KFp32Sort.exponentBits, KFp32Sort.significandBits
    ).cast()

    fun mkFp64Sort(): KFp64Sort = fpSortCache.createIfContextActive(
        KFp64Sort.exponentBits, KFp64Sort.significandBits
    ).cast()

    fun mkFp128Sort(): KFp128Sort = fpSortCache.createIfContextActive(
        KFp128Sort.exponentBits, KFp128Sort.significandBits
    ).cast()

    fun mkFpSort(exponentBits: UInt, significandBits: UInt): KFpSort =
        fpSortCache.createIfContextActive(exponentBits, significandBits)

    private val roundingModeSortCache = mkClosableCache<KFpRoundingModeSort> { KFpRoundingModeSort(this) }
    fun mkFpRoundingModeSort(): KFpRoundingModeSort = roundingModeSortCache.createIfContextActive()

    // utils
    val boolSort: KBoolSort
        get() = mkBoolSort()

    val intSort: KIntSort
        get() = mkIntSort()

    val realSort: KRealSort
        get() = mkRealSort()

    val bv1Sort: KBv1Sort
        get() = mkBv1Sort()

    /*
    * expressions
    * */
    // bool
    private val andCache = mkContextListCheckingCache { args: List<KExpr<KBoolSort>> ->
        KAndExpr(this, args)
    }

    fun mkAnd(args: List<KExpr<KBoolSort>>): KAndExpr = andCache.createIfContextActive(args)

    @Suppress("MemberVisibilityCanBePrivate")
    fun mkAnd(vararg args: KExpr<KBoolSort>) = mkAnd(args.toList())

    private val orCache = mkContextListCheckingCache { args: List<KExpr<KBoolSort>> ->
        KOrExpr(this, args)
    }

    fun mkOr(args: List<KExpr<KBoolSort>>): KOrExpr = orCache.createIfContextActive(args)

    @Suppress("MemberVisibilityCanBePrivate")
    fun mkOr(vararg args: KExpr<KBoolSort>) = mkOr(args.toList())

    private val notCache = mkContextCheckingCache { arg: KExpr<KBoolSort> ->
        KNotExpr(this, arg)
    }

    fun mkNot(arg: KExpr<KBoolSort>): KNotExpr = notCache.createIfContextActive(arg)

    private val impliesCache = mkContextCheckingCache { p: KExpr<KBoolSort>, q: KExpr<KBoolSort> ->
        KImpliesExpr(this, p, q)
    }

    fun mkImplies(
        p: KExpr<KBoolSort>,
        q: KExpr<KBoolSort>
    ): KImpliesExpr = impliesCache.createIfContextActive(p, q)

    private val xorCache = mkContextCheckingCache { a: KExpr<KBoolSort>, b: KExpr<KBoolSort> ->
        KXorExpr(this, a, b)
    }

    fun mkXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KXorExpr = xorCache.createIfContextActive(a, b)

    private val trueCache = mkClosableCache<KTrue> { KTrue(this) }
    fun mkTrue() = trueCache.createIfContextActive()

    private val falseCache = mkClosableCache<KFalse> { KFalse(this) }
    fun mkFalse() = falseCache.createIfContextActive()

    private val eqCache = mkContextCheckingCache { l: KExpr<KSort>, r: KExpr<KSort> ->
        KEqExpr(this, l, r)
    }

    fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>): KEqExpr<T> =
        eqCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val distinctCache = mkContextListCheckingCache { args: List<KExpr<KSort>> ->
        KDistinctExpr(this, args)
    }

    fun <T : KSort> mkDistinct(args: List<KExpr<T>>): KDistinctExpr<T> =
        distinctCache.createIfContextActive(args.cast()).cast()

    private val iteCache = mkContextCheckingCache { c: KExpr<KBoolSort>, t: KExpr<KSort>, f: KExpr<KSort> ->
        KIteExpr(this, c, t, f)
    }

    fun <T : KSort> mkIte(
        condition: KExpr<KBoolSort>,
        trueBranch: KExpr<T>,
        falseBranch: KExpr<T>
    ): KIteExpr<T> {
        val trueArg: KExpr<KSort> = trueBranch.cast()
        val falseArg: KExpr<KSort> = falseBranch.cast()

        return iteCache.createIfContextActive(condition, trueArg, falseArg).cast()
    }

    infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
    operator fun KExpr<KBoolSort>.not() = mkNot(this)
    infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
    infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
    infix fun KExpr<KBoolSort>.xor(other: KExpr<KBoolSort>) = mkXor(this, other)
    infix fun KExpr<KBoolSort>.implies(other: KExpr<KBoolSort>) = mkImplies(this, other)
    infix fun <T : KSort> KExpr<T>.neq(other: KExpr<T>) = !(this eq other)

    val trueExpr: KTrue
        get() = mkTrue()

    val falseExpr: KFalse
        get() = mkFalse()

    val Boolean.expr
        get() = if (this) trueExpr else falseExpr

    // functions
    /*
    * For builtin declarations e.g. KAndDecl, mkApp must return the same object as a corresponding builder.
    * For example, mkApp(KAndDecl, a, b) and mkAnd(a, b) must end up with the same KAndExpr object.
    * To achieve such behaviour we override apply for all builtin declarations.
    */
    fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = with(decl) { apply(args) }

    private val functionAppCache = mkClosableCache { decl: KDecl<*>, args: List<KExpr<*>> ->
        ensureContextMatch(decl)
        ensureContextMatch(args)
        KFunctionApp(this, decl, args)
    }

    internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>): KApp<T, *> =
        if (args.isEmpty()) mkConstApp(decl) else functionAppCache.createIfContextActive(decl, args).cast()

    private val constAppCache = mkClosableCache { decl: KDecl<*> ->
        ensureContextMatch(decl)
        KConst(this, decl)
    }

    fun <T : KSort> mkConstApp(decl: KDecl<T>): KConst<T> = constAppCache.createIfContextActive(decl).cast()

    fun <T : KSort> mkConst(name: String, sort: T): KApp<T, *> = with(mkConstDecl(name, sort)) { apply() }

    fun <T : KSort> mkFreshConst(name: String, sort: T): KApp<T, *> = with(mkFreshConstDecl(name, sort)) { apply() }

    // array
    private val arrayStoreCache = mkContextCheckingCache { a: KExpr<KArraySort<KSort, KSort>>,
                                                           i: KExpr<KSort>,
                                                           v: KExpr<KSort> ->
        KArrayStore(this, a, i, v)
    }

    fun <D : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
        value: KExpr<R>
    ): KArrayStore<D, R> {
        val arrayArg: KExpr<KArraySort<KSort, KSort>> = array.cast()
        val indexArg: KExpr<KSort> = index.cast()
        val valueArg: KExpr<KSort> = value.cast()

        return arrayStoreCache.createIfContextActive(arrayArg, indexArg, valueArg).cast()
    }

    private val arraySelectCache = mkContextCheckingCache { array: KExpr<KArraySort<KSort, KSort>>,
                                                            index: KExpr<KSort> ->
        KArraySelect(this, array, index)
    }

    fun <D : KSort, R : KSort> mkArraySelect(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>
    ): KArraySelect<D, R> = arraySelectCache.createIfContextActive(array.cast(), index.cast()).cast()

    private val arrayConstCache = mkContextCheckingCache { array: KArraySort<KSort, KSort>,
                                                           value: KExpr<KSort> ->
        KArrayConst(this, array, value)
    }

    fun <D : KSort, R : KSort> mkArrayConst(
        arraySort: KArraySort<D, R>,
        value: KExpr<R>
    ): KArrayConst<D, R> = arrayConstCache.createIfContextActive(arraySort.cast(), value.cast()).cast()

    private val functionAsArrayCache = mkContextCheckingCache { function: KFuncDecl<KSort> ->
        KFunctionAsArray<KSort, KSort>(this, function)
    }

    fun <D : KSort, R : KSort> mkFunctionAsArray(function: KFuncDecl<R>): KFunctionAsArray<D, R> =
        functionAsArrayCache.createIfContextActive(function.cast()).cast()

    private val arrayLambdaCache = mkContextCheckingCache { indexVar: KDecl<*>, body: KExpr<*> ->
        KArrayLambda(this, indexVar, body)
    }

    fun <D : KSort, R : KSort> mkArrayLambda(indexVar: KDecl<D>, body: KExpr<R>): KArrayLambda<D, R> =
        arrayLambdaCache.createIfContextActive(indexVar, body).cast()

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
        mkArrayStore(this, index, value)

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) = mkArraySelect(this, index)

    // arith
    private val arithAddCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KAddArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithAdd(args: List<KExpr<T>>): KAddArithExpr<T> =
        arithAddCache.createIfContextActive(args.cast()).cast()

    private val arithMulCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KMulArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithMul(args: List<KExpr<T>>): KMulArithExpr<T> =
        arithMulCache.createIfContextActive(args.cast()).cast()

    private val arithSubCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KSubArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithSub(args: List<KExpr<T>>): KSubArithExpr<T> =
        arithSubCache.createIfContextActive(args.cast()).cast()

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KArithSort<T>> mkArithAdd(vararg args: KExpr<T>) = mkArithAdd(args.toList())

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KArithSort<T>> mkArithMul(vararg args: KExpr<T>) = mkArithMul(args.toList())

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KArithSort<T>> mkArithSub(vararg args: KExpr<T>) = mkArithSub(args.toList())

    private val arithUnaryMinusCache = mkContextCheckingCache { arg: KExpr<KArithSort<*>> ->
        KUnaryMinusArithExpr(this, arg)
    }

    fun <T : KArithSort<T>> mkArithUnaryMinus(arg: KExpr<T>): KUnaryMinusArithExpr<T> =
        arithUnaryMinusCache.createIfContextActive(arg.cast()).cast()

    private val arithDivCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>,
                                                         r: KExpr<KArithSort<*>> ->
        KDivArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KDivArithExpr<T> =
        arithDivCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithPowerCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>,
                                                           r: KExpr<KArithSort<*>> ->
        KPowerArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KPowerArithExpr<T> =
        arithPowerCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithLtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>,
                                                        r: KExpr<KArithSort<*>> ->
        KLtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KLtArithExpr<T> =
        arithLtCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithLeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>,
                                                        r: KExpr<KArithSort<*>> ->
        KLeArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KLeArithExpr<T> =
        arithLeCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithGtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>,
                                                        r: KExpr<KArithSort<*>> ->
        KGtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KGtArithExpr<T> =
        arithGtCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithGeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>,
                                                        r: KExpr<KArithSort<*>> ->
        KGeArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KGeArithExpr<T> =
        arithGeCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    operator fun <T : KArithSort<T>> KExpr<T>.plus(other: KExpr<T>) = mkArithAdd(this, other)
    operator fun <T : KArithSort<T>> KExpr<T>.times(other: KExpr<T>) = mkArithMul(this, other)
    operator fun <T : KArithSort<T>> KExpr<T>.minus(other: KExpr<T>) = mkArithSub(this, other)
    operator fun <T : KArithSort<T>> KExpr<T>.unaryMinus() = mkArithUnaryMinus(this)
    operator fun <T : KArithSort<T>> KExpr<T>.div(other: KExpr<T>) = mkArithDiv(this, other)
    fun <T : KArithSort<T>> KExpr<T>.power(other: KExpr<T>) = mkArithPower(this, other)

    infix fun <T : KArithSort<T>> KExpr<T>.lt(other: KExpr<T>) = mkArithLt(this, other)
    infix fun <T : KArithSort<T>> KExpr<T>.le(other: KExpr<T>) = mkArithLe(this, other)
    infix fun <T : KArithSort<T>> KExpr<T>.gt(other: KExpr<T>) = mkArithGt(this, other)
    infix fun <T : KArithSort<T>> KExpr<T>.ge(other: KExpr<T>) = mkArithGe(this, other)

    // integer
    private val intModCache = mkContextCheckingCache { l: KExpr<KIntSort>,
                                                       r: KExpr<KIntSort> ->
        KModIntExpr(this, l, r)
    }

    fun mkIntMod(
        lhs: KExpr<KIntSort>,
        rhs: KExpr<KIntSort>
    ): KModIntExpr = intModCache.createIfContextActive(lhs, rhs)

    private val intRemCache = mkContextCheckingCache { l: KExpr<KIntSort>,
                                                       r: KExpr<KIntSort> ->
        KRemIntExpr(this, l, r)
    }

    fun mkIntRem(
        lhs: KExpr<KIntSort>,
        rhs: KExpr<KIntSort>
    ): KRemIntExpr = intRemCache.createIfContextActive(lhs, rhs)

    private val intToRealCache = mkContextCheckingCache { arg: KExpr<KIntSort> ->
        KToRealIntExpr(this, arg)
    }

    fun mkIntToReal(arg: KExpr<KIntSort>): KToRealIntExpr = intToRealCache.createIfContextActive(arg)

    private val int32NumCache = mkClosableCache { v: Int ->
        KInt32NumExpr(this, v)
    }

    fun mkIntNum(value: Int): KInt32NumExpr = int32NumCache.createIfContextActive(value)

    private val int64NumCache = mkClosableCache { v: Long ->
        KInt64NumExpr(this, v)
    }

    fun mkIntNum(value: Long): KInt64NumExpr = int64NumCache.createIfContextActive(value)

    private val intBigNumCache = mkClosableCache { v: BigInteger ->
        KIntBigNumExpr(this, v)
    }

    fun mkIntNum(value: BigInteger): KIntBigNumExpr = intBigNumCache.createIfContextActive(value)
    fun mkIntNum(value: String) =
        value.toIntOrNull()
            ?.let { mkIntNum(it) }
            ?: value.toLongOrNull()?.let { mkIntNum(it) }
            ?: mkIntNum(value.toBigInteger())

    infix fun KExpr<KIntSort>.mod(rhs: KExpr<KIntSort>) = mkIntMod(this, rhs)
    infix fun KExpr<KIntSort>.rem(rhs: KExpr<KIntSort>) = mkIntRem(this, rhs)
    fun KExpr<KIntSort>.toRealExpr() = mkIntToReal(this)

    val Int.expr
        get() = mkIntNum(this)
    val Long.expr
        get() = mkIntNum(this)
    val BigInteger.expr
        get() = mkIntNum(this)

    // real
    private val realToIntCache = mkContextCheckingCache { arg: KExpr<KRealSort> ->
        KToIntRealExpr(this, arg)
    }

    fun mkRealToInt(arg: KExpr<KRealSort>): KToIntRealExpr = realToIntCache.createIfContextActive(arg)

    private val realIsIntCache = mkContextCheckingCache { arg: KExpr<KRealSort> ->
        KIsIntRealExpr(this, arg)
    }

    fun mkRealIsInt(arg: KExpr<KRealSort>): KIsIntRealExpr = realIsIntCache.createIfContextActive(arg)

    private val realNumCache = mkContextCheckingCache { numerator: KIntNumExpr,
                                                        denominator: KIntNumExpr ->
        KRealNumExpr(this, numerator, denominator)
    }

    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr): KRealNumExpr =
        realNumCache.createIfContextActive(numerator, denominator)

    @Suppress("MemberVisibilityCanBePrivate")
    fun mkRealNum(numerator: KIntNumExpr) = mkRealNum(numerator, 1.expr)
    fun mkRealNum(numerator: Int) = mkRealNum(mkIntNum(numerator))
    fun mkRealNum(numerator: Int, denominator: Int) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))
    fun mkRealNum(numerator: Long) = mkRealNum(mkIntNum(numerator))
    fun mkRealNum(numerator: Long, denominator: Long) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))
    fun mkRealNum(value: String): KRealNumExpr {
        val parts = value.split('/')

        return when (parts.size) {
            1 -> mkRealNum(mkIntNum(parts[0]))
            2 -> mkRealNum(mkIntNum(parts[0]), mkIntNum(parts[1]))
            else -> error("incorrect real num format")
        }
    }

    fun KExpr<KRealSort>.toIntExpr() = mkRealToInt(this)
    fun KExpr<KRealSort>.isIntExpr() = mkRealIsInt(this)

    // bitvectors
    private val bv1Cache = mkClosableCache { value: Boolean -> KBitVec1Value(this, value) }
    private val bv8Cache = mkClosableCache { value: Byte -> KBitVec8Value(this, value) }
    private val bv16Cache = mkClosableCache { value: Short -> KBitVec16Value(this, value) }
    private val bv32Cache = mkClosableCache { value: Int -> KBitVec32Value(this, value) }
    private val bv64Cache = mkClosableCache { value: Long -> KBitVec64Value(this, value) }
    private val bvCache = mkClosableCache { value: String, sizeBits: UInt ->
        KBitVecCustomValue(this, value, sizeBits)
    }

    fun mkBv(value: Boolean): KBitVec1Value = bv1Cache.createIfContextActive(value)
    fun mkBv(value: Boolean, sizeBits: UInt): KBitVecValue<KBvSort> {
        val intValue = (if (value) 1 else 0) as Number
        return mkBv(intValue, sizeBits)
    }

    fun Boolean.toBv(): KBitVec1Value = mkBv(this)
    fun Boolean.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)


    fun mkBv(value: Byte): KBitVec8Value = bv8Cache.createIfContextActive(value)
    fun mkBv(value: Byte, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun Byte.toBv(): KBitVec8Value = mkBv(this)
    fun Byte.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun UByte.toBv(): KBitVec8Value = mkBv(toByte())

    fun mkBv(value: Short): KBitVec16Value = bv16Cache.createIfContextActive(value)
    fun mkBv(value: Short, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun Short.toBv(): KBitVec16Value = mkBv(this)
    fun Short.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun UShort.toBv(): KBitVec16Value = mkBv(toShort())

    fun mkBv(value: Int): KBitVec32Value = bv32Cache.createIfContextActive(value)
    fun mkBv(value: Int, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun Int.toBv(): KBitVec32Value = mkBv(this)
    fun Int.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun UInt.toBv(): KBitVec32Value = mkBv(toInt())

    fun mkBv(value: Long): KBitVec64Value = bv64Cache.createIfContextActive(value)
    fun mkBv(value: Long, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun Long.toBv(): KBitVec64Value = mkBv(this)
    fun Long.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun ULong.toBv(): KBitVec64Value = mkBv(toLong())

    /**
     * Constructs a bit vector from the given [value] containing of [sizeBits] bits.
     *
     * Note: if [sizeBits] is less than is required to represent the [value],
     * the last [sizeBits] bits of the [value] will be taken.
     *
     * At the same time, if [sizeBits] is greater than it is required,
     * binary representation of the [value] will be padded from the start with its sign bit.
     */
    private fun mkBv(value: Number, sizeBits: UInt): KBitVecValue<KBvSort> {
        val binaryString = value.toBinary().takeLast(sizeBits.toInt())
        val paddedString = binaryString.padStart(sizeBits.toInt(), binaryString.first())

        return mkBv(paddedString, sizeBits)
    }

    private fun Number.toBv(sizeBits: UInt) = mkBv(this, sizeBits)

    fun mkBv(value: String, sizeBits: UInt): KBitVecValue<KBvSort> = when (sizeBits.toInt()) {
        1 -> mkBv(value.toUInt(radix = 2).toInt() != 0).cast()
        Byte.SIZE_BITS -> mkBv(value.toUByte(radix = 2).toByte()).cast()
        Short.SIZE_BITS -> mkBv(value.toUShort(radix = 2).toShort()).cast()
        Int.SIZE_BITS -> mkBv(value.toUInt(radix = 2).toInt()).cast()
        Long.SIZE_BITS -> mkBv(value.toULong(radix = 2).toLong()).cast()
        else -> bvCache.createIfContextActive(value, sizeBits)
    }

    private val bvNotExprCache = mkClosableCache { value: KExpr<KBvSort> -> KBvNotExpr(this, value) }
    fun <T : KBvSort> mkBvNotExpr(value: KExpr<T>): KBvNotExpr<T> =
        bvNotExprCache.createIfContextActive(value.cast()).cast()

    private val bvRedAndExprCache = mkClosableCache { value: KExpr<KBvSort> ->
        KBvReductionAndExpr(this, value)
    }

    fun <T : KBvSort> mkBvReductionAndExpr(value: KExpr<T>): KBvReductionAndExpr<T> =
        bvRedAndExprCache.createIfContextActive(value.cast()).cast()

    fun <T : KBvSort> KExpr<T>.reductionAnd(): KBvReductionAndExpr<T> = mkBvReductionAndExpr(this)

    private val bvRedOrExprCache = mkClosableCache { value: KExpr<KBvSort> ->
        KBvReductionOrExpr(this, value)
    }

    fun <T : KBvSort> mkBvReductionOrExpr(value: KExpr<T>): KBvReductionOrExpr<T> =
        bvRedOrExprCache.createIfContextActive(value.cast()).cast()

    fun <T : KBvSort> KExpr<T>.reductionOr(): KBvReductionOrExpr<T> = mkBvReductionOrExpr(this)

    private val bvAndExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                   arg1: KExpr<KBvSort> ->
        KBvAndExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvAndExpr<T> =
        bvAndExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvOrExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                  arg1: KExpr<KBvSort> ->
        KBvOrExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvOrExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvOrExpr<T> =
        bvOrExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvXorExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                   arg1: KExpr<KBvSort> ->
        KBvXorExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvXorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvXorExpr<T> =
        bvXorExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNAndExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                    arg1: KExpr<KBvSort> ->
        KBvNAndExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvNAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvNAndExpr<T> =
        bvNAndExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNorExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                   arg1: KExpr<KBvSort> ->
        KBvNorExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvNorExpr<T> =
        bvNorExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvXNorExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                    arg1: KExpr<KBvSort> ->
        KBvXNorExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvXNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvXNorExpr<T> =
        bvXNorExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNegationExprCache = mkClosableCache { value: KExpr<KBvSort> ->
        KBvNegationExpr(this, value)
    }

    fun <T : KBvSort> mkBvNegationExpr(value: KExpr<T>): KBvNegationExpr<T> =
        bvNegationExprCache.createIfContextActive(value.cast()).cast()

    private val bvAddExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                   arg1: KExpr<KBvSort> ->
        KBvAddExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvAddExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddExpr<T> =
        bvAddExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSubExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                   arg1: KExpr<KBvSort> ->
        KBvSubExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSubExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubExpr<T> =
        bvSubExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvMulExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                   arg1: KExpr<KBvSort> ->
        KBvMulExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvMulExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvMulExpr<T> =
        bvMulExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedDivExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                           arg1: KExpr<KBvSort> ->
        KBvUnsignedDivExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvUnsignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedDivExpr<T> =
        bvUnsignedDivExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedDivExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                         arg1: KExpr<KBvSort> ->
        KBvSignedDivExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedDivExpr<T> =
        bvSignedDivExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedRemExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                           arg1: KExpr<KBvSort> ->
        KBvUnsignedRemExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvUnsignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedRemExpr<T> =
        bvUnsignedRemExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedRemExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                         arg1: KExpr<KBvSort> ->
        KBvSignedRemExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedRemExpr<T> =
        bvSignedRemExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedModExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                         arg1: KExpr<KBvSort> ->
        KBvSignedModExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSignedModExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedModExpr<T> =
        bvSignedModExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedLessExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                            arg1: KExpr<KBvSort> ->
        KBvUnsignedLessExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvUnsignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedLessExpr<T> =
        bvUnsignedLessExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedLessExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                          arg1: KExpr<KBvSort> ->
        KBvSignedLessExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedLessExpr<T> =
        bvSignedLessExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedLessOrEqualExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                                 arg1: KExpr<KBvSort> ->
        KBvSignedLessOrEqualExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSignedLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedLessOrEqualExpr<T> =
        bvSignedLessOrEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedLessOrEqualExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                                   arg1: KExpr<KBvSort> ->
        KBvUnsignedLessOrEqualExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvUnsignedLessOrEqualExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedLessOrEqualExpr<T> {
        val a0: KExpr<KBvSort> = arg0.cast()
        val a1: KExpr<KBvSort> = arg1.cast()

        return bvUnsignedLessOrEqualExprCache.createIfContextActive(a0, a1).cast()
    }

    private val bvUnsignedGreaterOrEqualExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                                      arg1: KExpr<KBvSort> ->
        KBvUnsignedGreaterOrEqualExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvUnsignedGreaterOrEqualExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedGreaterOrEqualExpr<T> {
        val a0: KExpr<KBvSort> = arg0.cast()
        val a1: KExpr<KBvSort> = arg1.cast()

        return bvUnsignedGreaterOrEqualExprCache.createIfContextActive(a0, a1).cast()
    }

    private val bvSignedGreaterOrEqualExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                                    arg1: KExpr<KBvSort> ->
        KBvSignedGreaterOrEqualExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSignedGreaterOrEqualExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvSignedGreaterOrEqualExpr<T> {
        val a0: KExpr<KBvSort> = arg0.cast()
        val a1: KExpr<KBvSort> = arg1.cast()

        return bvSignedGreaterOrEqualExprCache.createIfContextActive(a0, a1).cast()
    }

    private val bvUnsignedGreaterExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                               arg1: KExpr<KBvSort> ->
        KBvUnsignedGreaterExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvUnsignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedGreaterExpr<T> =
        bvUnsignedGreaterExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedGreaterExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                             arg1: KExpr<KBvSort> ->
        KBvSignedGreaterExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedGreaterExpr<T> =
        bvSignedGreaterExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val concatExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                    arg1: KExpr<KBvSort> ->
        KBvConcatExpr(this, arg0, arg1)
    }

    fun <T : KBvSort, S : KBvSort> mkBvConcatExpr(arg0: KExpr<T>, arg1: KExpr<S>): KBvConcatExpr =
        concatExprCache.createIfContextActive(arg0.cast(), arg1.cast())

    private val extractExprCache = mkClosableCache { high: Int,
                                                     low: Int,
                                                     value: KExpr<KBvSort> ->
        KBvExtractExpr(this, high, low, value)
    }

    fun <T : KBvSort> mkBvExtractExpr(high: Int, low: Int, value: KExpr<T>) =
        extractExprCache.createIfContextActive(high, low, value.cast())

    private val signExtensionExprCache = mkClosableCache { i: Int,
                                                           value: KExpr<KBvSort> ->
        KBvSignExtensionExpr(this, i, value)
    }

    fun <T : KBvSort> mkBvSignExtensionExpr(
        i: Int,
        value: KExpr<T>
    ) = signExtensionExprCache.createIfContextActive(i, value.cast())

    private val zeroExtensionExprCache = mkClosableCache { i: Int,
                                                           value: KExpr<KBvSort> ->
        KBvZeroExtensionExpr(this, i, value)
    }

    fun <T : KBvSort> mkBvZeroExtensionExpr(
        i: Int,
        value: KExpr<T>
    ) = zeroExtensionExprCache.createIfContextActive(i, value.cast())

    private val repeatExprCache = mkClosableCache { i: Int,
                                                    value: KExpr<KBvSort> ->
        KBvRepeatExpr(this, i, value)
    }

    fun <T : KBvSort> mkBvRepeatExpr(
        i: Int,
        value: KExpr<T>
    ) = repeatExprCache.createIfContextActive(i, value.cast())

    private val bvShiftLeftExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                         arg1: KExpr<KBvSort> ->
        KBvShiftLeftExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvShiftLeftExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvShiftLeftExpr<T> =
        bvShiftLeftExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvLogicalShiftRightExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                                 arg1: KExpr<KBvSort> ->
        KBvLogicalShiftRightExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvLogicalShiftRightExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvLogicalShiftRightExpr<T> =
        bvLogicalShiftRightExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvArithShiftRightExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                               arg1: KExpr<KBvSort> ->
        KBvArithShiftRightExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvArithShiftRightExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvArithShiftRightExpr<T> =
        bvArithShiftRightExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvRotateLeftExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                          arg1: KExpr<KBvSort> ->
        KBvRotateLeftExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvRotateLeftExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvRotateLeftExpr<T> =
        bvRotateLeftExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvRotateLeftIndexedExprCache = mkClosableCache { i: Int,
                                                                 value: KExpr<KBvSort> ->
        KBvRotateLeftIndexedExpr(this, i, value)
    }

    fun <T : KBvSort> mkBvRotateLeftIndexedExpr(i: Int, value: KExpr<T>): KBvRotateLeftIndexedExpr<T> =
        bvRotateLeftIndexedExprCache.createIfContextActive(i, value.cast()).cast()

    fun <T : KBvSort> mkBvRotateLeftExpr(arg0: Int, arg1: KExpr<T>): KBvRotateLeftExpr<T> =
        mkBvRotateLeftExpr(mkBv(arg0, arg1.sort.sizeBits), arg1.cast()).cast()

    private val bvRotateRightIndexedExprCache = mkClosableCache { i: Int,
                                                                  value: KExpr<KBvSort> ->
        KBvRotateRightIndexedExpr(this, i, value)
    }

    fun <T : KBvSort> mkBvRotateRightIndexedExpr(i: Int, value: KExpr<T>): KBvRotateRightIndexedExpr<T> =
        bvRotateRightIndexedExprCache.createIfContextActive(i, value.cast()).cast()


    private val bvRotateRightExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                           arg1: KExpr<KBvSort> ->
        KBvRotateRightExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvRotateRightExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvRotateRightExpr<T> =
        bvRotateRightExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    fun <T : KBvSort> mkBvRotateRightExpr(arg0: Int, arg1: KExpr<T>): KBvRotateRightExpr<T> =
        mkBvRotateRightExpr(mkBv(arg0, arg1.sort.sizeBits), arg1.cast()).cast()

    private val bv2IntExprCache = mkClosableCache { value: KExpr<KBvSort>,
                                                    isSigned: Boolean ->
        KBv2IntExpr(this, value, isSigned)
    }

    fun <T : KBvSort> mkBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KBv2IntExpr =
        bv2IntExprCache.createIfContextActive(value.cast(), isSigned)

    private val bvAddNoOverflowExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                             arg1: KExpr<KBvSort>,
                                                             isSigned: Boolean ->
        KBvAddNoOverflowExpr(this, arg0, arg1, isSigned)
    }

    fun <T : KBvSort> mkBvAddNoOverflowExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvAddNoOverflowExpr<T> {
        val a0: KExpr<KBvSort> = arg0.cast()
        val a1: KExpr<KBvSort> = arg1.cast()

        return bvAddNoOverflowExprCache.createIfContextActive(a0, a1, isSigned).cast()
    }

    private val bvAddNoUnderflowExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                              arg1: KExpr<KBvSort> ->
        KBvAddNoUnderflowExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvAddNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddNoUnderflowExpr<T> =
        bvAddNoUnderflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSubNoOverflowExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                             arg1: KExpr<KBvSort> ->
        KBvSubNoOverflowExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSubNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubNoOverflowExpr<T> =
        bvSubNoOverflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSubNoUnderflowExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                              arg1: KExpr<KBvSort>,
                                                              isSigned: Boolean ->
        KBvSubNoUnderflowExpr(this, arg0, arg1, isSigned)
    }

    fun <T : KBvSort> mkBvSubNoUnderflowExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvSubNoUnderflowExpr<T> {
        val a0: KExpr<KBvSort> = arg0.cast()
        val a1: KExpr<KBvSort> = arg1.cast()

        return bvSubNoUnderflowExprCache.createIfContextActive(a0, a1, isSigned).cast()
    }


    private val bvDivNoOverflowExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                             arg1: KExpr<KBvSort> ->
        KBvDivNoOverflowExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvDivNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvDivNoOverflowExpr<T> =
        bvDivNoOverflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNegNoOverflowExprCache =
        mkClosableCache { value: KExpr<KBvSort> -> KBvNegNoOverflowExpr(this, value) }

    fun <T : KBvSort> mkBvNegationNoOverflowExpr(value: KExpr<T>): KBvNegNoOverflowExpr<T> =
        bvNegNoOverflowExprCache.createIfContextActive(value.cast()).cast()

    private val bvMulNoOverflowExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                             arg1: KExpr<KBvSort>,
                                                             isSigned: Boolean ->
        KBvMulNoOverflowExpr(this, arg0, arg1, isSigned)
    }

    fun <T : KBvSort> mkBvMulNoOverflowExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvMulNoOverflowExpr<T> {
        val a0: KExpr<KBvSort> = arg0.cast()
        val a1: KExpr<KBvSort> = arg1.cast()

        return bvMulNoOverflowExprCache.createIfContextActive(a0, a1, isSigned).cast()
    }

    private val bvMulNoUnderflowExprCache = mkClosableCache { arg0: KExpr<KBvSort>,
                                                              arg1: KExpr<KBvSort> ->
        KBvMulNoUnderflowExpr(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvMulNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvMulNoUnderflowExpr<T> =
        bvMulNoUnderflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    // fp values
    private val fp16Cache = mkClosableCache { value: Float -> KFp16Value(this, value) }
    private val fp32Cache = mkClosableCache { value: Float -> KFp32Value(this, value) }
    private val fp64Cache = mkClosableCache { value: Double -> KFp64Value(this, value) }
    private val fp128Cache = mkClosableCache { significand: Long,
                                               exponent: Long,
                                               signBit: Boolean ->
        KFp128Value(this, significand, exponent, signBit)
    }
    private val fpCustomSizeCache = mkClosableCache { significandSize: UInt,
                                                      exponentSize: UInt,
                                                      significand: Long,
                                                      exponent: Long,
                                                      signBit: Boolean ->
        KFpCustomSizeValue(this, significandSize, exponentSize, significand, exponent, signBit)
    }

    /**
     * Creates FP16 from the [value].
     *
     * Important: we suppose that [value] has biased exponent, but FP16 will be created from the unbiased one.
     * So, at first, we'll subtract [KFp16Sort.exponentShiftSize] from the [value]'s exponent,
     * take required for FP16 bits, and this will be **unbiased** FP16 exponent.
     * The same is true for other methods but [mkFpCustomSize].
     * */
    fun mkFp16(value: Float): KFp16Value = fp16Cache.createIfContextActive(value)
    fun mkFp32(value: Float): KFp32Value = fp32Cache.createIfContextActive(value)
    fun mkFp64(value: Double): KFp64Value = fp64Cache.createIfContextActive(value)
    fun mkFp128(significand: Long, exponent: Long, signBit: Boolean): KFp128Value =
        fp128Cache.createIfContextActive(significand, exponent, signBit)

    val Float.expr
        get() = mkFp32(this)

    val Double.expr
        get() = mkFp64(this)

    /**
     * Creates FP with a custom size.
     * Important: [exponent] here is an **unbiased** value.
     */
    fun <T : KFpSort> mkFpCustomSize(
        exponentSize: UInt,
        significandSize: UInt,
        exponent: Long,
        significand: Long,
        signBit: Boolean
    ): KFpValue<T> {
        val intSignBit = if (signBit) 1 else 0

        return when (mkFpSort(exponentSize, significandSize)) {
            is KFp16Sort -> {
                val number = constructFp16Number(exponent, significand, intSignBit)

                mkFp16(number).cast()
            }
            is KFp32Sort -> {
                val number = constructFp32Number(exponent, significand, intSignBit)

                mkFp32(number).cast()
            }
            is KFp64Sort -> {
                val number = constructFp64Number(exponent, significand, intSignBit)

                mkFp64(number).cast()
            }
            is KFp128Sort -> mkFp128(significand, exponent, signBit).cast()
            else -> {
                val fpValue = fpCustomSizeCache.createIfContextActive(
                    significandSize, exponentSize, significand, exponent, signBit
                )

                fpValue.cast()
            }
        }
    }

    fun <T : KFpSort> mkFpCustomSize(
        exponent: KBitVecValue<out KBvSort>,
        significand: KBitVecValue<out KBvSort>,
        signBit: Boolean
    ): KFpValue<T> {
        // TODO we should not use numbers and work with bitvectors. Change it later
        val exponentLongValue = exponent.stringValue.toULong(radix = 2).toLong()
        val significandLongValue = significand.stringValue.toULong(radix = 2).toLong()

        return mkFpCustomSize(
            exponent.sort.sizeBits,
            significand.sort.sizeBits,
            exponentLongValue,
            significandLongValue,
            signBit
        )
    }

    @Suppress("MagicNumber")
    private fun constructFp16Number(exponent: Long, significand: Long, intSignBit: Int): Float {
        // get sign and `body` of the unbiased exponent
        val exponentSign = (exponent.toInt() shr 4) and 1
        val otherExponent = exponent.toInt() and 0b1111

        // get fp16 significand part -- last teb bits (eleventh stored implicitly)
        val significandBits = significand.toInt() and 0b1111_1111_11

        // Transform fp16 exponent into fp32 exponent adding three zeroes between the sign and the body
        // Then add the bias for fp32 and apply the mask to avoid overflow of the eight bits
        val biasedFloatExponent = (((exponentSign shl 7) or otherExponent) + KFp32Sort.exponentShiftSize) and 0xff

        val bits = (intSignBit shl 31) or (biasedFloatExponent shl 23) or (significandBits shl 13)

        return intBitsToFloat(bits)
    }

    @Suppress("MagicNumber")
    private fun constructFp32Number(exponent: Long, significand: Long, intSignBit: Int): Float {
        // `and 0xff` here is to avoid overloading when we have a number greater than 255,
        // and the result of the addition will affect the sign bit
        val biasedExponent = (exponent.toInt() + KFp32Sort.exponentShiftSize) and 0xff
        val intValue = (intSignBit shl 31) or (biasedExponent shl 23) or significand.toInt()

        return intBitsToFloat(intValue)
    }

    @Suppress("MagicNumber")
    private fun constructFp64Number(exponent: Long, significand: Long, intSignBit: Int): Double {
        // `and 0b111_1111_1111` here is to avoid overloading when we have a number greater than 255,
        // and the result of the addition will affect the sign bit
        val biasedExponent = (exponent + KFp64Sort.exponentShiftSize) and 0b111_1111_1111
        val longValue = (intSignBit.toLong() shl 63) or (biasedExponent shl 52) or significand

        return longBitsToDouble(longValue)
    }

    fun <T : KFpSort> mkFp(value: Float, sort: T): KExpr<T> {
        val significand = value.extractSignificand(sort)
        val exponent = value.extractExponent(sort, isBiased = false).extendWithLeadingZeros()
        val sign = value.booleanSignBit

        return mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            exponent,
            significand.extendWithLeadingZeros(),
            sign
        )
    }

    fun <T : KFpSort> mkFp(value: Double, sort: T): KExpr<T> {
        val significand = value.extractSignificand(sort)
        val exponent = value.extractExponent(sort, isBiased = false)
        val sign = value.booleanSignBit

        return mkFpCustomSize(sort.exponentBits, sort.significandBits, exponent, significand, sign)
    }

    fun Double.toFp(sort: KFpSort = mkFp64Sort()): KExpr<KFpSort> = mkFp(this, sort)

    fun Float.toFp(sort: KFpSort = mkFp32Sort()): KExpr<KFpSort> = mkFp(this, sort)

    fun <T : KFpSort> mkFp(significand: Int, exponent: Int, signBit: Boolean, sort: T): KExpr<T> =
        mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            exponent.extendWithLeadingZeros(),
            significand.extendWithLeadingZeros(),
            signBit
        )

    fun <T : KFpSort> mkFp(significand: Long, exponent: Long, signBit: Boolean, sort: T): KExpr<T> =
        mkFpCustomSize(sort.exponentBits, sort.significandBits, exponent, significand, signBit)

    /**
     * Special Fp values
     * */
    fun <T : KFpSort> mkFpZero(signBit: Boolean, sort: T): KExpr<T> =
        mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            exponent = fpZeroExponentUnbiased(sort),
            significand = 0L,
            signBit = signBit
        )

    fun <T : KFpSort> mkFpInf(signBit: Boolean, sort: T): KExpr<T> =
        mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            exponent = fpTopExponentUnbiased(sort),
            significand = 0L,
            signBit
        )

    fun <T : KFpSort> mkFpNan(sort: T): KExpr<T> =
        mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            exponent = fpTopExponentUnbiased(sort),
            significand = 1L,
            signBit = false
        )

    private fun fpTopExponentUnbiased(sort: KFpSort): Long = when (sort) {
        is KFp16Sort -> Float.POSITIVE_INFINITY.getHalfPrecisionExponent(isBiased = false).toLong()
        is KFp32Sort -> Float.POSITIVE_INFINITY.getExponent(isBiased = false).toLong()
        is KFp64Sort -> Double.POSITIVE_INFINITY.getExponent(isBiased = false)
        is KFp128Sort -> TODO("fp 128 top exponent")
        is KFpCustomSizeSort -> TODO("custom fp top exponent")
    }

    private fun fpZeroExponentUnbiased(sort: KFpSort): Long = when (sort) {
        is KFp16Sort -> 0.0f.getHalfPrecisionExponent(isBiased = false).toLong()
        is KFp32Sort -> 0.0f.getExponent(isBiased = false).toLong()
        is KFp64Sort -> 0.0.getExponent(isBiased = false)
        is KFp128Sort -> TODO("fp 128 zero exponent")
        is KFpCustomSizeSort -> TODO("custom fp zero exponent")
    }

    private val roundingModeCache = mkClosableCache { value: KFpRoundingMode ->
        KFpRoundingModeExpr(this, value)
    }

    fun mkFpRoundingModeExpr(
        value: KFpRoundingMode
    ): KFpRoundingModeExpr = roundingModeCache.createIfContextActive(value)

    private val fpAbsExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpAbsExpr(this, value)
    }

    fun <T : KFpSort> mkFpAbsExpr(
        value: KExpr<T>
    ): KFpAbsExpr<T> = fpAbsExprCache.createIfContextActive(value.cast()).cast()

    private val fpNegationExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpNegationExpr(this, value)
    }

    fun <T : KFpSort> mkFpNegationExpr(
        value: KExpr<T>
    ): KFpNegationExpr<T> = fpNegationExprCache.createIfContextActive(value.cast()).cast()

    private val fpAddExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                   arg0: KExpr<KFpSort>,
                                                   arg1: KExpr<KFpSort> ->
        KFpAddExpr(this, roundingMode, arg0, arg1)
    }

    fun <T : KFpSort> mkFpAddExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpAddExpr<T> {
        val rm = roundingMode.cast()
        val a0: KExpr<KFpSort> = arg0.cast()
        val a1: KExpr<KFpSort> = arg1.cast()

        return fpAddExprCache.createIfContextActive(rm, a0, a1).cast()
    }


    private val fpSubExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                   arg0: KExpr<KFpSort>,
                                                   arg1: KExpr<KFpSort> ->
        KFpSubExpr(this, roundingMode, arg0, arg1)
    }

    fun <T : KFpSort> mkFpSubExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpSubExpr<T> {
        val rm = roundingMode.cast()
        val a0: KExpr<KFpSort> = arg0.cast()
        val a1: KExpr<KFpSort> = arg1.cast()

        return fpSubExprCache.createIfContextActive(rm, a0, a1).cast()
    }

    private val fpMulExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                   arg0: KExpr<KFpSort>,
                                                   arg1: KExpr<KFpSort> ->
        KFpMulExpr(this, roundingMode, arg0, arg1)
    }

    fun <T : KFpSort> mkFpMulExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpMulExpr<T> {
        val rm = roundingMode.cast()
        val a0: KExpr<KFpSort> = arg0.cast()
        val a1: KExpr<KFpSort> = arg1.cast()

        return fpMulExprCache.createIfContextActive(rm, a0, a1).cast()
    }

    private val fpDivExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                   arg0: KExpr<KFpSort>,
                                                   arg1: KExpr<KFpSort> ->
        KFpDivExpr(this, roundingMode, arg0, arg1)
    }

    fun <T : KFpSort> mkFpDivExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpDivExpr<T> {
        val rm = roundingMode.cast()
        val a0: KExpr<KFpSort> = arg0.cast()
        val a1: KExpr<KFpSort> = arg1.cast()

        return fpDivExprCache.createIfContextActive(rm, a0, a1).cast()
    }

    private val fpFusedMulAddExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                           arg0: KExpr<KFpSort>,
                                                           arg1: KExpr<KFpSort>,
                                                           arg2: KExpr<KFpSort> ->
        KFpFusedMulAddExpr(this, roundingMode, arg0, arg1, arg2)
    }

    fun <T : KFpSort> mkFpFusedMulAddExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KFpFusedMulAddExpr<T> {
        val rm = roundingMode.cast()
        val a0: KExpr<KFpSort> = arg0.cast()
        val a1: KExpr<KFpSort> = arg1.cast()
        val a2: KExpr<KFpSort> = arg2.cast()

        return fpFusedMulAddExprCache.createIfContextActive(rm, a0, a1, a2).cast()
    }

    private val fpSqrtExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                    value: KExpr<KFpSort> ->
        KFpSqrtExpr(this, roundingMode, value)
    }

    fun <T : KFpSort> mkFpSqrtExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KFpSqrtExpr<T> {
        val rm = roundingMode.cast()
        val arg: KExpr<KFpSort> = value.cast()

        return fpSqrtExprCache.createIfContextActive(rm, arg).cast()
    }

    private val fpRemExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                   arg1: KExpr<KFpSort> ->
        KFpRemExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpRemExpr<T> =
        fpRemExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpRoundToIntegralExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                               value: KExpr<KFpSort> ->
        KFpRoundToIntegralExpr(this, roundingMode, value)
    }

    fun <T : KFpSort> mkFpRoundToIntegralExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KFpRoundToIntegralExpr<T> {
        val rm = roundingMode.cast()
        val arg: KExpr<KFpSort> = value.cast()

        return fpRoundToIntegralExprCache.createIfContextActive(rm, arg).cast()
    }

    private val fpMinExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                   arg1: KExpr<KFpSort> ->
        KFpMinExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpMinExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpMinExpr<T> =
        fpMinExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpMaxExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                   arg1: KExpr<KFpSort> ->
        KFpMaxExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpMaxExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpMaxExpr<T> =
        fpMaxExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpLessOrEqualExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                           arg1: KExpr<KFpSort> ->
        KFpLessOrEqualExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpLessOrEqualExpr<T> =
        fpLessOrEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpLessExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                    arg1: KExpr<KFpSort> ->
        KFpLessExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpLessExpr<T> =
        fpLessExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpGreaterOrEqualExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                              arg1: KExpr<KFpSort> ->
        KFpGreaterOrEqualExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpGreaterOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpGreaterOrEqualExpr<T> =
        fpGreaterOrEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpGreaterExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                       arg1: KExpr<KFpSort> ->
        KFpGreaterExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpGreaterExpr<T> =
        fpGreaterExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpEqualExprCache = mkClosableCache { arg0: KExpr<KFpSort>,
                                                     arg1: KExpr<KFpSort> ->
        KFpEqualExpr(this, arg0, arg1)
    }

    fun <T : KFpSort> mkFpEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KFpEqualExpr<T> =
        fpEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val fpIsNormalExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpIsNormalExpr(this, value)
    }

    fun <T : KFpSort> mkFpIsNormalExpr(value: KExpr<T>): KFpIsNormalExpr<T> =
        fpIsNormalExprCache.createIfContextActive(value.cast()).cast()

    private val fpIsSubnormalExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpIsSubnormalExpr(this, value)
    }

    fun <T : KFpSort> mkFpIsSubnormalExpr(value: KExpr<T>): KFpIsSubnormalExpr<T> =
        fpIsSubnormalExprCache.createIfContextActive(value.cast()).cast()

    private val fpIsZeroExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpIsZeroExpr(this, value)
    }

    fun <T : KFpSort> mkFpIsZeroExpr(value: KExpr<T>): KFpIsZeroExpr<T> =
        fpIsZeroExprCache.createIfContextActive(value.cast()).cast()

    private val fpIsInfiniteExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpIsInfiniteExpr(this, value)
    }

    fun <T : KFpSort> mkFpIsInfiniteExpr(value: KExpr<T>): KFpIsInfiniteExpr<T> =
        fpIsInfiniteExprCache.createIfContextActive(value.cast()).cast()

    private val fpIsNaNExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpIsNaNExpr(this, value)
    }

    fun <T : KFpSort> mkFpIsNaNExpr(value: KExpr<T>): KFpIsNaNExpr<T> =
        fpIsNaNExprCache.createIfContextActive(value.cast()).cast()

    private val fpIsNegativeExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpIsNegativeExpr(this, value)
    }

    fun <T : KFpSort> mkFpIsNegativeExpr(value: KExpr<T>): KFpIsNegativeExpr<T> =
        fpIsNegativeExprCache.createIfContextActive(value.cast()).cast()

    private val fpIsPositiveExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpIsPositiveExpr(this, value)
    }

    fun <T : KFpSort> mkFpIsPositiveExpr(value: KExpr<T>): KFpIsPositiveExpr<T> =
        fpIsPositiveExprCache.createIfContextActive(value.cast()).cast()

    private val fpToBvExprCache = mkClosableCache { roundingMode: KExpr<KFpRoundingModeSort>,
                                                    value: KExpr<KFpSort>,
                                                    bvSize: Int,
                                                    isSigned: Boolean ->
        KFpToBvExpr(this, roundingMode, value, bvSize, isSigned)
    }

    fun <T : KFpSort> mkFpToBvExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>,
        bvSize: Int,
        isSigned: Boolean
    ): KFpToBvExpr<T> {
        val rm = roundingMode.cast()
        val arg: KExpr<KFpSort> = value.cast()

        return fpToBvExprCache.createIfContextActive(rm, arg, bvSize, isSigned).cast()
    }

    private val fpToRealExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpToRealExpr(this, value)
    }

    fun <T : KFpSort> mkFpToRealExpr(value: KExpr<T>): KFpToRealExpr<T> =
        fpToRealExprCache.createIfContextActive(value.cast()).cast()

    private val fpToIEEEBvExprCache = mkClosableCache { value: KExpr<KFpSort> ->
        KFpToIEEEBvExpr(this, value)
    }

    fun <T : KFpSort> mkFpToIEEEBvExpr(value: KExpr<T>): KFpToIEEEBvExpr<T> =
        fpToIEEEBvExprCache.createIfContextActive(value.cast()).cast()

    private val fpFromBvExprCache = mkClosableCache { sign: KExpr<KBv1Sort>,
                                                      exponent: KExpr<out KBvSort>,
                                                      significand: KExpr<out KBvSort> ->
        val exponentBits = exponent.sort.sizeBits
        // +1 it required since bv doesn't contain `hidden bit`
        val significandBits = significand.sort.sizeBits + 1u
        val sort = mkFpSort(exponentBits, significandBits)

        KFpFromBvExpr(this, sort, sign, exponent, significand)
    }

    fun <T : KFpSort> mkFpFromBvExpr(
        sign: KExpr<KBv1Sort>,
        exponent: KExpr<out KBvSort>,
        significand: KExpr<out KBvSort>,
    ): KFpFromBvExpr<T> = fpFromBvExprCache.createIfContextActive(sign, exponent, significand).cast()

    private val fpToFpExprCache = mkClosableCache { sort: KFpSort,
                                                    rm: KExpr<KFpRoundingModeSort>,
                                                    value: KExpr<out KFpSort> ->
        KFpToFpExpr(this, sort, rm, value)
    }
    private val realToFpExprCache = mkClosableCache { sort: KFpSort,
                                                      rm: KExpr<KFpRoundingModeSort>,
                                                      value: KExpr<KRealSort> ->
        KRealToFpExpr(this, sort, rm, value)
    }
    private val bvToFpExprCache = mkClosableCache { sort: KFpSort,
                                                    rm: KExpr<KFpRoundingModeSort>,
                                                    value: KExpr<KBvSort>,
                                                    signed: Boolean ->
        KBvToFpExpr(this, sort, rm, value, signed)
    }

    fun <T : KFpSort> mkFpToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<out KFpSort>
    ): KFpToFpExpr<T> = fpToFpExprCache.createIfContextActive(sort, roundingMode, value).cast()

    fun <T : KFpSort> mkRealToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KRealSort>
    ): KRealToFpExpr<T> = realToFpExprCache.createIfContextActive(sort, roundingMode, value).cast()

    fun <T : KFpSort> mkBvToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KBvSort>,
        signed: Boolean
    ): KBvToFpExpr<T> = bvToFpExprCache.createIfContextActive(sort, roundingMode, value, signed).cast()

    // quantifiers
    private val existentialQuantifierCache = mkClosableCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KExistentialQuantifier(this, body, bounds)
    }

    fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        existentialQuantifierCache.createIfContextActive(body, bounds)

    private val universalQuantifierCache = mkClosableCache { body: KExpr<KBoolSort>,
                                                             bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KUniversalQuantifier(this, body, bounds)
    }

    fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        universalQuantifierCache.createIfContextActive(body, bounds)

    // utils
    private val exprSortCache = mkClosableCache { expr: KExpr<*> -> computeExprSort(expr) }
    private fun computeExprSort(expr: KExpr<*>): KSort {
        val exprsToComputeSorts = arrayListOf<KExpr<*>>()
        val dependency = arrayListOf<KExpr<*>>()

        expr.sortComputationExprDependency(dependency)

        while (dependency.isNotEmpty()) {
            val e = dependency.removeLast()

            if (e in exprSortCache) continue

            val sizeBeforeExpand = dependency.size
            e.sortComputationExprDependency(dependency)

            if (sizeBeforeExpand != dependency.size) {
                exprsToComputeSorts += e
            }
        }

        exprsToComputeSorts.asReversed().forEach {
            it.sort
        }

        return expr.computeExprSort()
    }

    fun <T : KSort> getExprSort(expr: KExpr<T>): T =
        exprSortCache.createIfContextActive(expr).uncheckedCast()

    private val exprDeclCache = mkClosableCache { expr: KApp<*, *> -> with(expr) { decl() } }
    val <T : KSort> KApp<T, *>.decl: KDecl<T>
        get() = exprDeclCache.createIfContextActive(this).uncheckedCast()

    /*
    * declarations
    * */

    // functions
    private val funcDeclCache = mkClosableCache { name: String, sort: KSort, args: List<KSort> ->
        ensureContextMatch(sort)
        ensureContextMatch(args)
        KFuncDecl(this, name, sort, args)
    }

    fun <T : KSort> mkFuncDecl(
        name: String,
        sort: T,
        args: List<KSort>
    ): KFuncDecl<T> = if (args.isEmpty()) {
        mkConstDecl(name, sort)
    } else {
        funcDeclCache.createIfContextActive(name, sort, args).cast()
    }

    private val constDeclCache = mkClosableCache { name: String, sort: KSort ->
        ensureContextMatch(sort)
        KConstDecl(this, name, sort)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> mkConstDecl(name: String, sort: T): KConstDecl<T> =
        constDeclCache.createIfContextActive(name, sort).cast()

    /* Since any two KFuncDecl are only equivalent if they are the same kotlin object,
     * we can guarantee that the returned KFuncDecl is not equal to any other declaration.
    */
    private var freshConstIdx = 0
    fun <T : KSort> mkFreshFuncDecl(name: String, sort: T, args: List<KSort>): KFuncDecl<T> {
        if (args.isEmpty()) return mkFreshConstDecl(name, sort)

        ensureContextMatch(sort)
        ensureContextMatch(args)

        return KFuncDecl(this, "$name!fresh!${freshConstIdx++}", sort, args)
    }

    fun <T : KSort> mkFreshConstDecl(name: String, sort: T): KConstDecl<T> {
        ensureContextMatch(sort)
        return KConstDecl(this, "$name!fresh!${freshConstIdx++}", sort)
    }

    // bool
    private val falseDeclCache = mkClosableCache<KFalseDecl> { KFalseDecl(this) }
    fun mkFalseDecl(): KFalseDecl = falseDeclCache.createIfContextActive()

    private val trueDeclCache = mkClosableCache<KTrueDecl> { KTrueDecl(this) }
    fun mkTrueDecl(): KTrueDecl = trueDeclCache.createIfContextActive()

    private val andDeclCache = mkClosableCache<KAndDecl> { KAndDecl(this) }
    fun mkAndDecl(): KAndDecl = andDeclCache.createIfContextActive()

    private val orDeclCache = mkClosableCache<KOrDecl> { KOrDecl(this) }
    fun mkOrDecl(): KOrDecl = orDeclCache.createIfContextActive()

    private val notDeclCache = mkClosableCache<KNotDecl> { KNotDecl(this) }
    fun mkNotDecl(): KNotDecl = notDeclCache.createIfContextActive()

    private val impliesDeclCache = mkClosableCache<KImpliesDecl> { KImpliesDecl(this) }
    fun mkImpliesDecl(): KImpliesDecl = impliesDeclCache.createIfContextActive()

    private val xorDeclCache = mkClosableCache<KXorDecl> { KXorDecl(this) }
    fun mkXorDecl(): KXorDecl = xorDeclCache.createIfContextActive()

    private val eqDeclCache = mkContextCheckingCache { arg: KSort ->
        KEqDecl(this, arg)
    }

    fun <T : KSort> mkEqDecl(arg: T): KEqDecl<T> = eqDeclCache.createIfContextActive(arg).cast()

    private val distinctDeclCache = mkContextCheckingCache { arg: KSort ->
        KDistinctDecl(this, arg)
    }

    fun <T : KSort> mkDistinctDecl(arg: T): KDistinctDecl<T> = distinctDeclCache.createIfContextActive(arg).cast()

    private val iteDeclCache = mkContextCheckingCache { arg: KSort ->
        KIteDecl(this, arg)
    }

    fun <T : KSort> mkIteDecl(arg: T): KIteDecl<T> = iteDeclCache.createIfContextActive(arg).cast()

    // array
    private val arraySelectDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArraySelectDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArraySelectDecl(array: KArraySort<D, R>): KArraySelectDecl<D, R> =
        arraySelectDeclCache.createIfContextActive(array).cast()

    private val arrayStoreDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArrayStoreDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArrayStoreDecl(array: KArraySort<D, R>): KArrayStoreDecl<D, R> =
        arrayStoreDeclCache.createIfContextActive(array).cast()

    private val arrayConstDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArrayConstDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArrayConstDecl(array: KArraySort<D, R>): KArrayConstDecl<D, R> =
        arrayConstDeclCache.createIfContextActive(array).cast()

    // arith
    private val arithAddDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithAddDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithAddDecl(arg: T): KArithAddDecl<T> =
        arithAddDeclCache.createIfContextActive(arg).cast()

    private val arithSubDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithSubDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithSubDecl(arg: T): KArithSubDecl<T> =
        arithSubDeclCache.createIfContextActive(arg).cast()

    private val arithMulDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithMulDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithMulDecl(arg: T): KArithMulDecl<T> =
        arithMulDeclCache.createIfContextActive(arg).cast()

    private val arithDivDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithDivDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithDivDecl(arg: T): KArithDivDecl<T> =
        arithDivDeclCache.createIfContextActive(arg).cast()

    private val arithPowerDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithPowerDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithPowerDecl(arg: T): KArithPowerDecl<T> =
        arithPowerDeclCache.createIfContextActive(arg).cast()

    private val arithUnaryMinusDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithUnaryMinusDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithUnaryMinusDecl(arg: T): KArithUnaryMinusDecl<T> =
        arithUnaryMinusDeclCache.createIfContextActive(arg).cast()

    private val arithGeDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithGeDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithGeDecl(
        arg: T
    ): KArithGeDecl<T> = arithGeDeclCache.createIfContextActive(arg).cast()

    private val arithGtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithGtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithGtDecl(
        arg: T
    ): KArithGtDecl<T> = arithGtDeclCache.createIfContextActive(arg).cast()

    private val arithLeDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLeDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLeDecl(
        arg: T
    ): KArithLeDecl<T> = arithLeDeclCache.createIfContextActive(arg).cast()

    private val arithLtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLtDecl(
        arg: T
    ): KArithLtDecl<T> = arithLtDeclCache.createIfContextActive(arg).cast()

    // int
    private val intModDeclCache = mkClosableCache<KIntModDecl> { KIntModDecl(this) }
    fun mkIntModDecl(): KIntModDecl = intModDeclCache.createIfContextActive()

    private val intToRealDeclCache = mkClosableCache<KIntToRealDecl> { KIntToRealDecl(this) }
    fun mkIntToRealDecl(): KIntToRealDecl = intToRealDeclCache.createIfContextActive()

    private val intRemDeclCache = mkClosableCache<KIntRemDecl> { KIntRemDecl(this) }
    fun mkIntRemDecl(): KIntRemDecl = intRemDeclCache.createIfContextActive()

    private val intNumDeclCache = mkClosableCache { value: String -> KIntNumDecl(this, value) }
    fun mkIntNumDecl(value: String): KIntNumDecl = intNumDeclCache.createIfContextActive(value)

    // real
    private val realIsIntDeclCache = mkClosableCache<KRealIsIntDecl> { KRealIsIntDecl(this) }
    fun mkRealIsIntDecl(): KRealIsIntDecl = realIsIntDeclCache.createIfContextActive()

    private val realToIntDeclCache = mkClosableCache<KRealToIntDecl> { KRealToIntDecl(this) }
    fun mkRealToIntDecl(): KRealToIntDecl = realToIntDeclCache.createIfContextActive()

    private val realNumDeclCache = mkClosableCache { value: String -> KRealNumDecl(this, value) }
    fun mkRealNumDecl(value: String): KRealNumDecl = realNumDeclCache.createIfContextActive(value)

    private val bv1DeclCache = mkClosableCache { value: Boolean -> KBitVec1ValueDecl(this, value) }
    private val bv8DeclCache = mkClosableCache { value: Byte -> KBitVec8ValueDecl(this, value) }
    private val bv16DeclCache = mkClosableCache { value: Short -> KBitVec16ValueDecl(this, value) }
    private val bv32DeclCache = mkClosableCache { value: Int -> KBitVec32ValueDecl(this, value) }
    private val bv64DeclCache = mkClosableCache { value: Long -> KBitVec64ValueDecl(this, value) }
    private val bvCustomSizeDeclCache = mkClosableCache { value: String, sizeBits: UInt ->
        KBitVecCustomSizeValueDecl(this, value, sizeBits)
    }

    fun mkBvDecl(value: Boolean): KDecl<KBv1Sort> = bv1DeclCache.createIfContextActive(value)
    fun mkBvDecl(value: Byte): KDecl<KBv8Sort> = bv8DeclCache.createIfContextActive(value)
    fun mkBvDecl(value: Short): KDecl<KBv16Sort> = bv16DeclCache.createIfContextActive(value)
    fun mkBvDecl(value: Int): KDecl<KBv32Sort> = bv32DeclCache.createIfContextActive(value)
    fun mkBvDecl(value: Long): KDecl<KBv64Sort> = bv64DeclCache.createIfContextActive(value)

    fun mkBvDecl(value: String, sizeBits: UInt): KDecl<KBvSort> = when (sizeBits.toInt()) {
        1 -> mkBvDecl(value.toUInt(radix = 2).toInt() != 0).cast()
        Byte.SIZE_BITS -> mkBvDecl(value.toUByte(radix = 2).toByte()).cast()
        Short.SIZE_BITS -> mkBvDecl(value.toUShort(radix = 2).toShort()).cast()
        Int.SIZE_BITS -> mkBvDecl(value.toUInt(radix = 2).toInt()).cast()
        Long.SIZE_BITS -> mkBvDecl(value.toULong(radix = 2).toLong()).cast()
        else -> bvCustomSizeDeclCache.createIfContextActive(value, sizeBits).cast()
    }

    private val bvNotDeclCache = mkClosableCache { sort: KBvSort -> KBvNotDecl(this, sort) }
    fun <T : KBvSort> mkBvNotDecl(sort: T): KBvNotDecl<T> = bvNotDeclCache.createIfContextActive(sort).cast()

    private val bvRedAndDeclCache = mkClosableCache { sort: KBvSort -> KBvReductionAndDecl(this, sort) }
    fun <T : KBvSort> mkBvReductionAndDecl(sort: T): KBvReductionAndDecl<T> =
        bvRedAndDeclCache.createIfContextActive(sort).cast()

    private val bvRedOrDeclCache = mkClosableCache { sort: KBvSort -> KBvReductionOrDecl(this, sort) }
    fun <T : KBvSort> mkBvReductionOrDecl(sort: T): KBvReductionOrDecl<T> =
        bvRedOrDeclCache.createIfContextActive(sort).cast()

    private val bvAndDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvAndDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvAndDecl(arg0: T, arg1: T): KBvAndDecl<T> =
        bvAndDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvOrDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvOrDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvOrDecl(arg0: T, arg1: T): KBvOrDecl<T> =
        bvOrDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvXorDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvXorDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvXorDecl(arg0: T, arg1: T): KBvXorDecl<T> =
        bvXorDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvNAndDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvNAndDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvNAndDecl(arg0: T, arg1: T): KBvNAndDecl<T> =
        bvNAndDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvNorDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvNorDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvNorDecl(arg0: T, arg1: T): KBvNorDecl<T> =
        bvNorDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvXNorDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvXNorDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvXNorDecl(arg0: T, arg1: T): KBvXNorDecl<T> =
        bvXNorDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvNegDeclCache = mkClosableCache { sort: KBvSort -> KBvNegationDecl(this, sort) }
    fun <T : KBvSort> mkBvNegationDecl(sort: T): KBvNegationDecl<T> = bvNegDeclCache.createIfContextActive(sort).cast()

    private val bvAddDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvAddDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvAddDecl(arg0: T, arg1: T): KBvAddDecl<T> =
        bvAddDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSubDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSubDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvSubDecl(arg0: T, arg1: T): KBvSubDecl<T> =
        bvSubDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvMulDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvMulDecl(this, arg0, arg1) }
    fun <T : KBvSort> mkBvMulDecl(arg0: T, arg1: T): KBvMulDecl<T> =
        bvMulDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvUnsignedDivDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedDivDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedDivDecl(arg0: T, arg1: T): KBvUnsignedDivDecl<T> =
        bvUnsignedDivDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSignedDivDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedDivDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedDivDecl(arg0: T, arg1: T): KBvSignedDivDecl<T> =
        bvSignedDivDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvUnsignedRemDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedRemDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedRemDecl(arg0: T, arg1: T): KBvUnsignedRemDecl<T> =
        bvUnsignedRemDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSignedRemDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedRemDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedRemDecl(arg0: T, arg1: T): KBvSignedRemDecl<T> =
        bvSignedRemDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSignedModDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedModDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedModDecl(arg0: T, arg1: T): KBvSignedModDecl<T> =
        bvSignedModDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvUnsignedLessDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedLessDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedLessDecl(arg0: T, arg1: T): KBvUnsignedLessDecl<T> =
        bvUnsignedLessDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSignedLessDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedLessDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedLessDecl(arg0: T, arg1: T): KBvSignedLessDecl<T> =
        bvSignedLessDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSignedLessOrEqualDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedLessOrEqualDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedLessOrEqualDecl(arg0: T, arg1: T): KBvSignedLessOrEqualDecl<T> =
        bvSignedLessOrEqualDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvUnsignedLessOrEqualDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedLessOrEqualDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedLessOrEqualDecl(arg0: T, arg1: T): KBvUnsignedLessOrEqualDecl<T> =
        bvUnsignedLessOrEqualDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvUnsignedGreaterOrEqualDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedGreaterOrEqualDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedGreaterOrEqualDecl(arg0: T, arg1: T): KBvUnsignedGreaterOrEqualDecl<T> =
        bvUnsignedGreaterOrEqualDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSignedGreaterOrEqualDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedGreaterOrEqualDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedGreaterOrEqualDecl(arg0: T, arg1: T): KBvSignedGreaterOrEqualDecl<T> =
        bvSignedGreaterOrEqualDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvUnsignedGreaterDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedGreaterDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedGreaterDecl(arg0: T, arg1: T): KBvUnsignedGreaterDecl<T> =
        bvUnsignedGreaterDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSignedGreaterDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedGreaterDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedGreaterDecl(arg0: T, arg1: T): KBvSignedGreaterDecl<T> =
        bvSignedGreaterDeclCache.createIfContextActive(arg0, arg1).cast()

    private val concatDeclCache = mkClosableCache { arg0: KBvSort,
                                                    arg1: KBvSort ->
        KBvConcatDecl(this, arg0, arg1)
    }

    fun mkBvConcatDecl(
        arg0: KBvSort,
        arg1: KBvSort
    ): KBvConcatDecl = concatDeclCache.createIfContextActive(arg0, arg1)

    private val extractDeclCache = mkClosableCache { high: Int, low: Int, value: KExpr<KBvSort> ->
        KBvExtractDecl(this, high, low, value)
    }

    fun mkBvExtractDecl(high: Int, low: Int, value: KExpr<KBvSort>) =
        extractDeclCache.createIfContextActive(high, low, value)

    private val signExtDeclCache = mkClosableCache { i: Int, value: KBvSort -> KSignExtDecl(this, i, value) }
    fun mkBvSignExtensionDecl(i: Int, value: KBvSort) = signExtDeclCache.createIfContextActive(i, value)

    private val zeroExtDeclCache = mkClosableCache { i: Int, value: KBvSort -> KZeroExtDecl(this, i, value) }
    fun mkBvZeroExtensionDecl(i: Int, value: KBvSort) = zeroExtDeclCache.createIfContextActive(i, value)

    private val repeatDeclCache = mkClosableCache { i: Int, value: KBvSort -> KBvRepeatDecl(this, i, value) }
    fun mkBvRepeatDecl(i: Int, value: KBvSort) = repeatDeclCache.createIfContextActive(i, value)

    private val bvShiftLeftDeclCache = mkClosableCache { arg0: KBvSort,
                                                         arg1: KBvSort ->
        KBvShiftLeftDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvShiftLeftDecl(arg0: T, arg1: T): KBvShiftLeftDecl<T> =
        bvShiftLeftDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvLogicalShiftRightDeclCache = mkClosableCache { arg0: KBvSort,
                                                                 arg1: KBvSort ->
        KBvLogicalShiftRightDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvLogicalShiftRightDecl(arg0: T, arg1: T): KBvLogicalShiftRightDecl<T> =
        bvLogicalShiftRightDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvArithShiftRightDeclCache = mkClosableCache { arg0: KBvSort,
                                                               arg1: KBvSort ->
        KBvArithShiftRightDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvArithShiftRightDecl(arg0: T, arg1: T): KBvArithShiftRightDecl<T> =
        bvArithShiftRightDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvRotateLeftDeclCache = mkClosableCache { arg0: KBvSort,
                                                          arg1: KBvSort ->
        KBvRotateLeftDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvRotateLeftDecl(arg0: T, arg1: T): KBvRotateLeftDecl<T> =
        bvRotateLeftDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvRotateLeftIndexedDeclCache = mkClosableCache { i: Int,
                                                                 valueSort: KBvSort ->
        KBvRotateLeftIndexedDecl(this, i, valueSort)
    }

    fun <T : KBvSort> mkBvRotateLeftIndexedDecl(i: Int, valueSort: T): KBvRotateLeftIndexedDecl<T> =
        bvRotateLeftIndexedDeclCache.createIfContextActive(i, valueSort).cast()


    private val bvRotateRightDeclCache = mkClosableCache { arg0: KBvSort,
                                                           arg1: KBvSort ->
        KBvRotateRightDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvRotateRightDecl(arg0: T, arg1: T): KBvRotateRightDecl<T> =
        bvRotateRightDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvRotateRightIndexedDeclCache = mkClosableCache { i: Int,
                                                                  valueSort: KBvSort ->
        KBvRotateRightIndexedDecl(this, i, valueSort)
    }

    fun <T : KBvSort> mkBvRotateRightIndexedDecl(i: Int, valueSort: T): KBvRotateRightIndexedDecl<T> =
        bvRotateRightIndexedDeclCache.createIfContextActive(i, valueSort).cast()


    private val bv2IntDeclCache = mkClosableCache { value: KBvSort,
                                                    isSigned: Boolean ->
        KBv2IntDecl(this, value, isSigned)
    }

    fun mkBv2IntDecl(value: KBvSort, isSigned: Boolean) = bv2IntDeclCache.createIfContextActive(value, isSigned)

    private val bvAddNoOverflowDeclCache = mkClosableCache { arg0: KBvSort,
                                                             arg1: KBvSort,
                                                             isSigned: Boolean ->
        KBvAddNoOverflowDecl(this, arg0, arg1, isSigned)
    }

    fun <T : KBvSort> mkBvAddNoOverflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvAddNoOverflowDecl<T> =
        bvAddNoOverflowDeclCache.createIfContextActive(arg0, arg1, isSigned).cast()

    private val bvAddNoUnderflowDeclCache = mkClosableCache { arg0: KBvSort,
                                                              arg1: KBvSort ->
        KBvAddNoUnderflowDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvAddNoUnderflowDecl(arg0: T, arg1: T): KBvAddNoUnderflowDecl<T> =
        bvAddNoUnderflowDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSubNoOverflowDeclCache = mkClosableCache { arg0: KBvSort,
                                                             arg1: KBvSort ->
        KBvSubNoOverflowDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvSubNoOverflowDecl(arg0: T, arg1: T): KBvSubNoOverflowDecl<T> =
        bvSubNoOverflowDeclCache.createIfContextActive(arg0, arg1).cast()


    private val bvSubUnderflowDeclCache = mkClosableCache { arg0: KBvSort,
                                                            arg1: KBvSort,
                                                            isSigned: Boolean ->
        KBvSubNoUnderflowDecl(this, arg0, arg1, isSigned)
    }

    fun <T : KBvSort> mkBvSubNoUnderflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvSubNoUnderflowDecl<T> =
        bvSubUnderflowDeclCache.createIfContextActive(arg0, arg1, isSigned).cast()

    private val bvDivNoOverflowDeclCache = mkClosableCache { arg0: KBvSort,
                                                             arg1: KBvSort ->
        KBvDivNoOverflowDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvDivNoOverflowDecl(arg0: T, arg1: T): KBvDivNoOverflowDecl<T> =
        bvDivNoOverflowDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvNegationNoOverflowDeclCache = mkClosableCache { value: KBvSort ->
        KBvNegNoOverflowDecl(this, value)
    }

    fun <T : KBvSort> mkBvNegationNoOverflowDecl(value: T): KBvNegNoOverflowDecl<T> =
        bvNegationNoOverflowDeclCache.createIfContextActive(value).cast()

    private val bvMulNoOverflowDeclCache = mkClosableCache { arg0: KBvSort,
                                                             arg1: KBvSort,
                                                             isSigned: Boolean ->
        KBvMulNoOverflowDecl(this, arg0, arg1, isSigned)
    }

    fun <T : KBvSort> mkBvMulNoOverflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvMulNoOverflowDecl<T> =
        bvMulNoOverflowDeclCache.createIfContextActive(arg0, arg1, isSigned).cast()

    private val bvMulNoUnderflowDeclCache = mkClosableCache { arg0: KBvSort,
                                                              arg1: KBvSort ->
        KBvMulNoUnderflowDecl(this, arg0, arg1)
    }

    fun <T : KBvSort> mkBvMulNoUnderflowDecl(arg0: T, arg1: T): KBvMulNoUnderflowDecl<T> =
        bvMulNoUnderflowDeclCache.createIfContextActive(arg0, arg1).cast()

    // FP
    private val fp16DeclCache = mkClosableCache { value: Float -> KFp16Decl(this, value) }
    fun mkFp16Decl(value: Float): KFp16Decl = fp16DeclCache.createIfContextActive(value)

    private val fp32DeclCache = mkClosableCache { value: Float -> KFp32Decl(this, value) }
    fun mkFp32Decl(value: Float): KFp32Decl = fp32DeclCache.createIfContextActive(value)

    private val fp64DeclCache = mkClosableCache { value: Double -> KFp64Decl(this, value) }
    fun mkFp64Decl(value: Double): KFp64Decl = fp64DeclCache.createIfContextActive(value)

    private val fp128DeclCache = mkClosableCache { significand: Long,
                                                   exponent: Long,
                                                   signBit: Boolean ->
        KFp128Decl(this, significand, exponent, signBit)
    }

    fun mkFp128Decl(significandBits: Long, exponent: Long, signBit: Boolean): KFp128Decl =
        fp128DeclCache.createIfContextActive(significandBits, exponent, signBit)

    private val fpCustomSizeDeclCache = mkClosableCache { significandSize: UInt,
                                                          exponentSize: UInt,
                                                          significand: Long,
                                                          exponent: Long,
                                                          signBit: Boolean ->
        KFpCustomSizeDecl(this, significandSize, exponentSize, significand, exponent, signBit)
    }

    fun <T : KFpSort> mkFpCustomSizeDecl(
        significandSize: UInt,
        exponentSize: UInt,
        significand: Long,
        exponent: Long,
        signBit: Boolean
    ): KFpDecl<T, *> {
        val sort = mkFpSort(exponentSize, significandSize)

        if (sort is KFpCustomSizeSort) {
            val fpDecl = fpCustomSizeDeclCache.createIfContextActive(
                significandSize, exponentSize, significand, exponent, signBit
            )

            return fpDecl.cast()
        }

        if (sort is KFp128Sort) {
            return fp128DeclCache.createIfContextActive(significand, exponent, signBit).cast()
        }

        val intSignBit = if (signBit) 1 else 0

        return when (sort) {
            is KFp16Sort -> {
                val fp16Number = constructFp16Number(exponent, significand, intSignBit)

                mkFp16Decl(fp16Number).cast()
            }
            is KFp32Sort -> {
                val fp32Number = constructFp32Number(exponent, significand, intSignBit)

                mkFp32Decl(fp32Number).cast()
            }
            is KFp64Sort -> {
                val fp64Number = constructFp64Number(exponent, significand, intSignBit)

                mkFp64Decl(fp64Number).cast()
            }
            else -> error("Sort declaration for an unknown $sort")
        }
    }

    private val roundingModeDeclCache = mkClosableCache { value: KFpRoundingMode ->
        KFpRoundingModeDecl(this, value)
    }

    fun mkFpRoundingModeDecl(value: KFpRoundingMode) = roundingModeDeclCache.createIfContextActive(value)

    private val fpAbsDeclCache = mkClosableCache { valueSort: KFpSort ->
        KFpAbsDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpAbsDecl(
        valueSort: T
    ): KFpAbsDecl<T> = fpAbsDeclCache.createIfContextActive(valueSort).cast()

    private val fpNegationDeclCache = mkClosableCache { valueSort: KFpSort ->
        KFpNegationDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpNegationDecl(valueSort: T): KFpNegationDecl<T> =
        fpNegationDeclCache.createIfContextActive(valueSort).cast()

    private val fpAddDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                   arg0Sort: KFpSort,
                                                   arg1Sort: KFpSort ->
        KFpAddDecl(this, roundingMode, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpAddDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpAddDecl<T> = fpAddDeclCache.createIfContextActive(roundingMode, arg0Sort, arg1Sort).cast()

    private val fpSubDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                   arg0Sort: KFpSort,
                                                   arg1Sort: KFpSort ->
        KFpSubDecl(this, roundingMode, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpSubDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpSubDecl<T> = fpSubDeclCache.createIfContextActive(roundingMode, arg0Sort, arg1Sort).cast()

    private val fpMulDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                   arg0Sort: KFpSort,
                                                   arg1Sort: KFpSort ->
        KFpMulDecl(this, roundingMode, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpMulDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpMulDecl<T> = fpMulDeclCache.createIfContextActive(roundingMode, arg0Sort, arg1Sort).cast()

    private val fpDivDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                   arg0Sort: KFpSort,
                                                   arg1Sort: KFpSort ->
        KFpDivDecl(this, roundingMode, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpDivDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpDivDecl<T> = fpDivDeclCache.createIfContextActive(roundingMode, arg0Sort, arg1Sort).cast()

    private val fpFusedMulAddDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                           arg0sort: KFpSort,
                                                           arg1Sort: KFpSort,
                                                           arg2Sort: KFpSort ->
        KFpFusedMulAddDecl(this, roundingMode, arg0sort, arg1Sort, arg2Sort)
    }

    fun <T : KFpSort> mkFpFusedMulAddDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T,
        arg2Sort: T
    ): KFpFusedMulAddDecl<T> {
        val fpDecl = fpFusedMulAddDeclCache.createIfContextActive(
            roundingMode,
            arg0Sort,
            arg1Sort,
            arg2Sort
        )
        return fpDecl.cast()
    }

    private val fpSqrtDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                    valueSort: KFpSort ->
        KFpSqrtDecl(this, roundingMode, valueSort)
    }

    fun <T : KFpSort> mkFpSqrtDecl(roundingMode: KFpRoundingModeSort, valueSort: T): KFpSqrtDecl<T> =
        fpSqrtDeclCache.createIfContextActive(roundingMode, valueSort).cast()

    private val fpRemDeclCache = mkClosableCache { arg0Sort: KFpSort,
                                                   arg1Sort: KFpSort ->
        KFpRemDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpRemDecl(arg0Sort: T, arg1Sort: T): KFpRemDecl<T> =
        fpRemDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val roundToIntegralDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                             valueSort: KFpSort ->
        KFpRoundToIntegralDecl(this, roundingMode, valueSort)
    }

    fun <T : KFpSort> mkFpRoundToIntegralDecl(
        roundingMode: KFpRoundingModeSort,
        valueSort: T
    ): KFpRoundToIntegralDecl<T> = roundToIntegralDeclCache.createIfContextActive(roundingMode, valueSort).cast()

    private val fpMinDeclCache = mkClosableCache { arg0Sort: KFpSort, arg1Sort: KFpSort ->
        KFpMinDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpMinDecl(arg0Sort: T, arg1Sort: T): KFpMinDecl<T> =
        fpMinDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val fpMaxDeclCache = mkClosableCache { arg0Sort: KFpSort, arg1Sort: KFpSort ->
        KFpMaxDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpMaxDecl(arg0Sort: T, arg1Sort: T): KFpMaxDecl<T> =
        fpMaxDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val fpLessOrEqualDeclCache = mkClosableCache { arg0Sort: KFpSort, arg1Sort: KFpSort ->
        KFpLessOrEqualDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpLessOrEqualDecl(arg0Sort: T, arg1Sort: T): KFpLessOrEqualDecl<T> =
        fpLessOrEqualDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val fpLessDeclCache = mkClosableCache { arg0Sort: KFpSort, arg1Sort: KFpSort ->
        KFpLessDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpLessDecl(arg0Sort: T, arg1Sort: T): KFpLessDecl<T> =
        fpLessDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val fpGreaterOrEqualDeclCache = mkClosableCache { arg0Sort: KFpSort, arg1Sort: KFpSort ->
        KFpGreaterOrEqualDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpGreaterOrEqualDecl(arg0Sort: T, arg1Sort: T): KFpGreaterOrEqualDecl<T> =
        fpGreaterOrEqualDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val fpGreaterDeclCache = mkClosableCache { arg0Sort: KFpSort, arg1Sort: KFpSort ->
        KFpGreaterDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpGreaterDecl(arg0Sort: T, arg1Sort: T): KFpGreaterDecl<T> =
        fpGreaterDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val fpEqualDeclCache = mkClosableCache { arg0Sort: KFpSort, arg1Sort: KFpSort ->
        KFpEqualDecl(this, arg0Sort, arg1Sort)
    }

    fun <T : KFpSort> mkFpEqualDecl(arg0Sort: T, arg1Sort: T): KFpEqualDecl<T> =
        fpEqualDeclCache.createIfContextActive(arg0Sort, arg1Sort).cast()

    private val fpIsNormalDeclCache = mkClosableCache { valueSort: KFpSort ->
        KFpIsNormalDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpIsNormalDecl(valueSort: T): KFpIsNormalDecl<T> =
        fpIsNormalDeclCache.createIfContextActive(valueSort).cast()

    private val fpIsSubnormalDeclCache = mkClosableCache { valueSort: KFpSort ->
        KFpIsSubnormalDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpIsSubnormalDecl(valueSort: T): KFpIsSubnormalDecl<T> =
        fpIsSubnormalDeclCache.createIfContextActive(valueSort).cast()

    private val fpIsZeroDeclCache = mkClosableCache { valueSort: KFpSort ->
        KFpIsZeroDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpIsZeroDecl(valueSort: T): KFpIsZeroDecl<T> =
        fpIsZeroDeclCache.createIfContextActive(valueSort).cast()

    private val fpIsInfiniteDeclCache = mkClosableCache { valueSort: KFpSort ->
        KFpIsInfiniteDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpIsInfiniteDecl(valueSort: T): KFpIsInfiniteDecl<T> =
        fpIsInfiniteDeclCache.createIfContextActive(valueSort).cast()

    private val fpIsNaNDecl = mkClosableCache { valueSort: KFpSort ->
        KFpIsNaNDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpIsNaNDecl(valueSort: T): KFpIsNaNDecl<T> =
        fpIsNaNDecl.createIfContextActive(valueSort).cast()

    private val fpIsNegativeDecl = mkClosableCache { valueSort: KFpSort ->
        KFpIsNegativeDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpIsNegativeDecl(valueSort: T): KFpIsNegativeDecl<T> =
        fpIsNegativeDecl.createIfContextActive(valueSort).cast()

    private val fpIsPositiveDecl = mkClosableCache { valueSort: KFpSort ->
        KFpIsPositiveDecl(this, valueSort)
    }

    fun <T : KFpSort> mkFpIsPositiveDecl(valueSort: T): KFpIsPositiveDecl<T> =
        fpIsPositiveDecl.createIfContextActive(valueSort).cast()

    private val fpToBvDeclCache = mkClosableCache { roundingMode: KFpRoundingModeSort,
                                                    valueSort: KFpSort,
                                                    bvSize: Int,
                                                    isSigned: Boolean ->
        KFpToBvDecl(this, roundingMode, valueSort, bvSize, isSigned)
    }

    fun <T : KFpSort> mkFpToBvDecl(
        roundingMode: KFpRoundingModeSort,
        valueSort: T,
        bvSize: Int,
        isSigned: Boolean
    ): KFpToBvDecl<T> = fpToBvDeclCache.createIfContextActive(roundingMode, valueSort, bvSize, isSigned).cast()

    private val fpToRealDeclCache = mkClosableCache { valueSort: KFpSort -> KFpToRealDecl(this, valueSort) }

    fun <T : KFpSort> mkFpToRealDecl(valueSort: T): KFpToRealDecl<T> =
        fpToRealDeclCache.createIfContextActive(valueSort).cast()

    private val fpToIEEEBvDeclCache = mkClosableCache { valueSort: KFpSort -> KFpToIEEEBvDecl(this, valueSort) }

    fun <T : KFpSort> mkFpToIEEEBvDecl(valueSort: T): KFpToIEEEBvDecl<T> =
        fpToIEEEBvDeclCache.createIfContextActive(valueSort).cast()

    private val fpFromBvDeclCache = mkClosableCache { signSort: KBv1Sort,
                                                      expSort: KBvSort,
                                                      significandSort: KBvSort ->
        val exponentBits = expSort.sizeBits
        val significandBits = significandSort.sizeBits + 1u
        val sort = mkFpSort(exponentBits, significandBits)

        KFpFromBvDecl(this, sort, signSort, expSort, significandSort)
    }

    fun <T : KFpSort> mkFpFromBvDecl(
        signSort: KBv1Sort,
        expSort: KBvSort,
        significandSort: KBvSort
    ): KFpFromBvDecl<T> {
        val fpDecl = fpFromBvDeclCache.createIfContextActive(signSort, expSort, significandSort)
        return fpDecl.cast()
    }

    private val fpToFpDeclCache = mkClosableCache { sort: KFpSort,
                                                    rm: KFpRoundingModeSort,
                                                    value: KFpSort ->
        KFpToFpDecl(this, sort, rm, value)
    }

    private val realToFpDeclCache = mkClosableCache { sort: KFpSort,
                                                      rm: KFpRoundingModeSort,
                                                      value: KRealSort ->
        KRealToFpDecl(this, sort, rm, value)
    }

    private val bvToFpDeclCache = mkClosableCache { sort: KFpSort,
                                                    rm: KFpRoundingModeSort,
                                                    value: KBvSort,
                                                    signed: Boolean ->
        KBvToFpDecl(this, sort, rm, value, signed)
    }

    fun <T : KFpSort> mkFpToFpDecl(sort: T, rm: KFpRoundingModeSort, value: KFpSort): KFpToFpDecl<T> =
        fpToFpDeclCache.createIfContextActive(sort, rm, value).cast()

    fun <T : KFpSort> mkRealToFpDecl(sort: T, rm: KFpRoundingModeSort, value: KRealSort): KRealToFpDecl<T> =
        realToFpDeclCache.createIfContextActive(sort, rm, value).cast()

    fun <T : KFpSort> mkBvToFpDecl(sort: T, rm: KFpRoundingModeSort, value: KBvSort, signed: Boolean): KBvToFpDecl<T> =
        bvToFpDeclCache.createIfContextActive(sort, rm, value, signed).cast()

    /*
    * KAst
    * */

    /**
     * String representations are not cached since
     * it requires a lot of memory.
     * For example, (and a b) will store a full copy
     * of a and b string representations
     * */
    val KAst.stringRepr: String
        get() = buildString { print(this) }

    // context utils
    private fun ensureContextMatch(vararg args: KAst) {
        for (arg in args) {
            require(this === arg.ctx) { "Context mismatch" }
        }
    }

    @Suppress("SpreadOperator")
    fun ensureContextMatch(args: List<KAst>) {
        ensureContextMatch(*args.toTypedArray())
    }

    inline fun <T> ensureContextActive(block: () -> T): T {
        check(isActive) { "Context is not active" }
        return block()
    }

    private fun <T, A0 : KAst> mkContextCheckingCache(builder: (A0) -> T) = mkClosableCache { a0: A0 ->
        ensureContextMatch(a0)
        builder(a0)
    }

    private fun <T, A0 : List<KAst>> mkContextListCheckingCache(builder: (A0) -> T) = mkClosableCache { a0: A0 ->
        ensureContextMatch(a0)
        builder(a0)
    }

    private fun <T, A0 : KAst, A1 : KAst> mkContextCheckingCache(builder: (A0, A1) -> T) =
        mkClosableCache { a0: A0, a1: A1 ->
            ensureContextMatch(a0, a1)
            builder(a0, a1)
        }

    private fun <T, A0 : KAst, A1 : KAst, A2 : KAst> mkContextCheckingCache(builder: (A0, A1, A2) -> T) =
        mkClosableCache { a0: A0, a1: A1, a2: A2 ->
            ensureContextMatch(a0, a1, a2)
            builder(a0, a1, a2)
        }

    private inline fun <reified T : AutoCloseable> ensureClosed(block: () -> T): T =
        block().also { closableResources += it }

    private fun <T> mkClosableCache(builder: () -> T) = ensureClosed { mkCache(builder) }
    private fun <T, A0> mkClosableCache(builder: (A0) -> T) = ensureClosed { mkCache(builder) }
    private fun <T, A0, A1> mkClosableCache(builder: (A0, A1) -> T) = ensureClosed { mkCache(builder) }
    private fun <T, A0, A1, A2> mkClosableCache(builder: (A0, A1, A2) -> T) = ensureClosed { mkCache(builder) }
    private fun <T, A0, A1, A2, A3> mkClosableCache(
        builder: (A0, A1, A2, A3) -> T
    ) = ensureClosed { mkCache(builder) }

    private fun <T, A0, A1, A2, A3, A4> mkClosableCache(
        builder: (A0, A1, A2, A3, A4) -> T
    ) = ensureClosed { mkCache(builder) }

    private fun <T> Cache0<T>.createIfContextActive(): T = ensureContextActive { create() }

    private fun <T, A> Cache1<T, A>.createIfContextActive(arg: A): T = ensureContextActive { create(arg) }

    private fun <T, A0, A1> Cache2<T, A0, A1>.createIfContextActive(
        a0: A0, a1: A1
    ): T = ensureContextActive { create(a0, a1) }

    private fun <T, A0, A1, A2> Cache3<T, A0, A1, A2>.createIfContextActive(
        a0: A0, a1: A1, a2: A2
    ): T = ensureContextActive { create(a0, a1, a2) }

    private fun <T, A0, A1, A2, A3> Cache4<T, A0, A1, A2, A3>.createIfContextActive(
        a0: A0, a1: A1, a2: A2, a3: A3
    ): T = ensureContextActive { create(a0, a1, a2, a3) }

    private fun <T, A0, A1, A2, A3, A4> Cache5<T, A0, A1, A2, A3, A4>.createIfContextActive(
        a0: A0, a1: A1, a2: A2, a3: A3, a4: A4
    ): T = ensureContextActive { create(a0, a1, a2, a3, a4) }
}
