package org.ksmt

import org.ksmt.cache.Cache0
import org.ksmt.cache.Cache1
import org.ksmt.cache.Cache2
import org.ksmt.cache.Cache3
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
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import java.math.BigInteger
import kotlin.reflect.KProperty
import org.ksmt.decl.KBvRotateLeftIndexedDecl
import org.ksmt.decl.KBvRotateRightIndexedDecl
import org.ksmt.decl.KBvSubNoUnderflowDecl
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvSubNoUnderflowExpr
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.utils.cast
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

    fun mkImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KImpliesExpr = impliesCache.createIfContextActive(p, q)

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

    fun <T : KSort> mkIte(condition: KExpr<KBoolSort>, trueBranch: KExpr<T>, falseBranch: KExpr<T>): KIteExpr<T> =
        iteCache.createIfContextActive(condition, trueBranch.cast(), falseBranch.cast()).cast()


    infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
    operator fun KExpr<KBoolSort>.not() = mkNot(this)
    infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
    infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
    infix fun KExpr<KBoolSort>.xor(other: KExpr<KBoolSort>) = mkXor(this, other)
    infix fun KExpr<KBoolSort>.implies(other: KExpr<KBoolSort>) = mkImplies(this, other)

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

    fun <T : KSort> T.mkConst(name: String): KApp<T, *> = with(mkConstDecl(name)) { apply() }

    fun <T : KSort> T.mkFreshConst(name: String): KApp<T, *> = with(mkFreshConstDecl(name)) { apply() }

    inline operator fun <reified T : KSort> T.getValue(thisRef: Any?, property: KProperty<*>): KApp<T, *> =
        mkConst(property.name)

    // array
    private val arrayStoreCache =
        mkContextCheckingCache { a: KExpr<KArraySort<KSort, KSort>>, i: KExpr<KSort>, v: KExpr<KSort> ->
            KArrayStore(this, a, i, v)
        }

    fun <D : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArraySort<D, R>>, index: KExpr<D>, value: KExpr<R>
    ): KArrayStore<D, R> = arrayStoreCache.createIfContextActive(array.cast(), index.cast(), value.cast()).cast()

    private val arraySelectCache =
        mkContextCheckingCache { array: KExpr<KArraySort<KSort, KSort>>, index: KExpr<KSort> ->
            KArraySelect(this, array, index)
        }

    fun <D : KSort, R : KSort> mkArraySelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>): KArraySelect<D, R> =
        arraySelectCache.createIfContextActive(array.cast(), index.cast()).cast()

    private val arrayConstCache = mkContextCheckingCache { array: KArraySort<KSort, KSort>, value: KExpr<KSort> ->
        KArrayConst(this, array, value)
    }

    fun <D : KSort, R : KSort> mkArrayConst(arraySort: KArraySort<D, R>, value: KExpr<R>): KArrayConst<D, R> =
        arrayConstCache.createIfContextActive(arraySort.cast(), value.cast()).cast()

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

    private val arithDivCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KDivArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KDivArithExpr<T> =
        arithDivCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithPowerCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KPowerArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KPowerArithExpr<T> =
        arithPowerCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithLtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KLtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KLtArithExpr<T> =
        arithLtCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithLeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KLeArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KLeArithExpr<T> =
        arithLeCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithGtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KGtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KGtArithExpr<T> =
        arithGtCache.createIfContextActive(lhs.cast(), rhs.cast()).cast()

    private val arithGeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
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
    private val intModCache = mkContextCheckingCache { l: KExpr<KIntSort>, r: KExpr<KIntSort> ->
        KModIntExpr(this, l, r)
    }

    fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KModIntExpr = intModCache.createIfContextActive(lhs, rhs)

    private val intRemCache = mkContextCheckingCache { l: KExpr<KIntSort>, r: KExpr<KIntSort> ->
        KRemIntExpr(this, l, r)
    }

    fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KRemIntExpr = intRemCache.createIfContextActive(lhs, rhs)

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
        value.toIntOrNull()?.let { mkIntNum(it) } ?: value.toLongOrNull()?.let { mkIntNum(it) }
        ?: mkIntNum(value.toBigInteger())

    infix fun KExpr<KIntSort>.mod(rhs: KExpr<KIntSort>) = mkIntMod(this, rhs)
    infix fun KExpr<KIntSort>.rem(rhs: KExpr<KIntSort>) = mkIntRem(this, rhs)
    fun KExpr<KIntSort>.toRealExpr() = mkIntToReal(this)
    val Int.intExpr
        get() = mkIntNum(this)
    val Long.intExpr
        get() = mkIntNum(this)
    val BigInteger.intExpr
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

    private val realNumCache = mkContextCheckingCache { numerator: KIntNumExpr, denominator: KIntNumExpr ->
        KRealNumExpr(this, numerator, denominator)
    }

    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr): KRealNumExpr =
        realNumCache.createIfContextActive(numerator, denominator)

    @Suppress("MemberVisibilityCanBePrivate")
    fun mkRealNum(numerator: KIntNumExpr) = mkRealNum(numerator, 1.intExpr)
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
    private val bvCache = mkClosableCache { value: String, sizeBits: UInt -> KBitVecCustomValue(this, value, sizeBits) }

    fun mkBv(value: Boolean): KBitVec1Value = bv1Cache.createIfContextActive(value)
    fun Boolean.toBv(): KBitVec1Value = mkBv(this)
    fun mkBv(value: Byte): KBitVec8Value = bv8Cache.createIfContextActive(value)
    fun Byte.toBv(): KBitVec8Value = mkBv(this)
    fun UByte.toBv(): KBitVec8Value = mkBv(toByte())
    fun mkBv(value: Short): KBitVec16Value = bv16Cache.createIfContextActive(value)
    fun Short.toBv(): KBitVec16Value = mkBv(this)
    fun UShort.toBv(): KBitVec16Value = mkBv(toShort())
    fun mkBv(value: Int): KBitVec32Value = bv32Cache.createIfContextActive(value)
    fun Int.toBv(): KBitVec32Value = mkBv(this)
    fun UInt.toBv(): KBitVec32Value = mkBv(toInt())
    fun mkBv(value: Long): KBitVec64Value = bv64Cache.createIfContextActive(value)
    fun Long.toBv(): KBitVec64Value = mkBv(this)
    fun ULong.toBv(): KBitVec64Value = mkBv(toLong())
    fun mkBv(value: Number, sizeBits: UInt): KBitVecValue<KBvSort> {
        val binaryString = value.toBinary()

        require(binaryString.length <= sizeBits.toInt()) {
            "Cannot create a bitvector of size $sizeBits from the given number $value" +
                    " since its binary representation requires at least ${binaryString.length} bits"
        }

        val paddedString = binaryString.padStart(sizeBits.toInt(), binaryString.first())

        return mkBv(paddedString, sizeBits)
    }

    fun Number.toBv(sizeBits: UInt) = mkBv(this, sizeBits)
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

    private val bvRedAndExprCache = mkClosableCache { value: KExpr<KBvSort> -> KBvReductionAndExpr(this, value) }

    fun <T : KBvSort> mkBvReductionAndExpr(value: KExpr<T>): KBvReductionAndExpr<T> =
        bvRedAndExprCache.createIfContextActive(value.cast()).cast()

    fun <T : KBvSort> KExpr<T>.reductionAnd(): KBvReductionAndExpr<T> = mkBvReductionAndExpr(this)

    private val bvRedOrExprCache = mkClosableCache { value: KExpr<KBvSort> -> KBvReductionOrExpr(this, value) }
    fun <T : KBvSort> mkBvReductionOrExpr(value: KExpr<T>): KBvReductionOrExpr<T> =
        bvRedOrExprCache.createIfContextActive(value.cast()).cast()

    fun <T : KBvSort> KExpr<T>.reductionOr(): KBvReductionOrExpr<T> = mkBvReductionOrExpr(this)

    private val bvAndExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvAndExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvAndExpr<T> =
        bvAndExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvOrExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvOrExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvOrExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvOrExpr<T> =
        bvOrExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvXorExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvXorExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvXorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvXorExpr<T> =
        bvXorExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNAndExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvNAndExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvNAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvNAndExpr<T> =
        bvNAndExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNorExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvNorExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvNorExpr<T> =
        bvNorExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvXNorExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvXNorExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvXNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvXNorExpr<T> =
        bvXNorExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNegationExprCache = mkClosableCache { value: KExpr<KBvSort> -> KBvNegationExpr(this, value) }
    fun <T : KBvSort> mkBvNegationExpr(value: KExpr<T>): KBvNegationExpr<T> =
        bvNegationExprCache.createIfContextActive(value.cast()).cast()

    private val bvAddExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvAddExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvAddExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddExpr<T> =
        bvAddExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSubExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSubExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSubExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubExpr<T> =
        bvSubExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvMulExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvMulExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvMulExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvMulExpr<T> =
        bvMulExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedDivExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedDivExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedDivExpr<T> =
        bvUnsignedDivExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedDivExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedDivExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedDivExpr<T> =
        bvSignedDivExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedRemExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedRemExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedRemExpr<T> =
        bvUnsignedRemExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedRemExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedRemExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedRemExpr<T> =
        bvSignedRemExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedModExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedModExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedModExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedModExpr<T> =
        bvSignedModExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedLessExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedLessExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedLessExpr<T> =
        bvUnsignedLessExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedLessExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedLessExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedLessExpr<T> =
        bvSignedLessExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedLessOrEqualExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedLessOrEqualExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedLessOrEqualExpr<T> =
        bvSignedLessOrEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedLessOrEqualExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedLessOrEqualExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedLessOrEqualExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedLessOrEqualExpr<T> =
        bvUnsignedLessOrEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedGreaterOrEqualExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvUnsignedGreaterOrEqualExpr(this, arg0, arg1)
        }

    fun <T : KBvSort> mkBvUnsignedGreaterOrEqualExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedGreaterOrEqualExpr<T> =
        bvUnsignedGreaterOrEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedGreaterOrEqualExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedGreaterOrEqualExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedGreaterOrEqualExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvSignedGreaterOrEqualExpr<T> =
        bvSignedGreaterOrEqualExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvUnsignedGreaterExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedGreaterExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvUnsignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedGreaterExpr<T> =
        bvUnsignedGreaterExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSignedGreaterExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedGreaterExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvSignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedGreaterExpr<T> =
        bvSignedGreaterExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val concatExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvConcatExpr(this, arg0, arg1) }

    fun <T : KBvSort, S : KBvSort> mkBvConcatExpr(arg0: KExpr<T>, arg1: KExpr<S>): KBvConcatExpr =
        concatExprCache.createIfContextActive(arg0.cast(), arg1.cast())

    private val extractExprCache = mkClosableCache { high: Int, low: Int, value: KExpr<KBvSort> ->
        KBvExtractExpr(this, high, low, value)
    }

    fun <T : KBvSort> mkBvExtractExpr(high: Int, low: Int, value: KExpr<T>) =
        extractExprCache.createIfContextActive(high, low, value.cast())

    private val signExtensionExprCache =
        mkClosableCache { i: Int, value: KExpr<KBvSort> -> KBvSignExtensionExpr(this, i, value) }

    fun <T : KBvSort> mkBvSignExtensionExpr(i: Int, value: KExpr<T>) =
        signExtensionExprCache.createIfContextActive(i, value.cast())

    private val zeroExtensionExprCache =
        mkClosableCache { i: Int, value: KExpr<KBvSort> -> KBvZeroExtensionExpr(this, i, value) }

    fun <T : KBvSort> mkBvZeroExtensionExpr(i: Int, value: KExpr<T>) =
        zeroExtensionExprCache.createIfContextActive(i, value.cast())

    private val repeatExprCache = mkClosableCache { i: Int, value: KExpr<KBvSort> -> KBvRepeatExpr(this, i, value) }
    fun <T : KBvSort> mkBvRepeatExpr(i: Int, value: KExpr<T>) = repeatExprCache.createIfContextActive(i, value.cast())

    private val bvShiftLeftExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvShiftLeftExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvShiftLeftExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvShiftLeftExpr<T> =
        bvShiftLeftExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvLogicalShiftRightExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvLogicalShiftRightExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvLogicalShiftRightExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvLogicalShiftRightExpr<T> =
        bvLogicalShiftRightExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvArithShiftRightExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvArithShiftRightExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvArithShiftRightExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvArithShiftRightExpr<T> =
        bvArithShiftRightExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvRotateLeftExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvRotateLeftExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvRotateLeftExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvRotateLeftExpr<T> =
        bvRotateLeftExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvRotateLeftIndexedExprCache =
        mkClosableCache { i: Int, value: KExpr<KBvSort> -> KBvRotateLeftIndexedExpr(this, i, value) }

    fun <T : KBvSort> mkBvRotateLeftIndexedExpr(i: Int, value: KExpr<T>): KBvRotateLeftIndexedExpr<T> =
        bvRotateLeftIndexedExprCache.createIfContextActive(i, value.cast()).cast()

    fun <T : KBvSort> mkBvRotateLeftExpr(arg0: Int, arg1: KExpr<T>): KBvRotateLeftExpr<T> =
        mkBvRotateLeftExpr(mkBv(arg0, arg1.sort().sizeBits), arg1.cast()).cast()

    private val bvRotateRightIndexedExprCache =
        mkClosableCache { i: Int, value: KExpr<KBvSort> -> KBvRotateRightIndexedExpr(this, i, value) }

    fun <T : KBvSort> mkBvRotateRightIndexedExpr(i: Int, value: KExpr<T>): KBvRotateRightIndexedExpr<T> =
        bvRotateRightIndexedExprCache.createIfContextActive(i, value.cast()).cast()


    private val bvRotateRightExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvRotateRightExpr(this, arg0, arg1) }

    fun <T : KBvSort> mkBvRotateRightExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvRotateRightExpr<T> =
        bvRotateRightExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    fun <T : KBvSort> mkBvRotateRightExpr(arg0: Int, arg1: KExpr<T>): KBvRotateRightExpr<T> =
        mkBvRotateRightExpr(mkBv(arg0, arg1.sort().sizeBits), arg1.cast()).cast()

    private val bv2IntExprCache =
        mkClosableCache { value: KExpr<KBvSort>, isSigned: Boolean -> KBv2IntExpr(this, value, isSigned) }

    fun <T : KBvSort> mkBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KBv2IntExpr =
        bv2IntExprCache.createIfContextActive(value.cast(), isSigned)

    private val bvAddNoOverflowExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean ->
            KBvAddNoOverflowExpr(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun <T : KBvSort> mkBvAddNoOverflowExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvAddNoOverflowExpr<T> =
        bvAddNoOverflowExprCache.createIfContextActive(arg0.cast(), arg1.cast(), isSigned).cast()

    private val bvAddNoUnderflowExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvAddNoUnderflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun <T : KBvSort> mkBvAddNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddNoUnderflowExpr<T> =
        bvAddNoUnderflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSubNoOverflowExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvSubNoOverflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun <T : KBvSort> mkBvSubNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubNoOverflowExpr<T> =
        bvSubNoOverflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvSubNoUnderflowExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean ->
            KBvSubNoUnderflowExpr(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun <T : KBvSort> mkBvSubNoUnderflowExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvSubNoUnderflowExpr<T> =
        bvSubNoUnderflowExprCache.createIfContextActive(arg0.cast(), arg1.cast(), isSigned).cast()


    private val bvDivNoOverflowExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvDivNoOverflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun <T : KBvSort> mkBvDivNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvDivNoOverflowExpr<T> =
        bvDivNoOverflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    private val bvNegNoOverflowExprCache =
        mkClosableCache { value: KExpr<KBvSort> -> KBvNegNoOverflowExpr(this, value) }
    fun <T : KBvSort> mkBvNegationNoOverflowExpr(value: KExpr<T>): KBvNegNoOverflowExpr<T> =
        bvNegNoOverflowExprCache.createIfContextActive(value.cast()).cast()

    private val bvMulNoOverflowExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean ->
            KBvMulNoOverflowExpr(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun <T : KBvSort> mkBvMulNoOverflowExpr(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvMulNoOverflowExpr<T> =
        bvMulNoOverflowExprCache.createIfContextActive(arg0.cast(), arg1.cast(), isSigned).cast()

    private val bvMulNoUnderflowExprCache =
        mkClosableCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvMulNoUnderflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun <T : KBvSort> mkBvMulNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KBvMulNoUnderflowExpr<T> =
        bvMulNoUnderflowExprCache.createIfContextActive(arg0.cast(), arg1.cast()).cast()

    // quantifiers
    private val existentialQuantifierCache = mkClosableCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KExistentialQuantifier(this, body, bounds)
    }

    fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        existentialQuantifierCache.createIfContextActive(body, bounds)

    private val universalQuantifierCache = mkClosableCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KUniversalQuantifier(this, body, bounds)
    }

    fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        universalQuantifierCache.createIfContextActive(body, bounds)

    // utils
    private val exprSortCache = mkClosableCache { expr: KExpr<*> -> with(expr) { sort() } }
    val <T : KSort> KExpr<T>.sort: T
        get() = exprSortCache.createIfContextActive(this).uncheckedCast()

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

    fun <T : KSort> mkFuncDecl(name: String, sort: T, args: List<KSort>): KFuncDecl<T> =
        if (args.isEmpty()) mkConstDecl(name, sort) else funcDeclCache.createIfContextActive(name, sort, args).cast()

    private val constDeclCache = mkClosableCache { name: String, sort: KSort ->
        ensureContextMatch(sort)
        KConstDecl(this, name, sort)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> mkConstDecl(name: String, sort: T): KConstDecl<T> =
        constDeclCache.createIfContextActive(name, sort).cast()

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> T.mkConstDecl(name: String) = mkConstDecl(name, this)

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

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> T.mkFreshConstDecl(name: String) = mkFreshConstDecl(name, this)

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

    fun <T : KArithSort<T>> mkArithGeDecl(arg: T): KArithGeDecl<T> = arithGeDeclCache.createIfContextActive(arg).cast()

    private val arithGtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithGtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithGtDecl(arg: T): KArithGtDecl<T> = arithGtDeclCache.createIfContextActive(arg).cast()

    private val arithLeDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLeDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLeDecl(arg: T): KArithLeDecl<T> = arithLeDeclCache.createIfContextActive(arg).cast()

    private val arithLtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLtDecl(arg: T): KArithLtDecl<T> = arithLtDeclCache.createIfContextActive(arg).cast()


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

    private val concatDeclCache = mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvConcatDecl(this, arg0, arg1) }
    fun mkBvConcatDecl(arg0: KBvSort, arg1: KBvSort): KBvConcatDecl = concatDeclCache.createIfContextActive(arg0, arg1)

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

    private val bvShiftLeftDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvShiftLeftDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvShiftLeftDecl(arg0: T, arg1: T): KBvShiftLeftDecl<T> =
        bvShiftLeftDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvLogicalShiftRightDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvLogicalShiftRightDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvLogicalShiftRightDecl(arg0: T, arg1: T): KBvLogicalShiftRightDecl<T> =
        bvLogicalShiftRightDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvArithShiftRightDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvArithShiftRightDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvArithShiftRightDecl(arg0: T, arg1: T): KBvArithShiftRightDecl<T> =
        bvArithShiftRightDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvRotateLeftDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvRotateLeftDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvRotateLeftDecl(arg0: T, arg1: T): KBvRotateLeftDecl<T> =
        bvRotateLeftDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvRotateLeftIndexedDeclCache =
        mkClosableCache { i: Int, valueSort: KBvSort -> KBvRotateLeftIndexedDecl(this, i, valueSort) }

    fun <T : KBvSort> mkBvRotateLeftIndexedDecl(i: Int, valueSort: T): KBvRotateLeftIndexedDecl<T> =
        bvRotateLeftIndexedDeclCache.createIfContextActive(i, valueSort).cast()


    private val bvRotateRightDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort -> KBvRotateRightDecl(this, arg0, arg1) }

    fun <T : KBvSort> mkBvRotateRightDecl(arg0: T, arg1: T): KBvRotateRightDecl<T> =
        bvRotateRightDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvRotateRightIndexedDeclCache =
        mkClosableCache { i: Int, valueSort: KBvSort -> KBvRotateRightIndexedDecl(this, i, valueSort) }

    fun <T : KBvSort> mkBvRotateRightIndexedDecl(i: Int, valueSort: T): KBvRotateRightIndexedDecl<T> =
        bvRotateRightIndexedDeclCache.createIfContextActive(i, valueSort).cast()


    private val bv2IntDeclCache =
        mkClosableCache { value: KBvSort, isSigned: Boolean -> KBv2IntDecl(this, value, isSigned) }

    fun mkBv2IntDecl(value: KBvSort, isSigned: Boolean) = bv2IntDeclCache.createIfContextActive(value, isSigned)

    private val bvAddNoOverflowDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort, isSigned: Boolean ->
            KBvAddNoOverflowDecl(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun <T : KBvSort> mkBvAddNoOverflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvAddNoOverflowDecl<T> =
        bvAddNoOverflowDeclCache.createIfContextActive(arg0, arg1, isSigned).cast()

    private val bvAddNoUnderflowDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort ->
            KBvAddNoUnderflowDecl(
                this,
                arg0,
                arg1,
            )
        }

    fun <T : KBvSort> mkBvAddNoUnderflowDecl(arg0: T, arg1: T): KBvAddNoUnderflowDecl<T> =
        bvAddNoUnderflowDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvSubNoOverflowDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort ->
            KBvSubNoOverflowDecl(
                this,
                arg0,
                arg1,
            )
        }

    fun <T : KBvSort> mkBvSubNoOverflowDecl(arg0: T, arg1: T): KBvSubNoOverflowDecl<T> =
        bvSubNoOverflowDeclCache.createIfContextActive(arg0, arg1).cast()


    private val bvSubUnderflowDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort, isSigned: Boolean ->
            KBvSubNoUnderflowDecl(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun <T : KBvSort> mkBvSubNoUnderflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvSubNoUnderflowDecl<T> =
        bvSubUnderflowDeclCache.createIfContextActive(arg0, arg1, isSigned).cast()

    private val bvDivNoOverflowDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort ->
            KBvDivNoOverflowDecl(
                this,
                arg0,
                arg1
            )
        }

    fun <T : KBvSort> mkBvDivNoOverflowDecl(arg0: T, arg1: T): KBvDivNoOverflowDecl<T> =
        bvDivNoOverflowDeclCache.createIfContextActive(arg0, arg1).cast()

    private val bvNegationNoOverflowDeclCache = mkClosableCache { value: KBvSort -> KBvNegNoOverflowDecl(this, value) }

    fun <T : KBvSort> mkBvNegationNoOverflowDecl(value: T): KBvNegNoOverflowDecl<T> =
        bvNegationNoOverflowDeclCache.createIfContextActive(value).cast()

    private val bvMulNoOverflowDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort, isSigned: Boolean ->
            KBvMulNoOverflowDecl(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun <T : KBvSort> mkBvMulNoOverflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvMulNoOverflowDecl<T> =
        bvMulNoOverflowDeclCache.createIfContextActive(arg0, arg1, isSigned).cast()

    private val bvMulNoUnderflowDeclCache =
        mkClosableCache { arg0: KBvSort, arg1: KBvSort ->
            KBvMulNoUnderflowDecl(
                this,
                arg0,
                arg1,
            )
        }

    fun <T : KBvSort> mkBvMulNoUnderflowDecl(arg0: T, arg1: T): KBvMulNoUnderflowDecl<T> =
        bvMulNoUnderflowDeclCache.createIfContextActive(arg0, arg1).cast()

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

    private fun <T> Cache0<T>.createIfContextActive(): T =
        ensureContextActive { create() }

    private fun <T, A> Cache1<T, A>.createIfContextActive(arg: A): T =
        ensureContextActive { create(arg) }

    private fun <T, A0, A1> Cache2<T, A0, A1>.createIfContextActive(a0: A0, a1: A1): T =
        ensureContextActive { create(a0, a1) }

    private fun <T, A0, A1, A2> Cache3<T, A0, A1, A2>.createIfContextActive(a0: A0, a1: A1, a2: A2): T =
        ensureContextActive { create(a0, a1, a2) }

}
