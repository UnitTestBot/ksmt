package org.ksmt

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
import org.ksmt.decl.KConcatDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDistinctDecl
import org.ksmt.decl.KEqDecl
import org.ksmt.decl.KExtractDecl
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
import org.ksmt.decl.KRepeatDecl
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
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KBitVec16Expr
import org.ksmt.expr.KBitVec1Expr
import org.ksmt.expr.KBitVec32Expr
import org.ksmt.expr.KBitVec64Expr
import org.ksmt.expr.KBitVec8Expr
import org.ksmt.expr.KBitVecCustomExpr
import org.ksmt.expr.KBitVecExpr
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
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
import org.ksmt.expr.KConcatExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KExtractExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
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
import org.ksmt.expr.KRepeatExpr
import org.ksmt.expr.KSignExtensionExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KZeroExtExpr
import org.ksmt.expr.KXorExpr
import org.ksmt.expr.KZeroExtensionExpr
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
import java.math.BigInteger
import kotlin.reflect.KProperty
import org.ksmt.decl.KBvSubNoUnderflowDecl
import org.ksmt.decl.toBinary
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBvSubNoUnderflowExpr

@Suppress("TooManyFunctions", "unused")
open class KContext {
    /*
    * sorts
    * */
    private val boolSortCache = mkCache<KBoolSort> { KBoolSort(this) }
    fun mkBoolSort(): KBoolSort = boolSortCache.create()

    private val arraySortCache = mkContextCheckingCache { domain: KSort, range: KSort ->
        KArraySort(this, domain, range)
    }

    fun <D : KSort, R : KSort> mkArraySort(domain: D, range: R): KArraySort<D, R> =
        arraySortCache.create(domain, range).cast()

    private val intSortCache = mkCache<KIntSort> { KIntSort(this) }
    fun mkIntSort(): KIntSort = intSortCache.create()

    // bit-vec
    private val bvSortCache = mkCache { sizeBits: UInt ->
        when (sizeBits.toInt()) {
            1 -> KBv1Sort(this)
            Byte.SIZE_BITS -> KBv8Sort(this)
            Short.SIZE_BITS -> KBv16Sort(this)
            Int.SIZE_BITS -> KBv32Sort(this)
            Long.SIZE_BITS -> KBv64Sort(this)
            else -> KBvCustomSizeSort(this, sizeBits)
        }
    }

    fun mkBvSort(sizeBits: UInt): KBvSort = bvSortCache.create(sizeBits)
    fun mkBv1Sort(): KBv1Sort = mkBvSort(sizeBits = 1u).cast()
    fun mkBv8Sort(): KBv8Sort = mkBvSort(Byte.SIZE_BITS.toUInt()).cast()
    fun mkBv16Sort(): KBv16Sort = mkBvSort(Short.SIZE_BITS.toUInt()).cast()
    fun mkBv32Sort(): KBv32Sort = mkBvSort(Int.SIZE_BITS.toUInt()).cast()
    fun mkBv64Sort(): KBv64Sort = mkBvSort(Long.SIZE_BITS.toUInt()).cast()

    private val realSortCache = mkCache<KRealSort> { KRealSort(this) }
    fun mkRealSort(): KRealSort = realSortCache.create()

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

    fun mkAnd(args: List<KExpr<KBoolSort>>): KAndExpr = andCache.create(args)

    @Suppress("MemberVisibilityCanBePrivate")
    fun mkAnd(vararg args: KExpr<KBoolSort>) = mkAnd(args.toList())

    private val orCache = mkContextListCheckingCache { args: List<KExpr<KBoolSort>> ->
        KOrExpr(this, args)
    }

    fun mkOr(args: List<KExpr<KBoolSort>>): KOrExpr = orCache.create(args)

    @Suppress("MemberVisibilityCanBePrivate")
    fun mkOr(vararg args: KExpr<KBoolSort>) = mkOr(args.toList())

    private val notCache = mkContextCheckingCache { arg: KExpr<KBoolSort> ->
        KNotExpr(this, arg)
    }

    fun mkNot(arg: KExpr<KBoolSort>): KNotExpr = notCache.create(arg)

    private val impliesCache = mkContextCheckingCache { p: KExpr<KBoolSort>, q: KExpr<KBoolSort> ->
        KImpliesExpr(this, p, q)
    }

    fun mkImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KImpliesExpr = impliesCache.create(p, q)

    private val xorCache = mkContextCheckingCache { a: KExpr<KBoolSort>, b: KExpr<KBoolSort> ->
        KXorExpr(this, a, b)
    }

    fun mkXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KXorExpr = xorCache.create(a, b)

    private val trueCache = mkCache<KTrue> { KTrue(this) }
    fun mkTrue() = trueCache.create()

    private val falseCache = mkCache<KFalse> { KFalse(this) }
    fun mkFalse() = falseCache.create()

    private val eqCache = mkContextCheckingCache { l: KExpr<KSort>, r: KExpr<KSort> ->
        KEqExpr(this, l, r)
    }

    fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>): KEqExpr<T> = eqCache.create(lhs.cast(), rhs.cast()).cast()

    private val distinctCache = mkContextListCheckingCache { args: List<KExpr<KSort>> ->
        KDistinctExpr(this, args)
    }

    fun <T : KSort> mkDistinct(args: List<KExpr<T>>): KDistinctExpr<T> = distinctCache.create(args.cast()).cast()

    private val iteCache = mkContextCheckingCache { c: KExpr<KBoolSort>, t: KExpr<KSort>, f: KExpr<KSort> ->
        KIteExpr(this, c, t, f)
    }

    fun <T : KSort> mkIte(condition: KExpr<KBoolSort>, trueBranch: KExpr<T>, falseBranch: KExpr<T>): KIteExpr<T> =
        iteCache.create(condition, trueBranch.cast(), falseBranch.cast()).cast()


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

    private val functionAppCache = mkCache { decl: KDecl<*>, args: List<KExpr<*>> ->
        ensureContextMatch(decl)
        ensureContextMatch(args)
        KFunctionApp(this, decl, args)
    }

    internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>): KApp<T, *> =
        if (args.isEmpty()) mkConstApp(decl) else functionAppCache.create(decl, args).cast()

    private val constAppCache = mkCache { decl: KDecl<*> ->
        ensureContextMatch(decl)
        KConst(this, decl)
    }

    fun <T : KSort> mkConstApp(decl: KDecl<T>): KConst<T> = constAppCache.create(decl).cast()

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
    ): KArrayStore<D, R> = arrayStoreCache.create(array.cast(), index.cast(), value.cast()).cast()

    private val arraySelectCache =
        mkContextCheckingCache { array: KExpr<KArraySort<KSort, KSort>>, index: KExpr<KSort> ->
            KArraySelect(this, array, index)
        }

    fun <D : KSort, R : KSort> mkArraySelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>): KArraySelect<D, R> =
        arraySelectCache.create(array.cast(), index.cast()).cast()

    private val arrayConstCache = mkContextCheckingCache { array: KArraySort<KSort, KSort>, value: KExpr<KSort> ->
        KArrayConst(this, array, value)
    }

    fun <D : KSort, R : KSort> mkArrayConst(arraySort: KArraySort<D, R>, value: KExpr<R>): KArrayConst<D, R> =
        arrayConstCache.create(arraySort.cast(), value.cast()).cast()

    private val functionAsArrayCache = mkContextCheckingCache { function: KFuncDecl<KSort> ->
        KFunctionAsArray<KSort, KSort>(this, function)
    }

    fun <D : KSort, R : KSort> mkFunctionAsArray(function: KFuncDecl<R>): KFunctionAsArray<D, R> =
        functionAsArrayCache.create(function.cast()).cast()

    private val arrayLambdaCache = mkContextCheckingCache { indexVar: KDecl<*>, body: KExpr<*> ->
        KArrayLambda(this, indexVar, body)
    }

    fun <D : KSort, R : KSort> mkArrayLambda(indexVar: KDecl<D>, body: KExpr<R>): KArrayLambda<D, R> =
        arrayLambdaCache.create(indexVar, body).cast()

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
        mkArrayStore(this, index, value)

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) = mkArraySelect(this, index)

    // arith
    private val arithAddCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KAddArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithAdd(args: List<KExpr<T>>): KAddArithExpr<T> =
        arithAddCache.create(args.cast()).cast()

    private val arithMulCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KMulArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithMul(args: List<KExpr<T>>): KMulArithExpr<T> =
        arithMulCache.create(args.cast()).cast()

    private val arithSubCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KSubArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithSub(args: List<KExpr<T>>): KSubArithExpr<T> =
        arithSubCache.create(args.cast()).cast()

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
        arithUnaryMinusCache.create(arg.cast()).cast()

    private val arithDivCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KDivArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KDivArithExpr<T> =
        arithDivCache.create(lhs.cast(), rhs.cast()).cast()

    private val arithPowerCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KPowerArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KPowerArithExpr<T> =
        arithPowerCache.create(lhs.cast(), rhs.cast()).cast()

    private val arithLtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KLtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KLtArithExpr<T> =
        arithLtCache.create(lhs.cast(), rhs.cast()).cast()

    private val arithLeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KLeArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KLeArithExpr<T> =
        arithLeCache.create(lhs.cast(), rhs.cast()).cast()

    private val arithGtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KGtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KGtArithExpr<T> =
        arithGtCache.create(lhs.cast(), rhs.cast()).cast()

    private val arithGeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KGeArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KGeArithExpr<T> =
        arithGeCache.create(lhs.cast(), rhs.cast()).cast()

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

    fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KModIntExpr = intModCache.create(lhs, rhs)

    private val intRemCache = mkContextCheckingCache { l: KExpr<KIntSort>, r: KExpr<KIntSort> ->
        KRemIntExpr(this, l, r)
    }

    fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KRemIntExpr = intRemCache.create(lhs, rhs)

    private val intToRealCache = mkContextCheckingCache { arg: KExpr<KIntSort> ->
        KToRealIntExpr(this, arg)
    }

    fun mkIntToReal(arg: KExpr<KIntSort>): KToRealIntExpr = intToRealCache.create(arg)

    private val int32NumCache = mkCache { v: Int ->
        KInt32NumExpr(this, v)
    }

    fun mkIntNum(value: Int): KInt32NumExpr = int32NumCache.create(value)

    private val int64NumCache = mkCache { v: Long ->
        KInt64NumExpr(this, v)
    }

    fun mkIntNum(value: Long): KInt64NumExpr = int64NumCache.create(value)

    private val intBigNumCache = mkCache { v: BigInteger ->
        KIntBigNumExpr(this, v)
    }

    fun mkIntNum(value: BigInteger): KIntBigNumExpr = intBigNumCache.create(value)
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

    fun mkRealToInt(arg: KExpr<KRealSort>): KToIntRealExpr = realToIntCache.create(arg)

    private val realIsIntCache = mkContextCheckingCache { arg: KExpr<KRealSort> ->
        KIsIntRealExpr(this, arg)
    }

    fun mkRealIsInt(arg: KExpr<KRealSort>): KIsIntRealExpr = realIsIntCache.create(arg)

    private val realNumCache = mkContextCheckingCache { numerator: KIntNumExpr, denominator: KIntNumExpr ->
        KRealNumExpr(this, numerator, denominator)
    }

    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr): KRealNumExpr =
        realNumCache.create(numerator, denominator)

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
    private val bv1Cache = mkCache { value: Boolean -> KBitVec1Value(this, value) }
    private val bv8Cache = mkCache { value: Byte -> KBitVec8Value(this, value) }
    private val bv16Cache = mkCache { value: Short -> KBitVec16Value(this, value) }
    private val bv32Cache = mkCache { value: Int -> KBitVec32Value(this, value) }
    private val bv64Cache = mkCache { value: Long -> KBitVec64Value(this, value) }
    private val bvCache = mkCache { value: String, sizeBits: UInt -> KBitVecCustomValue(this, value, sizeBits) }

    fun mkBv(value: Boolean): KBitVec1Value = bv1Cache.create(value)
    fun mkBv(value: Byte): KBitVec8Value = bv8Cache.create(value)
    fun mkBv(value: Short): KBitVec16Value = bv16Cache.create(value)
    fun mkBv(value: Int): KBitVec32Value = bv32Cache.create(value)
    fun mkBv(value: Long): KBitVec64Value = bv64Cache.create(value)
    fun mkBv(value: Number, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value.toBinary(), sizeBits)
    fun mkBv(value: String, sizeBits: UInt): KBitVecValue<KBvSort> = when (sizeBits.toInt()) {
        1 -> mkBv(value.toInt(radix = 2) != 0).cast()
        Byte.SIZE_BITS -> mkBv(value.toByte(radix = 2)).cast()
        Short.SIZE_BITS -> mkBv(value.toShort(radix = 2)).cast()
        Int.SIZE_BITS -> mkBv(value.toInt(radix = 2)).cast()
        Long.SIZE_BITS -> mkBv(value.toLong(radix = 2)).cast()
        else -> bvCache.create(value, sizeBits)
    }

    private val bvNotExprCache = mkCache { value: KExpr<KBvSort> -> KBvNotExpr(this, value) }
    fun mkBvNotExpr(value: KExpr<KBvSort>): KBvNotExpr = bvNotExprCache.create(value)

    private val bvRedAndExprCache = mkCache { value: KExpr<KBvSort> -> KBvReductionAndExpr(this, value) }

    fun mkBvReductionAndExpr(value: KExpr<KBvSort>): KBvReductionAndExpr = bvRedAndExprCache.create(value)

    private val bvRedOrExprCache = mkCache { value: KExpr<KBvSort> -> KBvReductionOrExpr(this, value) }
    fun mkBvReductionOrExpr(value: KExpr<KBvSort>): KBvReductionOrExpr = bvRedOrExprCache.create(value)

    private val bvAndExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvAndExpr(this, arg0, arg1) }

    fun mkBvAndExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvAndExpr = bvAndExprCache.create(arg0, arg1)

    private val bvOrExprCache = mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvOrExpr(this, arg0, arg1) }
    fun mkBvOrExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvOrExpr = bvOrExprCache.create(arg0, arg1)

    private val bvXorExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvXorExpr(this, arg0, arg1) }

    fun mkBvXorExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvXorExpr = bvXorExprCache.create(arg0, arg1)

    private val bvNAndExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvNAndExpr(this, arg0, arg1) }

    fun mkBvNAndExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvNAndExpr = bvNAndExprCache.create(arg0, arg1)

    private val bvNorExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvNorExpr(this, arg0, arg1) }

    fun mkBvNorExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvNorExpr = bvNorExprCache.create(arg0, arg1)

    private val bvXNorExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvXNorExpr(this, arg0, arg1) }

    fun mkBvXNorExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvXNorExpr = bvXNorExprCache.create(arg0, arg1)

    private val bvNegationExprCache = mkCache { value: KExpr<KBvSort> -> KBvNegationExpr(this, value) }
    fun mkBvNegationExpr(value: KExpr<KBvSort>): KBvNegationExpr = bvNegationExprCache.create(value)

    private val bvAddExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvAddExpr(this, arg0, arg1) }

    fun mkBvAddExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvAddExpr = bvAddExprCache.create(arg0, arg1)

    private val bvSubExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSubExpr(this, arg0, arg1) }

    fun mkBvSubExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSubExpr = bvSubExprCache.create(arg0, arg1)

    private val bvMulExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvMulExpr(this, arg0, arg1) }

    fun mkBvMulExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvMulExpr = bvMulExprCache.create(arg0, arg1)

    private val bvUnsignedDivExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedDivExpr(this, arg0, arg1) }

    fun mkBvUnsignedDivExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvUnsignedDivExpr =
        bvUnsignedDivExprCache.create(arg0, arg1)

    private val bvSignedDivExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedDivExpr(this, arg0, arg1) }

    fun mkBvSignedDivExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSignedDivExpr =
        bvSignedDivExprCache.create(arg0, arg1)

    private val bvUnsignedRemExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedRemExpr(this, arg0, arg1) }

    fun mkBvUnsignedRemExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvUnsignedRemExpr =
        bvUnsignedRemExprCache.create(arg0, arg1)

    private val bvSignedRemExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedRemExpr(this, arg0, arg1) }

    fun mkBvSignedRemExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSignedRemExpr =
        bvSignedRemExprCache.create(arg0, arg1)

    private val bvSignedModExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedModExpr(this, arg0, arg1) }

    fun mkBvSignedModExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSignedModExpr =
        bvSignedModExprCache.create(arg0, arg1)

    private val bvUnsignedLessExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedLessExpr(this, arg0, arg1) }

    fun mkBvUnsignedLessExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvUnsignedLessExpr =
        bvUnsignedLessExprCache.create(arg0, arg1)

    private val bvSignedLessExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedLessExpr(this, arg0, arg1) }

    fun mkBvSignedLessExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSignedLessExpr =
        bvSignedLessExprCache.create(arg0, arg1)

    private val bvSignedLessOrEqualExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedLessOrEqualExpr(this, arg0, arg1) }

    fun mkBvSignedLessOrEqualExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSignedLessOrEqualExpr =
        bvSignedLessOrEqualExprCache.create(arg0, arg1)

    private val bvUnsignedLessOrEqualExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedLessOrEqualExpr(this, arg0, arg1) }

    fun mkBvUnsignedLessOrEqualExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvUnsignedLessOrEqualExpr =
        bvUnsignedLessOrEqualExprCache.create(arg0, arg1)

    private val bvUnsignedGreaterOrEqualExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedGreaterOrEqualExpr(this, arg0, arg1) }

    fun mkBvUnsignedGreaterOrEqualExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvUnsignedGreaterOrEqualExpr =
        bvUnsignedGreaterOrEqualExprCache.create(arg0, arg1)

    private val bvSignedGreaterOrEqualExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedGreaterOrEqualExpr(this, arg0, arg1) }

    fun mkBvSignedGreaterOrEqualExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSignedGreaterOrEqualExpr =
        bvSignedGreaterOrEqualExprCache.create(arg0, arg1)

    private val bvUnsignedGreaterExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvUnsignedGreaterExpr(this, arg0, arg1) }

    fun mkBvUnsignedGreaterExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvUnsignedGreaterExpr =
        bvUnsignedGreaterExprCache.create(arg0, arg1)

    private val bvSignedGreaterExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvSignedGreaterExpr(this, arg0, arg1) }

    fun mkBvSignedGreaterExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSignedGreaterExpr =
        bvSignedGreaterExprCache.create(arg0, arg1)

    private val concatExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KConcatExpr(this, arg0, arg1) }

    fun mkConcatExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KConcatExpr = concatExprCache.create(arg0, arg1)

    private val extractExprCache = mkCache { high: Int, low: Int, value: KExpr<KBvSort> ->
        KExtractExpr(this, high, low, value)
    }

    fun mkExtractExpr(high: Int, low: Int, value: KExpr<KBvSort>) = extractExprCache.create(high, low, value)

    private val signExtExprCache = mkCache { i: Int, value: KExpr<KBvSort> -> KSignExtensionExpr(this, i, value) }
    fun mkSignExtExpr(i: Int, value: KExpr<KBvSort>) = signExtExprCache.create(i, value)

    private val zeroExtExprCache = mkCache { i: Int, value: KExpr<KBvSort> -> KZeroExtensionExpr(this, i, value) }
    fun mkZeroExtExpr(i: Int, value: KExpr<KBvSort>) = zeroExtExprCache.create(i, value)

    private val repeatExprCache = mkCache { i: Int, value: KExpr<KBvSort> -> KRepeatExpr(this, i, value) }
    fun mkRepeatExpr(i: Int, value: KExpr<KBvSort>) = repeatExprCache.create(i, value)

    private val bvShiftLeftExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvShiftLeftExpr(this, arg0, arg1) }

    fun mkBvShiftLeftExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvShiftLeftExpr = bvShiftLeftExprCache.create(arg0, arg1)

    private val bvLogicalShiftRightExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvLogicalShiftRightExpr(this, arg0, arg1) }

    fun mkBvLogicalShiftRightExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvLogicalShiftRightExpr =
        bvLogicalShiftRightExprCache.create(arg0, arg1)

    private val bvArithShiftRightExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvArithShiftRightExpr(this, arg0, arg1) }

    fun mkBvArithShiftRightExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvArithShiftRightExpr =
        bvArithShiftRightExprCache.create(arg0, arg1)

    private val bvRotateLeftExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvRotateLeftExpr(this, arg0, arg1) }

    fun mkBvRotateLeftExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvRotateLeftExpr =
        bvRotateLeftExprCache.create(arg0, arg1)

    fun mkBvRotateLeftExpr(arg0: Int, arg1: KExpr<KBvSort>): KBvRotateLeftExpr =
        mkBvRotateLeftExpr(mkBv(arg0, arg1.sort().sizeBits), arg1)

    private val bvRotateRightExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> -> KBvRotateRightExpr(this, arg0, arg1) }

    fun mkBvRotateRightExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvRotateRightExpr =
        bvRotateRightExprCache.create(arg0, arg1)

    fun mkBvRotateRightExpr(arg0: Int, arg1: KExpr<KBvSort>): KBvRotateRightExpr =
        mkBvRotateRightExpr(mkBv(arg0, arg1.sort().sizeBits), arg1)

    private val bv2IntExprCache =
        mkCache { value: KExpr<KBvSort>, isSigned: Boolean -> KBv2IntExpr(this, value, isSigned) }

    fun mkBv2IntExpr(value: KExpr<KBvSort>, isSigned: Boolean): KBv2IntExpr = bv2IntExprCache.create(value, isSigned)

    private val bvAddNoOverflowExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean ->
            KBvAddNoOverflowExpr(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun mkBvAddNoOverflowExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean): KBvAddNoOverflowExpr =
        bvAddNoOverflowExprCache.create(arg0, arg1, isSigned)

    private val bvAddNoUnderflowExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvAddNoUnderflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun mkBvAddNoUnderflowExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvAddNoUnderflowExpr =
        bvAddNoUnderflowExprCache.create(arg0, arg1)

    private val bvSubNoOverflowExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvSubNoOverflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun mkBvSubNoOverflowExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvSubNoOverflowExpr =
        bvSubNoOverflowExprCache.create(arg0, arg1)

    private val bvSubNoUnderflowExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean ->
            KBvSubNoUnderflowExpr(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun mkBvSubNoUnderflowExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean): KBvSubNoUnderflowExpr =
        bvSubNoUnderflowExprCache.create(arg0, arg1, isSigned)


    private val bvDivNoOverflowExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvDivNoOverflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun mkBvDivNoOverflowExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvDivNoOverflowExpr =
        bvDivNoOverflowExprCache.create(arg0, arg1)

    private val bvNegNoOverflowExprCache = mkCache { value: KExpr<KBvSort> -> KBvNegNoOverflowExpr(this, value) }
    fun mkBvNegationNoOverflowExpr(value: KExpr<KBvSort>): KBvNegNoOverflowExpr =
        bvNegNoOverflowExprCache.create(value)

    private val bvMulNoOverflowExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean ->
            KBvMulNoOverflowExpr(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun mkBvMulNoOverflowExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>, isSigned: Boolean): KBvMulNoOverflowExpr =
        bvMulNoOverflowExprCache.create(arg0, arg1, isSigned)

    private val bvMulNoUnderflowExprCache =
        mkCache { arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort> ->
            KBvMulNoUnderflowExpr(
                this,
                arg0,
                arg1,
            )
        }

    fun mkBvMulNoUnderflowExpr(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KBvMulNoUnderflowExpr =
        bvMulNoUnderflowExprCache.create(arg0, arg1)

    // quantifiers
    private val existentialQuantifierCache = mkCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KExistentialQuantifier(this, body, bounds)
    }

    fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        existentialQuantifierCache.create(body, bounds)

    private val universalQuantifierCache = mkCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KUniversalQuantifier(this, body, bounds)
    }

    fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        universalQuantifierCache.create(body, bounds)

    // utils
    private val exprSortCache = mkCache { expr: KExpr<*> -> with(expr) { sort() } }
    val <T : KSort> KExpr<T>.sort: T
        get() = exprSortCache.create(this).uncheckedCast()

    private val exprDeclCache = mkCache { expr: KApp<*, *> -> with(expr) { decl() } }
    val <T : KSort> KApp<T, *>.decl: KDecl<T>
        get() = exprDeclCache.create(this).uncheckedCast()

    /*
    * declarations
    * */

    // functions
    private val funcDeclCache = mkCache { name: String, sort: KSort, args: List<KSort> ->
        ensureContextMatch(sort)
        ensureContextMatch(args)
        KFuncDecl(this, name, sort, args)
    }

    fun <T : KSort> mkFuncDecl(name: String, sort: T, args: List<KSort>): KFuncDecl<T> =
        if (args.isEmpty()) mkConstDecl(name, sort) else funcDeclCache.create(name, sort, args).cast()

    private val constDeclCache = mkCache { name: String, sort: KSort ->
        ensureContextMatch(sort)
        KConstDecl(this, name, sort)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> mkConstDecl(name: String, sort: T): KConstDecl<T> = constDeclCache.create(name, sort).cast()

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
    private val falseDeclCache = mkCache<KFalseDecl> { KFalseDecl(this) }
    fun mkFalseDecl(): KFalseDecl = falseDeclCache.create()

    private val trueDeclCache = mkCache<KTrueDecl> { KTrueDecl(this) }
    fun mkTrueDecl(): KTrueDecl = trueDeclCache.create()

    private val andDeclCache = mkCache<KAndDecl> { KAndDecl(this) }
    fun mkAndDecl(): KAndDecl = andDeclCache.create()

    private val orDeclCache = mkCache<KOrDecl> { KOrDecl(this) }
    fun mkOrDecl(): KOrDecl = orDeclCache.create()

    private val notDeclCache = mkCache<KNotDecl> { KNotDecl(this) }
    fun mkNotDecl(): KNotDecl = notDeclCache.create()

    private val impliesDeclCache = mkCache<KImpliesDecl> { KImpliesDecl(this) }
    fun mkImpliesDecl(): KImpliesDecl = impliesDeclCache.create()

    private val xorDeclCache = mkCache<KXorDecl> { KXorDecl(this) }
    fun mkXorDecl(): KXorDecl = xorDeclCache.create()

    private val eqDeclCache = mkContextCheckingCache { arg: KSort ->
        KEqDecl(this, arg)
    }

    fun <T : KSort> mkEqDecl(arg: T): KEqDecl<T> = eqDeclCache.create(arg).cast()

    private val distinctDeclCache = mkContextCheckingCache { arg: KSort ->
        KDistinctDecl(this, arg)
    }

    fun <T : KSort> mkDistinctDecl(arg: T): KDistinctDecl<T> = distinctDeclCache.create(arg).cast()

    private val iteDeclCache = mkContextCheckingCache { arg: KSort ->
        KIteDecl(this, arg)
    }

    fun <T : KSort> mkIteDecl(arg: T): KIteDecl<T> = iteDeclCache.create(arg).cast()

    // array
    private val arraySelectDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArraySelectDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArraySelectDecl(array: KArraySort<D, R>): KArraySelectDecl<D, R> =
        arraySelectDeclCache.create(array).cast()

    private val arrayStoreDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArrayStoreDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArrayStoreDecl(array: KArraySort<D, R>): KArrayStoreDecl<D, R> =
        arrayStoreDeclCache.create(array).cast()

    private val arrayConstDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArrayConstDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArrayConstDecl(array: KArraySort<D, R>): KArrayConstDecl<D, R> =
        arrayConstDeclCache.create(array).cast()

    // arith
    private val arithAddDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithAddDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithAddDecl(arg: T): KArithAddDecl<T> = arithAddDeclCache.create(arg).cast()

    private val arithSubDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithSubDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithSubDecl(arg: T): KArithSubDecl<T> = arithSubDeclCache.create(arg).cast()

    private val arithMulDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithMulDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithMulDecl(arg: T): KArithMulDecl<T> = arithMulDeclCache.create(arg).cast()

    private val arithDivDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithDivDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithDivDecl(arg: T): KArithDivDecl<T> = arithDivDeclCache.create(arg).cast()

    private val arithPowerDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithPowerDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithPowerDecl(arg: T): KArithPowerDecl<T> = arithPowerDeclCache.create(arg).cast()

    private val arithUnaryMinusDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithUnaryMinusDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithUnaryMinusDecl(arg: T): KArithUnaryMinusDecl<T> =
        arithUnaryMinusDeclCache.create(arg).cast()

    private val arithGeDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithGeDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithGeDecl(arg: T): KArithGeDecl<T> = arithGeDeclCache.create(arg).cast()

    private val arithGtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithGtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithGtDecl(arg: T): KArithGtDecl<T> = arithGtDeclCache.create(arg).cast()

    private val arithLeDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLeDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLeDecl(arg: T): KArithLeDecl<T> = arithLeDeclCache.create(arg).cast()

    private val arithLtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLtDecl(arg: T): KArithLtDecl<T> = arithLtDeclCache.create(arg).cast()


    // int
    private val intModDeclCache = mkCache<KIntModDecl> { KIntModDecl(this) }
    fun mkIntModDecl(): KIntModDecl = intModDeclCache.create()

    private val intToRealDeclCache = mkCache<KIntToRealDecl> { KIntToRealDecl(this) }
    fun mkIntToRealDecl(): KIntToRealDecl = intToRealDeclCache.create()

    private val intRemDeclCache = mkCache<KIntRemDecl> { KIntRemDecl(this) }
    fun mkIntRemDecl(): KIntRemDecl = intRemDeclCache.create()

    private val intNumDeclCache = mkCache { value: String -> KIntNumDecl(this, value) }
    fun mkIntNumDecl(value: String): KIntNumDecl = intNumDeclCache.create(value)

    // real
    private val realIsIntDeclCache = mkCache<KRealIsIntDecl> { KRealIsIntDecl(this) }
    fun mkRealIsIntDecl(): KRealIsIntDecl = realIsIntDeclCache.create()

    private val realToIntDeclCache = mkCache<KRealToIntDecl> { KRealToIntDecl(this) }
    fun mkRealToIntDecl(): KRealToIntDecl = realToIntDeclCache.create()

    private val realNumDeclCache = mkCache { value: String -> KRealNumDecl(this, value) }
    fun mkRealNumDecl(value: String): KRealNumDecl = realNumDeclCache.create(value)

    private val bv1DeclCache = mkCache { value: Boolean -> KBitVec1ValueDecl(this, value) }
    private val bv8DeclCache = mkCache { value: Byte -> KBitVec8ValueDecl(this, value) }
    private val bv16DeclCache = mkCache { value: Short -> KBitVec16ValueDecl(this, value) }
    private val bv32DeclCache = mkCache { value: Int -> KBitVec32ValueDecl(this, value) }
    private val bv64DeclCache = mkCache { value: Long -> KBitVec64ValueDecl(this, value) }
    private val bvCustomSizeDeclCache = mkCache { value: String, sizeBits: UInt ->
        KBitVecCustomSizeValueDecl(this, value, sizeBits)
    }

    fun mkBvDecl(value: Boolean): KDecl<KBv1Sort> = bv1DeclCache.create(value)
    fun mkBvDecl(value: Byte): KDecl<KBv8Sort> = bv8DeclCache.create(value)
    fun mkBvDecl(value: Short): KDecl<KBv16Sort> = bv16DeclCache.create(value)
    fun mkBvDecl(value: Int): KDecl<KBv32Sort> = bv32DeclCache.create(value)
    fun mkBvDecl(value: Long): KDecl<KBv64Sort> = bv64DeclCache.create(value)

    fun mkBvDecl(value: String, sizeBits: UInt): KDecl<KBvSort> = when (sizeBits.toInt()) {
        1 -> mkBvDecl(value.toInt(radix = 2) != 0).cast()
        Byte.SIZE_BITS -> mkBvDecl(value.toByte(radix = 2)).cast()
        Short.SIZE_BITS -> mkBvDecl(value.toShort(radix = 2)).cast()
        Int.SIZE_BITS -> mkBvDecl(value.toInt(radix = 2)).cast()
        Long.SIZE_BITS -> mkBvDecl(value.toLong(radix = 2)).cast()
        else -> bvCustomSizeDeclCache.create(value, sizeBits).cast()
    }

    private val bvNotDeclCache = mkCache { sort: KBvSort -> KBvNotDecl(this, sort) }
    fun mkBvNotDecl(sort: KBvSort): KBvNotDecl = bvNotDeclCache.create(sort)

    private val bvRedAndDeclCache = mkCache { sort: KBvSort -> KBvReductionAndDecl(this, sort) }
    fun mkBvReductionAndDecl(sort: KBvSort): KBvReductionAndDecl = bvRedAndDeclCache.create(sort)

    private val bvRedOrDeclCache = mkCache { sort: KBvSort -> KBvReductionOrDecl(this, sort) }
    fun mkBvReductionOrDecl(sort: KBvSort): KBvReductionOrDecl = bvRedOrDeclCache.create(sort)

    private val bvAndDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvAndDecl(this, arg0, arg1) }
    fun mkBvAndDecl(arg0: KBvSort, arg1: KBvSort): KBvAndDecl = bvAndDeclCache.create(arg0, arg1)

    private val bvOrDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvOrDecl(this, arg0, arg1) }
    fun mkBvOrDecl(arg0: KBvSort, arg1: KBvSort): KBvOrDecl = bvOrDeclCache.create(arg0, arg1)

    private val bvXorDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvXorDecl(this, arg0, arg1) }
    fun mkBvXorDecl(arg0: KBvSort, arg1: KBvSort): KBvXorDecl = bvXorDeclCache.create(arg0, arg1)

    private val bvNAndDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvNAndDecl(this, arg0, arg1) }
    fun mkBvNAndDecl(arg0: KBvSort, arg1: KBvSort): KBvNAndDecl = bvNAndDeclCache.create(arg0, arg1)

    private val bvNorDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvNorDecl(this, arg0, arg1) }
    fun mkBvNorDecl(arg0: KBvSort, arg1: KBvSort): KBvNorDecl = bvNorDeclCache.create(arg0, arg1)

    private val bvXNorDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvXNorDecl(this, arg0, arg1) }
    fun mkBvXNorDecl(arg0: KBvSort, arg1: KBvSort): KBvXNorDecl = bvXNorDeclCache.create(arg0, arg1)

    private val bvNegDeclCache = mkCache { sort: KBvSort -> KBvNegationDecl(this, sort) }
    fun mkBvNegationDecl(sort: KBvSort): KBvNegationDecl = bvNegDeclCache.create(sort)

    private val bvAddDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvAddDecl(this, arg0, arg1) }
    fun mkBvAddDecl(arg0: KBvSort, arg1: KBvSort): KBvAddDecl = bvAddDeclCache.create(arg0, arg1)

    private val bvSubDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSubDecl(this, arg0, arg1) }
    fun mkBvSubDecl(arg0: KBvSort, arg1: KBvSort): KBvSubDecl = bvSubDeclCache.create(arg0, arg1)

    private val bvMulDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvMulDecl(this, arg0, arg1) }
    fun mkBvMulDecl(arg0: KBvSort, arg1: KBvSort): KBvMulDecl = bvMulDeclCache.create(arg0, arg1)

    private val bvUnsignedDivDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedDivDecl(this, arg0, arg1) }
    fun mkBvUnsignedDivDecl(arg0: KBvSort, arg1: KBvSort): KBvUnsignedDivDecl = bvUnsignedDivDeclCache.create(arg0, arg1)

    private val bvSignedDivDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedDivDecl(this, arg0, arg1) }
    fun mkBvSignedDivDecl(arg0: KBvSort, arg1: KBvSort): KBvSignedDivDecl = bvSignedDivDeclCache.create(arg0, arg1)

    private val bvUnsignedRemDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedRemDecl(this, arg0, arg1) }
    fun mkBvUnsignedRemDecl(arg0: KBvSort, arg1: KBvSort): KBvUnsignedRemDecl = bvUnsignedRemDeclCache.create(arg0, arg1)

    private val bvSignedRemDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedRemDecl(this, arg0, arg1) }
    fun mkBvSignedRemDecl(arg0: KBvSort, arg1: KBvSort): KBvSignedRemDecl = bvSignedRemDeclCache.create(arg0, arg1)

    private val bvSignedModDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedModDecl(this, arg0, arg1) }
    fun mkBvSignedModDecl(arg0: KBvSort, arg1: KBvSort): KBvSignedModDecl = bvSignedModDeclCache.create(arg0, arg1)

    private val bvUnsignedLessDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedLessDecl(this, arg0, arg1) }
    fun mkBvUnsignedLessDecl(arg0: KBvSort, arg1: KBvSort): KBvUnsignedLessDecl = bvUnsignedLessDeclCache.create(arg0, arg1)

    private val bvSignedLessDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedLessDecl(this, arg0, arg1) }
    fun mkBvSignedLessDecl(arg0: KBvSort, arg1: KBvSort): KBvSignedLessDecl = bvSignedLessDeclCache.create(arg0, arg1)

    private val bvSignedLessOrEqualDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedLessOrEqualDecl(this, arg0, arg1) }

    fun mkBvSignedLessOrEqualDecl(arg0: KBvSort, arg1: KBvSort): KBvSignedLessOrEqualDecl =
        bvSignedLessOrEqualDeclCache.create(arg0, arg1)

    private val bvUnsignedLessOrEqualDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedLessOrEqualDecl(this, arg0, arg1) }

    fun mkBvUnsignedLessOrEqualDecl(arg0: KBvSort, arg1: KBvSort): KBvUnsignedLessOrEqualDecl =
        bvUnsignedLessOrEqualDeclCache.create(arg0, arg1)

    private val bvUnsignedGreaterOrEqualDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedGreaterOrEqualDecl(this, arg0, arg1) }

    fun mkBvUnsignedGreaterOrEqualDecl(arg0: KBvSort, arg1: KBvSort): KBvUnsignedGreaterOrEqualDecl =
        bvUnsignedGreaterOrEqualDeclCache.create(arg0, arg1)

    private val bvSignedGreaterOrEqualDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedGreaterOrEqualDecl(this, arg0, arg1) }

    fun mkBvSignedGreaterOrEqualDecl(arg0: KBvSort, arg1: KBvSort): KBvSignedGreaterOrEqualDecl =
        bvSignedGreaterOrEqualDeclCache.create(arg0, arg1)

    private val bvUnsignedGreaterDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvUnsignedGreaterDecl(this, arg0, arg1) }
    fun mkBvUnsignedGreaterDecl(arg0: KBvSort, arg1: KBvSort): KBvUnsignedGreaterDecl =
        bvUnsignedGreaterDeclCache.create(arg0, arg1)

    private val bvSignedGreaterDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvSignedGreaterDecl(this, arg0, arg1) }
    fun mkBvSignedGreaterDecl(arg0: KBvSort, arg1: KBvSort): KBvSignedGreaterDecl = bvSignedGreaterDeclCache.create(arg0, arg1)

    private val concatDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KConcatDecl(this, arg0, arg1) }
    fun mkConcatDecl(arg0: KBvSort, arg1: KBvSort): KConcatDecl = concatDeclCache.create(arg0, arg1)

    private val extractDeclCache = mkCache { high: Int, low: Int, value: KExpr<KBvSort> ->
        KExtractDecl(this, high, low, value)
    }

    fun mkExtractDecl(high: Int, low: Int, value: KExpr<KBvSort>) = extractDeclCache.create(high, low, value)

    private val signExtDeclCache = mkCache { i: Int, value: KBvSort -> KSignExtDecl(this, i, value) }
    fun mkSignExtensionDecl(i: Int, value: KBvSort) = signExtDeclCache.create(i, value)

    private val zeroExtDeclCache = mkCache { i: Int, value: KBvSort -> KZeroExtDecl(this, i, value) }
    fun mkZeroExtensionDecl(i: Int, value: KBvSort) = zeroExtDeclCache.create(i, value)

    private val repeatDeclCache = mkCache { i: Int, value: KBvSort -> KRepeatDecl(this, i, value) }
    fun mkRepeatDecl(i: Int, value: KBvSort) = repeatDeclCache.create(i, value)

    private val bvShiftLeftDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvShiftLeftDecl(this, arg0, arg1) }
    fun mkBvShiftLeftDecl(arg0: KBvSort, arg1: KBvSort): KBvShiftLeftDecl = bvShiftLeftDeclCache.create(arg0, arg1)

    private val bvLogicalShiftRightDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvLogicalShiftRightDecl(this, arg0, arg1) }
    fun mkBvLogicalShiftRightDecl(arg0: KBvSort, arg1: KBvSort): KBvLogicalShiftRightDecl =
        bvLogicalShiftRightDeclCache.create(arg0, arg1)

    private val bvArithShiftRightDeclCache = mkCache { arg0: KBvSort, arg1: KBvSort -> KBvArithShiftRightDecl(this, arg0, arg1) }
    fun mkBvArithShiftRightDecl(arg0: KBvSort, arg1: KBvSort): KBvArithShiftRightDecl =
        bvArithShiftRightDeclCache.create(arg0, arg1)

    private val bvRotateLeftDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort -> KBvRotateLeftDecl(this, arg0, arg1) }

    fun mkBvRotateLeftDecl(arg0: KBvSort, arg1: KBvSort): KBvRotateLeftDecl =
        bvRotateLeftDeclCache.create(arg0, arg1)

    private val bvRotateRightDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort -> KBvRotateRightDecl(this, arg0, arg1) }

    fun mkBvRotateRightDecl(arg0: KBvSort, arg1: KBvSort): KBvRotateRightDecl =
        bvRotateRightDeclCache.create(arg0, arg1)

    private val bv2IntDeclCache = mkCache { value: KBvSort, isSigned: Boolean -> KBv2IntDecl(this, value, isSigned) }
    fun mkBv2IntDecl(value: KBvSort, isSigned: Boolean) = bv2IntDeclCache.create(value, isSigned)

    private val bvAddNoOverflowDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort, isSigned: Boolean ->
            KBvAddNoOverflowDecl(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun mkBvAddNoOverflowDecl(arg0: KBvSort, arg1: KBvSort, isSigned: Boolean): KBvAddNoOverflowDecl =
        bvAddNoOverflowDeclCache.create(arg0, arg1, isSigned)

    private val bvAddNoUnderflowDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort ->
            KBvAddNoUnderflowDecl(
                this,
                arg0,
                arg1,
            )
        }

    fun mkBvAddNoUnderflowDecl(arg0: KBvSort, arg1: KBvSort): KBvAddNoUnderflowDecl =
        bvAddNoUnderflowDeclCache.create(arg0, arg1)

    private val bvSubNoOverflowDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort ->
            KBvSubNoOverflowDecl(
                this,
                arg0,
                arg1,
            )
        }

    fun mkBvSubNoOverflowDecl(arg0: KBvSort, arg1: KBvSort): KBvSubNoOverflowDecl =
        bvSubNoOverflowDeclCache.create(arg0, arg1)


    private val bvSubUnderflowDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort, isSigned: Boolean ->
            KBvSubNoUnderflowDecl(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun mkBvSubNoUnderflowDecl(arg0: KBvSort, arg1: KBvSort, isSigned: Boolean): KBvSubNoUnderflowDecl =
        bvSubUnderflowDeclCache.create(arg0, arg1, isSigned)

    private val bvDivNoOverflowDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort ->
            KBvDivNoOverflowDecl(
                this,
                arg0,
                arg1
            )
        }

    fun mkBvDivNoOverflowDecl(arg0: KBvSort, arg1: KBvSort): KBvDivNoOverflowDecl =
        bvDivNoOverflowDeclCache.create(arg0, arg1)

    private val bvNegNoOverflowDeclCache = mkCache { value: KBvSort -> KBvNegNoOverflowDecl(this, value) }

    fun mkBvNegNoOverflowDecl(value: KBvSort): KBvNegNoOverflowDecl =
        bvNegNoOverflowDeclCache.create(value)

    private val bvMulNoOverflowDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort, isSigned: Boolean ->
            KBvMulNoOverflowDecl(
                this,
                arg0,
                arg1,
                isSigned
            )
        }

    fun mkBvMulNoOverflowDecl(arg0: KBvSort, arg1: KBvSort, isSigned: Boolean): KBvMulNoOverflowDecl =
        bvMulNoOverflowDeclCache.create(arg0, arg1, isSigned)

    private val bvMulNoUnderflowDeclCache =
        mkCache { arg0: KBvSort, arg1: KBvSort ->
            KBvMulNoUnderflowDecl(
                this,
                arg0,
                arg1,
            )
        }

    fun mkBvMulNoUnderflowDecl(arg0: KBvSort, arg1: KBvSort): KBvMulNoUnderflowDecl =
        bvMulNoUnderflowDeclCache.create(arg0, arg1)

    /*
    * KAst
    * */

    // toString cache
    private val astStringReprCache = mkCache { ast: KAst -> ast.print() }
    val KAst.stringRepr: String
        get() = astStringReprCache.create(this)

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

    private fun <T, A0 : KAst> mkContextCheckingCache(builder: (A0) -> T) = mkCache { a0: A0 ->
        ensureContextMatch(a0)
        builder(a0)
    }

    private fun <T, A0 : List<KAst>> mkContextListCheckingCache(builder: (A0) -> T) = mkCache { a0: A0 ->
        ensureContextMatch(a0)
        builder(a0)
    }

    private fun <T, A0 : KAst, A1 : KAst> mkContextCheckingCache(builder: (A0, A1) -> T) = mkCache { a0: A0, a1: A1 ->
        ensureContextMatch(a0, a1)
        builder(a0, a1)
    }

    private fun <T, A0 : KAst, A1 : KAst, A2 : KAst> mkContextCheckingCache(builder: (A0, A1, A2) -> T) =
        mkCache { a0: A0, a1: A1, a2: A2 ->
            ensureContextMatch(a0, a1, a2)
            builder(a0, a1, a2)
        }

    private inline fun <reified T, reified Base> Base.cast(): T where T : Base = this as T

    @Suppress("UNCHECKED_CAST")
    private fun <Base, T> Base.uncheckedCast(): T = this as T
}
