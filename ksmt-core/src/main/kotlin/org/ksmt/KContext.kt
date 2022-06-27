package org.ksmt

import org.ksmt.cache.mkCache
import org.ksmt.decl.*
import org.ksmt.expr.*
import org.ksmt.sort.*
import java.math.BigInteger

open class KContext {
    /*
    * sorts
    * */
    val boolSortCache = mkCache<KBoolSort> { KBoolSort(this) }
    fun mkBoolSort(): KBoolSort = boolSortCache.create()

    val arraySortCache = mkContextCheckingCache { domain: KSort, range: KSort ->
        KArraySort(this, domain, range)
    }

    fun <D : KSort, R : KSort> mkArraySort(domain: D, range: R): KArraySort<D, R> =
        arraySortCache.create(domain, range).cast()

    val intSortCache = mkCache<KIntSort> { KIntSort(this) }
    fun mkIntSort(): KIntSort = intSortCache.create()

    val realSortCache = mkCache<KRealSort> { KRealSort(this) }
    fun mkRealSort(): KRealSort = realSortCache.create()

    // utils
    val boolSort: KBoolSort
        get() = mkBoolSort()

    val intSort: KIntSort
        get() = mkIntSort()

    val realSort: KRealSort
        get() = mkRealSort()


    /*
    * expressions
    * */
    // bool
    val andCache = mkContextListCheckingCache { args: List<KExpr<KBoolSort>> ->
        KAndExpr(this, args)
    }

    fun mkAnd(args: List<KExpr<KBoolSort>>): KAndExpr = andCache.create(args)
    fun mkAnd(vararg args: KExpr<KBoolSort>) = mkAnd(args.toList())

    val orCache = mkContextListCheckingCache { args: List<KExpr<KBoolSort>> ->
        KOrExpr(this, args)
    }

    fun mkOr(args: List<KExpr<KBoolSort>>): KOrExpr = orCache.create(args)
    fun mkOr(vararg args: KExpr<KBoolSort>) = mkOr(args.toList())

    val notCache = mkContextCheckingCache { arg: KExpr<KBoolSort> ->
        KNotExpr(this, arg)
    }

    fun mkNot(arg: KExpr<KBoolSort>): KNotExpr = notCache.create(arg)

    val trueCache = mkCache<KTrue> { KTrue(this) }
    fun mkTrue() = trueCache.create()

    val falseCache = mkCache<KFalse> { KFalse(this) }
    fun mkFalse() = falseCache.create()

    val eqCache = mkContextCheckingCache { l: KExpr<KSort>, r: KExpr<KSort> ->
        KEqExpr(this, l, r)
    }

    fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>): KEqExpr<T> =
        eqCache.create(lhs.cast(), rhs.cast()).cast()

    val iteCache = mkContextCheckingCache { c: KExpr<KBoolSort>, t: KExpr<KSort>, f: KExpr<KSort> ->
        KIteExpr(this, c, t, f)
    }

    fun <T : KSort> mkIte(condition: KExpr<KBoolSort>, trueBranch: KExpr<T>, falseBranch: KExpr<T>): KIteExpr<T> =
        iteCache.create(condition, trueBranch.cast(), falseBranch.cast()).cast()


    infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
    operator fun KExpr<KBoolSort>.not() = mkNot(this)
    infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
    infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)

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

    val functionAppCache = mkCache { decl: KDecl<*>, args: List<KExpr<*>> ->
        ensureContextMatch(decl)
        ensureContextMatch(args)
        KFunctionApp(this, decl, args)
    }

    internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>): KApp<T, *> =
        if (args.isEmpty()) mkConstApp(decl) else functionAppCache.create(decl, args).cast()

    val constAppCache = mkCache { decl: KDecl<*> ->
        ensureContextMatch(decl)
        KConst(this, decl)
    }

    fun <T : KSort> mkConstApp(decl: KDecl<T>): KConst<T> = constAppCache.create(decl).cast()

    fun <T : KSort> T.mkConst(name: String) = with(mkConstDecl(name)) { apply() }

    // array
    val arrayStoreCache =
        mkContextCheckingCache { a: KExpr<KArraySort<KSort, KSort>>, i: KExpr<KSort>, v: KExpr<KSort> ->
            KArrayStore(this, a, i, v)
        }

    fun <D : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArraySort<D, R>>, index: KExpr<D>, value: KExpr<R>
    ): KArrayStore<D, R> = arrayStoreCache.create(array.cast(), index.cast(), value.cast()).cast()

    val arraySelectCache = mkContextCheckingCache { array: KExpr<KArraySort<KSort, KSort>>, index: KExpr<KSort> ->
        KArraySelect(this, array, index)
    }

    fun <D : KSort, R : KSort> mkArraySelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>): KArraySelect<D, R> =
        arraySelectCache.create(array.cast(), index.cast()).cast()

    val arrayConstCache = mkCache { array: KArraySort<KSort, KSort>, value: KExpr<KSort> ->
        ensureContextMatch(array, value)
        KArrayConst(this, array, value)
    }
    fun <D : KSort, R : KSort> mkArrayConst(arraySort: KArraySort<D, R>, value: KExpr<R>): KArrayConst<D, R> =
        arrayConstCache.create(arraySort as KArraySort<KSort, KSort>, value as KExpr<KSort>) as KArrayConst<D, R>

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
        mkArrayStore(this, index, value)

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) = mkArraySelect(this, index)

    // arith
    val arithAddCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KAddArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithAdd(args: List<KExpr<T>>): KAddArithExpr<T> =
        arithAddCache.create(args.cast()).cast()

    val arithMulCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KMulArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithMul(args: List<KExpr<T>>): KMulArithExpr<T> =
        arithMulCache.create(args.cast()).cast()

    val arithSubCache = mkContextListCheckingCache { args: List<KExpr<KArithSort<*>>> ->
        KSubArithExpr(this, args)
    }

    fun <T : KArithSort<T>> mkArithSub(args: List<KExpr<T>>): KSubArithExpr<T> =
        arithSubCache.create(args.cast()).cast()

    fun <T : KArithSort<T>> mkArithAdd(vararg args: KExpr<T>) = mkArithAdd(args.toList())
    fun <T : KArithSort<T>> mkArithMul(vararg args: KExpr<T>) = mkArithMul(args.toList())
    fun <T : KArithSort<T>> mkArithSub(vararg args: KExpr<T>) = mkArithSub(args.toList())

    val arithUnaryMinusCache = mkContextCheckingCache { arg: KExpr<KArithSort<*>> ->
        KUnaryMinusArithExpr(this, arg)
    }

    fun <T : KArithSort<T>> mkArithUnaryMinus(arg: KExpr<T>): KUnaryMinusArithExpr<T> =
        arithUnaryMinusCache.create(arg.cast()).cast()

    val arithDivCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KDivArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KDivArithExpr<T> =
        arithDivCache.create(lhs.cast(), rhs.cast()).cast()

    val arithPowerCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KPowerArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KPowerArithExpr<T> =
        arithPowerCache.create(lhs.cast(), rhs.cast()).cast()

    val arithLtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KLtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KLtArithExpr<T> =
        arithLtCache.create(lhs.cast(), rhs.cast()).cast()

    val arithLeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KLeArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KLeArithExpr<T> =
        arithLeCache.create(lhs.cast(), rhs.cast()).cast()

    val arithGtCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
        KGtArithExpr(this, l, r)
    }

    fun <T : KArithSort<T>> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KGtArithExpr<T> =
        arithGtCache.create(lhs.cast(), rhs.cast()).cast()

    val arithGeCache = mkContextCheckingCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> ->
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
    val intModCache = mkContextCheckingCache { l: KExpr<KIntSort>, r: KExpr<KIntSort> ->
        KModIntExpr(this, l, r)
    }

    fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KModIntExpr = intModCache.create(lhs, rhs)

    val intRemCache = mkContextCheckingCache { l: KExpr<KIntSort>, r: KExpr<KIntSort> ->
        KRemIntExpr(this, l, r)
    }

    fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KRemIntExpr = intRemCache.create(lhs, rhs)

    val intToRealCache = mkContextCheckingCache { arg: KExpr<KIntSort> ->
        KToRealIntExpr(this, arg)
    }

    fun mkIntToReal(arg: KExpr<KIntSort>): KToRealIntExpr = intToRealCache.create(arg)

    val int32NumCache = mkCache { v: Int ->
        KInt32NumExpr(this, v)
    }

    fun mkIntNum(value: Int): KInt32NumExpr = int32NumCache.create(value)

    val int64NumCache = mkCache { v: Long ->
        KInt64NumExpr(this, v)
    }

    fun mkIntNum(value: Long): KInt64NumExpr = int64NumCache.create(value)

    val intBigNumCache = mkCache { v: BigInteger ->
        KIntBigNumExpr(this, v)
    }

    fun mkIntNum(value: BigInteger): KIntBigNumExpr = intBigNumCache.create(value)
    fun mkIntNum(value: String) =
        value.toIntOrNull()?.let { mkIntNum(it) }
            ?: value.toLongOrNull()?.let { mkIntNum(it) }
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
    val realToIntCache = mkContextCheckingCache { arg: KExpr<KRealSort> ->
        KToIntRealExpr(this, arg)
    }

    fun mkRealToInt(arg: KExpr<KRealSort>): KToIntRealExpr = realToIntCache.create(arg)

    val realIsIntCache = mkContextCheckingCache { arg: KExpr<KRealSort> ->
        KIsIntRealExpr(this, arg)
    }

    fun mkRealIsInt(arg: KExpr<KRealSort>): KIsIntRealExpr = realIsIntCache.create(arg)

    val realNumCache = mkContextCheckingCache { numerator: KIntNumExpr, denominator: KIntNumExpr ->
        KRealNumExpr(this, numerator, denominator)
    }

    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr): KRealNumExpr =
        realNumCache.create(numerator, denominator)

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

    // quantifiers
    val existentialQuantifierCache = mkCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KExistentialQuantifier(this, body, bounds)
    }

    fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        existentialQuantifierCache.create(body, bounds)

    val universalQuantifierCache = mkCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        ensureContextMatch(body)
        ensureContextMatch(bounds)
        KUniversalQuantifier(this, body, bounds)
    }

    fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        universalQuantifierCache.create(body, bounds)

    // utils
    val exprSortCache = mkCache { expr: KExpr<*> -> with(expr) { sort() } }
    val <T : KSort> KExpr<T>.sort: T
        get() = exprSortCache.create(this).uncheckedCast()

    val exprDeclCache = mkCache { expr: KApp<*, *> -> with(expr) { decl() } }
    val <T : KSort> KApp<T, *>.decl: KDecl<T>
        get() = exprDeclCache.create(this).uncheckedCast()

    /*
    * declarations
    * */

    // functions
    val funcDeclCache = mkCache { name: String, sort: KSort, args: List<KSort> ->
        ensureContextMatch(sort)
        ensureContextMatch(args)
        KFuncDecl(this, name, sort, args)
    }

    fun <T : KSort> mkFuncDecl(name: String, sort: T, args: List<KSort>): KFuncDecl<T> =
        if (args.isEmpty()) mkConstDecl(name, sort) else funcDeclCache.create(name, sort, args).cast()

    val constDeclCache = mkCache { name: String, sort: KSort ->
        ensureContextMatch(sort)
        KConstDecl(this, name, sort)
    }

    fun <T : KSort> mkConstDecl(name: String, sort: T): KConstDecl<T> =
        constDeclCache.create(name, sort).cast()

    fun <T : KSort> T.mkConstDecl(name: String) = mkConstDecl(name, this)

    // bool
    val falseDeclCache = mkCache<KFalseDecl> { KFalseDecl(this) }
    fun mkFalseDecl(): KFalseDecl = falseDeclCache.create()

    val trueDeclCache = mkCache<KTrueDecl> { KTrueDecl(this) }
    fun mkTrueDecl(): KTrueDecl = trueDeclCache.create()

    val andDeclCache = mkCache<KAndDecl> { KAndDecl(this) }
    fun mkAndDecl(): KAndDecl = andDeclCache.create()

    val orDeclCache = mkCache<KOrDecl> { KOrDecl(this) }
    fun mkOrDecl(): KOrDecl = orDeclCache.create()

    val notDeclCache = mkCache<KNotDecl> { KNotDecl(this) }
    fun mkNotDecl(): KNotDecl = notDeclCache.create()

    val eqDeclCache = mkContextCheckingCache { arg: KSort ->
        KEqDecl(this, arg)
    }

    fun <T : KSort> mkEqDecl(arg: T): KEqDecl<T> = eqDeclCache.create(arg).cast()

    val iteDeclCache = mkContextCheckingCache { arg: KSort ->
        KIteDecl(this, arg)
    }

    fun <T : KSort> mkIteDecl(arg: T): KIteDecl<T> = iteDeclCache.create(arg).cast()

    // array
    val arraySelectDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArraySelectDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArraySelectDecl(array: KArraySort<D, R>): KArraySelectDecl<D, R> =
        arraySelectDeclCache.create(array).cast()

    val arrayStoreDeclCache = mkContextCheckingCache { array: KArraySort<*, *> ->
        KArrayStoreDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArrayStoreDecl(array: KArraySort<D, R>): KArrayStoreDecl<D, R> =
        arrayStoreDeclCache.create(array).cast()

    val arrayConstDeclCache = mkCache { array: KArraySort<*, *> ->
        ensureContextMatch(array)
        KArrayConstDecl(this, array)
    }

    fun <D : KSort, R : KSort> mkArrayConstDecl(array: KArraySort<D, R>): KArrayConstDecl<D, R> =
        arrayConstDeclCache.create(array) as KArrayConstDecl<D, R>

    // arith
    val arithAddDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithAddDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithAddDecl(arg: T): KArithAddDecl<T> = arithAddDeclCache.create(arg).cast()

    val arithSubDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithSubDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithSubDecl(arg: T): KArithSubDecl<T> = arithSubDeclCache.create(arg).cast()

    val arithMulDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithMulDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithMulDecl(arg: T): KArithMulDecl<T> = arithMulDeclCache.create(arg).cast()

    val arithDivDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithDivDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithDivDecl(arg: T): KArithDivDecl<T> = arithDivDeclCache.create(arg).cast()

    val arithPowerDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithPowerDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithPowerDecl(arg: T): KArithPowerDecl<T> =
        arithPowerDeclCache.create(arg).cast()

    val arithUnaryMinusDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithUnaryMinusDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithUnaryMinusDecl(arg: T): KArithUnaryMinusDecl<T> =
        arithUnaryMinusDeclCache.create(arg).cast()

    val arithGeDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithGeDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithGeDecl(arg: T): KArithGeDecl<T> = arithGeDeclCache.create(arg).cast()

    val arithGtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithGtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithGtDecl(arg: T): KArithGtDecl<T> = arithGtDeclCache.create(arg).cast()

    val arithLeDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLeDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLeDecl(arg: T): KArithLeDecl<T> = arithLeDeclCache.create(arg).cast()

    val arithLtDeclCache = mkContextCheckingCache { arg: KArithSort<*> ->
        KArithLtDecl(this, arg)
    }

    fun <T : KArithSort<T>> mkArithLtDecl(arg: T): KArithLtDecl<T> = arithLtDeclCache.create(arg).cast()


    // int
    val intModDeclCache = mkCache<KIntModDecl> { KIntModDecl(this) }
    fun mkIntModDecl(): KIntModDecl = intModDeclCache.create()

    val intToRealDeclCache = mkCache<KIntToRealDecl> { KIntToRealDecl(this) }
    fun mkIntToRealDecl(): KIntToRealDecl = intToRealDeclCache.create()

    val intRemDeclCache = mkCache<KIntRemDecl> { KIntRemDecl(this) }
    fun mkIntRemDecl(): KIntRemDecl = intRemDeclCache.create()

    val intNumDeclCache = mkCache { value: String -> KIntNumDecl(this, value) }
    fun mkIntNumDecl(value: String): KIntNumDecl = intNumDeclCache.create(value)

    // real
    val realIsIntDeclCache = mkCache<KRealIsIntDecl> { KRealIsIntDecl(this) }
    fun mkRealIsIntDecl(): KRealIsIntDecl = realIsIntDeclCache.create()

    val realToIntDeclCache = mkCache<KRealToIntDecl> { KRealToIntDecl(this) }
    fun mkRealToIntDecl(): KRealToIntDecl = realToIntDeclCache.create()

    val realNumDeclCache = mkCache { value: String -> KRealNumDecl(this, value) }
    fun mkRealNumDecl(value: String): KRealNumDecl = realNumDeclCache.create(value)

    // context utils
    fun ensureContextMatch(vararg args: KAst) {
        for (arg in args) {
            require(this === arg.ctx) { "Context mismatch" }
        }
    }

    fun ensureContextMatch(args: List<KAst>) {
        ensureContextMatch(*args.toTypedArray())
    }

    fun <T, A0 : KAst> mkContextCheckingCache(builder: (A0) -> T) = mkCache { a0: A0 ->
        ensureContextMatch(a0)
        builder(a0)
    }

    fun <T, A0 : List<KAst>> mkContextListCheckingCache(builder: (A0) -> T) = mkCache { a0: A0 ->
        ensureContextMatch(a0)
        builder(a0)
    }

    fun <T, A0 : KAst, A1 : KAst> mkContextCheckingCache(builder: (A0, A1) -> T) = mkCache { a0: A0, a1: A1 ->
        ensureContextMatch(a0, a1)
        builder(a0, a1)
    }

    fun <T, A0 : KAst, A1 : KAst, A2 : KAst> mkContextCheckingCache(builder: (A0, A1, A2) -> T) =
        mkCache { a0: A0, a1: A1, a2: A2 ->
            ensureContextMatch(a0, a1, a2)
            builder(a0, a1, a2)
        }

    private inline fun <reified T, reified Base> Base.cast(): T where T : Base = this as T

    @Suppress("UNCHECKED_CAST")
    private fun <Base, T> Base.uncheckedCast(): T = this as T

}
