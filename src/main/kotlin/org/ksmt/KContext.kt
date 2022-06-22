package org.ksmt

import org.ksmt.cache.mkCache
import org.ksmt.decl.*
import org.ksmt.expr.*
import org.ksmt.sort.*
import java.math.BigInteger

@Suppress("UNCHECKED_CAST", "TYPE_MISMATCH_WARNING")
open class KContext {
    /*
    * sorts
    * */
    fun mkBoolSort() = KBoolSort
    val arraySortCache = mkCache { domain: KSort, range: KSort -> KArraySort(domain, range) }
    fun <D : KSort, R : KSort> mkArraySort(domain: D, range: R): KArraySort<D, R> =
        arraySortCache(domain, range) as KArraySort<D, R>

    fun mkIntSort() = KIntSort
    fun mkRealSort() = KRealSort

    /*
    * expressions
    * */
    // bool
    val andCache = mkCache { args: List<KExpr<KBoolSort>> -> KAndExpr(args) }
    fun mkAnd(args: List<KExpr<KBoolSort>>): KAndExpr = andCache(args)
    fun mkAnd(vararg args: KExpr<KBoolSort>) = mkAnd(args.toList())
    val orCache = mkCache { args: List<KExpr<KBoolSort>> -> KOrExpr(args) }
    fun mkOr(args: List<KExpr<KBoolSort>>): KOrExpr = orCache(args)
    fun mkOr(vararg args: KExpr<KBoolSort>) = mkOr(args.toList())
    val notCache = mkCache { arg: KExpr<KBoolSort> -> KNotExpr(arg) }
    fun mkNot(arg: KExpr<KBoolSort>): KNotExpr = notCache(arg)
    fun mkTrue() = KTrue
    fun mkFalse() = KFalse
    val eqCache = mkCache { l: KExpr<KSort>, r: KExpr<KSort> -> KEqExpr(l, r) }
    fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>): KEqExpr<T> =
        eqCache(lhs as KExpr<KSort>, rhs as KExpr<KSort>) as KEqExpr<T>

    val iteCache = mkCache { c: KExpr<KBoolSort>, t: KExpr<KSort>, f: KExpr<KSort> ->
        KIteExpr(c, t, f)
    }

    fun <T : KSort> mkIte(condition: KExpr<KBoolSort>, trueBranch: KExpr<T>, falseBranch: KExpr<T>): KIteExpr<T> =
        iteCache(condition, trueBranch as KExpr<KSort>, falseBranch as KExpr<KSort>) as KIteExpr<T>

    infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
    operator fun KExpr<KBoolSort>.not() = mkNot(this)
    infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
    infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
    val Boolean.expr
        get() = if (this) mkTrue() else mkFalse()

    // functions
    /*
    * For builtin declarations e.g. KAndDecl, mkApp must return the same object as a corresponding builder.
    * For example, mkApp(KAndDecl, a, b) and mkAnd(a, b) must end up with the same KAndExpr object.
    * To achieve such behaviour we override apply for all builtin declarations.
    */
    fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = with(decl) { apply(args) }

    val functionAppCache = mkCache { decl: KDecl<*>, args: List<KExpr<*>> ->
        KFunctionApp(decl, args)
    }

    internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>): KApp<T, *> = when {
        args.isEmpty() -> mkConstApp(decl)
        else -> functionAppCache(decl, args) as KFunctionApp<T>
    }

    val constAppCache = mkCache { decl: KDecl<*> -> KConst(decl) }
    fun <T : KSort> mkConstApp(decl: KDecl<T>): KConst<T> = constAppCache(decl) as KConst<T>

    fun <T : KSort> T.mkConst(name: String) = with(mkConstDecl(name)) { apply() }

    // array
    val arrayStoreCache = mkCache { a: KExpr<KArraySort<KSort, KSort>>, i: KExpr<KSort>, v: KExpr<KSort> ->
        KArrayStore(a, i, v)
    }

    fun <D : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArraySort<D, R>>, index: KExpr<D>, value: KExpr<R>
    ): KArrayStore<D, R> =
        arrayStoreCache(
            array as KExpr<KArraySort<KSort, KSort>>,
            index as KExpr<KSort>,
            value as KExpr<KSort>
        ) as KArrayStore<D, R>

    val arraySelectCache = mkCache { array: KExpr<KArraySort<KSort, KSort>>, index: KExpr<KSort> ->
        KArraySelect(array, index)
    }

    fun <D : KSort, R : KSort> mkArraySelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>): KArraySelect<D, R> =
        arraySelectCache(array as KExpr<KArraySort<KSort, KSort>>, index as KExpr<KSort>) as KArraySelect<D, R>

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
        mkArrayStore(this, index, value)

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) = mkArraySelect(this, index)

    // arith
    val arithAddCache = mkCache { args: List<KExpr<KArithSort<*>>> -> KAddArithExpr(args) }
    fun <T : KArithSort<T>> mkArithAdd(args: List<KExpr<T>>): KAddArithExpr<T> =
        arithAddCache(args as List<KExpr<KArithSort<*>>>) as KAddArithExpr<T>

    val arithMulCache = mkCache { args: List<KExpr<KArithSort<*>>> -> KMulArithExpr(args) }
    fun <T : KArithSort<T>> mkArithMul(args: List<KExpr<T>>): KMulArithExpr<T> =
        arithMulCache(args as List<KExpr<KArithSort<*>>>) as KMulArithExpr<T>

    val arithSubCache = mkCache { args: List<KExpr<KArithSort<*>>> -> KSubArithExpr(args) }
    fun <T : KArithSort<T>> mkArithSub(args: List<KExpr<T>>): KSubArithExpr<T> =
        arithSubCache(args as List<KExpr<KArithSort<*>>>) as KSubArithExpr<T>

    fun <T : KArithSort<T>> mkArithAdd(vararg args: KExpr<T>) = mkArithAdd(args.toList())
    fun <T : KArithSort<T>> mkArithMul(vararg args: KExpr<T>) = mkArithMul(args.toList())
    fun <T : KArithSort<T>> mkArithSub(vararg args: KExpr<T>) = mkArithSub(args.toList())

    val arithUnaryMinusCache = mkCache { arg: KExpr<KArithSort<*>> -> KUnaryMinusArithExpr(arg) }
    fun <T : KArithSort<T>> mkArithUnaryMinus(arg: KExpr<T>): KUnaryMinusArithExpr<T> =
        arithUnaryMinusCache(arg as KExpr<KArithSort<*>>) as KUnaryMinusArithExpr<T>

    val arithDivCache = mkCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> -> KDivArithExpr(l, r) }
    fun <T : KArithSort<T>> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KDivArithExpr<T> =
        arithDivCache(lhs as KExpr<KArithSort<*>>, rhs as KExpr<KArithSort<*>>) as KDivArithExpr<T>

    val arithPowerCache = mkCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> -> KPowerArithExpr(l, r) }
    fun <T : KArithSort<T>> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KPowerArithExpr<T> =
        arithPowerCache(lhs as KExpr<KArithSort<*>>, rhs as KExpr<KArithSort<*>>) as KPowerArithExpr<T>

    val arithLtCache = mkCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> -> KLtArithExpr(l, r) }
    fun <T : KArithSort<T>> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KLtArithExpr<T> =
        arithLtCache(lhs as KExpr<KArithSort<*>>, rhs as KExpr<KArithSort<*>>) as KLtArithExpr<T>

    val arithLeCache = mkCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> -> KLeArithExpr(l, r) }
    fun <T : KArithSort<T>> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KLeArithExpr<T> =
        arithLeCache(lhs as KExpr<KArithSort<*>>, rhs as KExpr<KArithSort<*>>) as KLeArithExpr<T>

    val arithGtCache = mkCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> -> KGtArithExpr(l, r) }
    fun <T : KArithSort<T>> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KGtArithExpr<T> =
        arithGtCache(lhs as KExpr<KArithSort<*>>, rhs as KExpr<KArithSort<*>>) as KGtArithExpr<T>

    val arithGeCache = mkCache { l: KExpr<KArithSort<*>>, r: KExpr<KArithSort<*>> -> KGeArithExpr(l, r) }
    fun <T : KArithSort<T>> mkArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KGeArithExpr<T> =
        arithGeCache(lhs as KExpr<KArithSort<*>>, rhs as KExpr<KArithSort<*>>) as KGeArithExpr<T>

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
    val intModCache = mkCache { l: KExpr<KIntSort>, r: KExpr<KIntSort> -> KModIntExpr(l, r) }
    fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KModIntExpr = intModCache(lhs, rhs)
    val intRemCache = mkCache { l: KExpr<KIntSort>, r: KExpr<KIntSort> -> KRemIntExpr(l, r) }
    fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KRemIntExpr = intRemCache(lhs, rhs)
    val intToRealCache = mkCache { arg: KExpr<KIntSort> -> KToRealIntExpr(arg) }
    fun mkIntToReal(arg: KExpr<KIntSort>): KToRealIntExpr = intToRealCache(arg)
    val int32NumCache = mkCache { v: Int -> KInt32NumExpr(v) }
    fun mkIntNum(value: Int): KInt32NumExpr = int32NumCache(value)
    val int64NumCache = mkCache { v: Long -> KInt64NumExpr(v) }
    fun mkIntNum(value: Long): KInt64NumExpr = int64NumCache(value)
    val intBigNumCache = mkCache { v: BigInteger -> KIntBigNumExpr(v) }
    fun mkIntNum(value: BigInteger): KIntBigNumExpr = intBigNumCache(value)
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
    val realToIntCache = mkCache { arg: KExpr<KRealSort> -> KToIntRealExpr(arg) }
    fun mkRealToInt(arg: KExpr<KRealSort>): KToIntRealExpr = realToIntCache(arg)
    val realIsIntCache = mkCache { arg: KExpr<KRealSort> -> KIsIntRealExpr(arg) }
    fun mkRealIsInt(arg: KExpr<KRealSort>): KIsIntRealExpr = realIsIntCache(arg)

    val realNumCache = mkCache { numerator: KIntNumExpr, denominator: KIntNumExpr ->
        KRealNumExpr(numerator, denominator)
    }

    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr): KRealNumExpr = realNumCache(numerator, denominator)

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
        KExistentialQuantifier(body, bounds)
    }

    fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        existentialQuantifierCache(body, bounds)

    val universalQuantifierCache = mkCache { body: KExpr<KBoolSort>, bounds: List<KDecl<*>> ->
        KUniversalQuantifier(body, bounds)
    }

    fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        universalQuantifierCache(body, bounds)

    // utils
    val exprSortCache = mkCache { expr: KExpr<*> -> with(expr) { sort() } }
    val <T : KSort> KExpr<T>.sort: T
        get() = exprSortCache(this) as T

    val exprDeclCache = mkCache { expr: KApp<*, *> -> with(expr) { decl() } }
    val <T : KSort> KApp<T, *>.decl: KDecl<T>
        get() = exprDeclCache(this) as KDecl<T>

    /*
    * declarations
    * */

    // functions
    val funcDeclCache = mkCache { name: String, sort: KSort, args: List<KSort> -> KFuncDecl(name, sort, args) }
    fun <T : KSort> mkFuncDecl(name: String, sort: T, args: List<KSort>): KFuncDecl<T> =
        funcDeclCache(name, sort, args) as KFuncDecl<T>

    val constDeclCache = mkCache { name: String, sort: KSort -> KConstDecl(name, sort) }
    fun <T : KSort> mkConstDecl(name: String, sort: T): KConstDecl<T> =
        constDeclCache(name, sort) as KConstDecl<T>

    fun <T : KSort> T.mkConstDecl(name: String) = mkConstDecl(name, this)

    // bool
    fun mkFalseDecl(): KDecl<KBoolSort> = KFalseDecl
    fun mkTrueDecl(): KDecl<KBoolSort> = KTrueDecl
    fun mkAndDecl(): KDecl<KBoolSort> = KAndDecl
    fun mkOrDecl(): KDecl<KBoolSort> = KOrDecl
    fun mkNotDecl(): KDecl<KBoolSort> = KNotDecl
    fun <T : KSort> mkEqDecl(arg: T): KDecl<KBoolSort> = KEqDecl(arg)
    fun <T : KSort> mkIteDecl(arg: T): KDecl<T> = KIteDecl(arg)

    // array
    fun <D : KSort, R : KSort> mkArraySelectDecl(array: KArraySort<D, R>): KDecl<R> = KArraySelectDecl(array)

    fun <D : KSort, R : KSort> mkArrayStoreDecl(array: KArraySort<D, R>): KDecl<KArraySort<D, R>> =
        KArrayStoreDecl(array)

    // arith
    fun <T : KArithSort<T>> mkArithAddDecl(arg: T) = KArithAddDecl(arg)
    fun <T : KArithSort<T>> mkArithSubDecl(arg: T) = KArithSubDecl(arg)
    fun <T : KArithSort<T>> mkArithMulDecl(arg: T) = KArithMulDecl(arg)
    fun <T : KArithSort<T>> mkArithDivDecl(arg: T) = KArithDivDecl(arg)
    fun <T : KArithSort<T>> mkArithPowerDecl(arg: T) = KArithPowerDecl(arg)
    fun <T : KArithSort<T>> mkArithUnaryMinusDecl(arg: T) = KArithUnaryMinusDecl(arg)
    fun <T : KArithSort<T>> mkArithGeDecl(arg: T) = KArithGeDecl(arg)
    fun <T : KArithSort<T>> mkArithGtDecl(arg: T) = KArithGtDecl(arg)
    fun <T : KArithSort<T>> mkArithLeDecl(arg: T) = KArithLeDecl(arg)
    fun <T : KArithSort<T>> mkArithLtDecl(arg: T) = KArithLtDecl(arg)

    // int
    fun mkIntModDecl() = KIntModDecl
    fun mkIntToRealDecl() = KIntToRealDecl
    fun mkIntRemDecl() = KIntRemDecl
    fun mkIntNumDecl(value: String) = KIntNumDecl(value)

    // real
    fun mkRealIsIntDecl() = KRealIsIntDecl
    fun mkRealToIntDecl() = KRealToIntDecl
    fun mkRealNumDecl(value: String) = KRealNumDecl(value)
}
