package org.ksmt

import org.ksmt.decl.*
import org.ksmt.expr.*
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.*
import java.math.BigInteger

open class KContext {
    /*
    * sorts
    * */
    fun mkBoolSort() = KBoolSort
    fun <D : KSort, R : KSort> mkArraySort(domain: D, range: R) = KArraySort(domain, range)
    fun mkIntSort() = KIntSort
    fun mkRealSort() = KRealSort

    /*
    * expressions
    * */
    // bool
    fun mkAnd(vararg args: KExpr<KBoolSort>) = KAndExpr(args.toList()).intern()
    fun mkAnd(args: List<KExpr<KBoolSort>>) = KAndExpr(args).intern()
    fun mkOr(vararg args: KExpr<KBoolSort>) = KOrExpr(args.toList()).intern()
    fun mkOr(args: List<KExpr<KBoolSort>>) = KOrExpr(args).intern()
    fun mkNot(arg: KExpr<KBoolSort>) = KNotExpr(arg).intern()
    fun mkTrue() = KTrue
    fun mkFalse() = KFalse
    fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>) = KEqExpr(lhs, rhs).intern()
    fun <T : KSort> mkIte(condition: KExpr<KBoolSort>, trueBranch: KExpr<T>, falseBranch: KExpr<T>) =
        KIteExpr(condition, trueBranch, falseBranch).intern()

    infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
    operator fun KExpr<KBoolSort>.not() = mkNot(this)
    infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
    infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
    val Boolean.expr
        get() = if (this) mkTrue() else mkFalse()

    // functions
    internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>) = when {
        args.isEmpty() -> KConst(decl).intern()
        else -> KFunctionApp(decl, args).intern()
    }

    /*
    * For builtin declarations e.g. KAndDecl, mkApp must return the same object as a corresponding builder.
    * For example, mkApp(KAndDecl, a, b) and mkAnd(a, b) must end up with the same KAndExpr object.
    * To achieve such behaviour we override apply for all builtin declarations.
    */
    fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = with(decl) { apply(args) }
    fun <T : KSort> mkConstApp(decl: KConstDecl<T>) = KConst(decl).intern()
    fun <T : KSort> T.mkConst(name: String) = with(mkConstDecl(name)) { apply() }

    // array
    fun <D : KSort, R : KSort> mkArrayStore(array: KExpr<KArraySort<D, R>>, index: KExpr<D>, value: KExpr<R>) =
        KArrayStore(array, index, value).intern()

    fun <D : KSort, R : KSort> mkArraySelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>) =
        KArraySelect(array, index).intern()

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
        mkArrayStore(this, index, value)

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) =
        mkArraySelect(this, index)

    // arith
    fun <T : KArithSort<T>> mkArithAdd(vararg args: KExpr<T>) = KAddArithExpr(args.toList()).intern()
    fun <T : KArithSort<T>> mkArithAdd(args: List<KExpr<T>>) = KAddArithExpr(args).intern()
    fun <T : KArithSort<T>> mkArithMul(vararg args: KExpr<T>) = KMulArithExpr(args.toList()).intern()
    fun <T : KArithSort<T>> mkArithMul(args: List<KExpr<T>>) = KMulArithExpr(args).intern()
    fun <T : KArithSort<T>> mkArithSub(vararg args: KExpr<T>) = KSubArithExpr(args.toList()).intern()
    fun <T : KArithSort<T>> mkArithSub(args: List<KExpr<T>>) = KSubArithExpr(args).intern()

    fun <T : KArithSort<T>> mkArithUnaryMinus(arg: KExpr<T>) = KUnaryMinusArithExpr(arg).intern()
    fun <T : KArithSort<T>> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>) = KDivArithExpr(lhs, rhs).intern()
    fun <T : KArithSort<T>> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>) = KPowerArithExpr(lhs, rhs).intern()

    fun <T : KArithSort<T>> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>) = KLtArithExpr(lhs, rhs).intern()
    fun <T : KArithSort<T>> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>) = KLeArithExpr(lhs, rhs).intern()
    fun <T : KArithSort<T>> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>) = KGtArithExpr(lhs, rhs).intern()
    fun <T : KArithSort<T>> mkArithGe(lhs: KExpr<T>, rhs: KExpr<T>) = KGeArithExpr(lhs, rhs).intern()

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
    fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>) = KModIntExpr(lhs, rhs).intern()
    fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>) = KRemIntExpr(lhs, rhs).intern()
    fun mkIntToReal(arg: KExpr<KIntSort>) = KToRealIntExpr(arg).intern()
    fun mkIntNum(value: Int) = KInt32NumExpr(value).intern()
    fun mkIntNum(value: Long) = KInt64NumExpr(value).intern()
    fun mkIntNum(value: BigInteger) = KIntBigNumExpr(value).intern()
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
    fun mkRealToInt(arg: KExpr<KRealSort>) = KToIntRealExpr(arg).intern()
    fun mkRealIsInt(arg: KExpr<KRealSort>) = KIsIntRealExpr(arg).intern()
    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr) = KRealNumExpr(numerator, denominator).intern()
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
    fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        KExistentialQuantifier(body, bounds).intern()

    fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        KUniversalQuantifier(body, bounds).intern()

    // utils
    val <T : KSort> KExpr<T>.sort: T
        get() = sort()

    val <T : KSort> KApp<T, *>.decl: KDecl<T>
        get() = decl()

    /*
    * declarations
    * */

    // functions
    fun <T : KSort> mkFuncDecl(name: String, sort: T, args: List<KSort>) = KFuncDecl(name, sort, args)
    fun <T : KSort> mkConstDecl(name: String, sort: T) = KConstDecl(name, sort)
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
    fun <D : KSort, R : KSort> mkArraySelectDecl(array: KArraySort<D, R>): KDecl<R> =
        KArraySelectDecl(array)

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
