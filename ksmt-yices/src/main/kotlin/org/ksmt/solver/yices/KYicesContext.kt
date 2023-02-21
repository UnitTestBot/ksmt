package org.ksmt.solver.yices

import com.sri.yices.Terms
import com.sri.yices.Types
import com.sri.yices.Yices
import org.ksmt.decl.KDecl
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.sort.KSort
import org.ksmt.utils.NativeLibraryLoader
import org.ksmt.utils.uncheckedCast
import java.math.BigInteger

open class KYicesContext : AutoCloseable {
    private var isClosed = false

    protected val expressions = HashMap<KExpr<*>, YicesTerm>()
    private val yicesExpressions = HashMap<YicesTerm, KExpr<*>>()
    protected val sorts = HashMap<KSort, YicesSort>()
    private val yicesSorts = HashMap<YicesSort, KSort>()
    protected val decls = HashMap<KDecl<*>, YicesTerm>()
    private val yicesDecls = HashMap<YicesTerm, KDecl<*>>()
    private val transformed = HashMap<KExpr<*>, KExpr<*>>()

    private val yicesTypes = HashSet<YicesSort>()
    private val yicesTerms = HashSet<YicesTerm>()

    val isActive: Boolean
        get() = !isClosed

    fun findInternalizedExpr(expr: KExpr<*>): YicesTerm? = expressions[expr]

    fun findConvertedExpr(expr: YicesTerm): KExpr<*>? = yicesExpressions[expr]

    fun findConvertedDecl(decl: YicesTerm): KDecl<*>? = yicesDecls[decl]

    fun <K : KExpr<*>> substituteDecls(expr: K, transform: (K) -> K): K =
        transformed.getOrPut(expr) { transform(expr) }.uncheckedCast()

    open fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> YicesTerm): YicesTerm =
        expressions.getOrPut(expr) {
            internalizer(expr).also {
                if (expr is KInterpretedValue<*> || expr is KConst<*>)
                    yicesExpressions[it] = expr
            }
        }


    open fun internalizeSort(sort: KSort, internalizer: (KSort) -> YicesSort): YicesSort =
        internalize(sorts, yicesSorts, sort, internalizer)

    open fun internalizeDecl(decl: KDecl<*>, internalizer: (KDecl<*>) -> YicesTerm): YicesTerm =
        internalize(decls, yicesDecls, decl, internalizer)

    fun convertExpr(expr: YicesTerm, converter: (YicesTerm) -> KExpr<*>): KExpr<*> =
        convert(expressions, yicesExpressions, expr, converter)

    fun convertSort(sort: YicesSort, converter: (YicesSort) -> KSort): KSort =
        convert(sorts, yicesSorts, sort, converter)

    fun convertDecl(decl: YicesTerm, converter: (YicesTerm) -> KDecl<*>): KDecl<*> =
        convert(decls, yicesDecls, decl, converter)

    private inline fun <K, V> internalize(
        cache: MutableMap<K, V>,
        reverseCache: MutableMap<V, K>,
        key: K,
        internalizer: (K) -> V
    ): V = cache.getOrPut(key) {
        internalizer(key).also { reverseCache[it] = key }
    }

    private inline fun <K, V> convert(
        cache: MutableMap<K, V>,
        reverseCache: MutableMap<V, K>,
        key: V,
        converter: (V) -> K
    ): K {
        val current = reverseCache[key]

        if (current != null) return current

        val converted = converter(key)
        cache.getOrPut(converted) { key }
        reverseCache[key] = converted

        return converted
    }

    val bool = Types.BOOL
    val int = Types.INT
    val real = Types.REAL

    private inline fun mkType(mk: () -> YicesSort): YicesSort {
        val type = mk()

        if (yicesTypes.add(type))
            Yices.yicesIncrefType(type)

        return type
    }

    fun bvType(sizeBits: UInt) = mkType { Types.bvType(sizeBits.toInt()) }
    fun functionType(domain: YicesSort, range: YicesSort) = mkType { Types.functionType(domain, range) }
    fun functionType(args: List<YicesSort>) = mkType { Types.functionType(args) }
    fun newUninterpretedType(name: String) = mkType { Types.newUninterpretedType(name) }

    val nullTerm = Terms.NULL_TERM
    val zero = mkTerm { Terms.intConst(0L) }
    val one = mkTerm { Terms.intConst(1L) }
    val minusOne = mkTerm { Terms.intConst(-1L) }

    private inline fun mkTerm(mk: () -> YicesTerm): YicesTerm {
        val term = mk()

        if (yicesTerms.add(term))
            Yices.yicesIncrefTerm(term)

        return term
    }

    fun newUninterpretedTerm(name: String, type: YicesSort) = mkTerm {
        Terms.newUninterpretedTerm(name, type)
    }

    fun newVariable(type: YicesSort) = mkTerm { Terms.newVariable(type) }
    fun newVariable(name: String, type: YicesSort) = mkTerm { Terms.newVariable(name, type) }
    fun funApplication(func: YicesTerm, index: YicesTerm) = mkTerm { Terms.funApplication(func, index) }
    fun funApplication(func: YicesTerm, args: List<YicesTerm>) = mkTerm { Terms.funApplication(func, args) }
    fun and(args: List<YicesTerm>) = mkTerm { Terms.and(args) }
    fun or(args: List<YicesTerm>) = mkTerm { Terms.or(args) }
    fun not(term: YicesTerm) = mkTerm { Terms.not(term) }
    fun implies(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.implies(arg0, arg1) }
    fun xor(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.xor(arg0, arg1) }
    fun mkTrue() = mkTerm(Terms::mkTrue)
    fun mkFalse() = mkTerm(Terms::mkFalse)
    fun eq(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.eq(arg0, arg1) }
    fun distinct(args: List<YicesTerm>) = mkTerm { Terms.distinct(args) }
    fun ifThenElse(condition: YicesTerm, trueBranch: YicesTerm, falseBranch: YicesTerm) = mkTerm {
        Terms.ifThenElse(condition, trueBranch, falseBranch)
    }

    fun bvConst(sizeBits: UInt, value: Long) = mkTerm { Terms.bvConst(sizeBits.toInt(), value) }
    fun parseBvBin(value: String) = mkTerm { Terms.parseBvBin(value) }
    fun bvNot(arg: YicesTerm) = mkTerm { Terms.bvNot(arg) }
    fun bvRedAnd(arg: YicesTerm) = mkTerm { Terms.bvRedAnd(arg) }
    fun bvRedOr(arg: YicesTerm) = mkTerm { Terms.bvRedOr(arg) }
    fun bvAnd(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvAnd(arg0, arg1) }
    fun bvOr(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvOr(arg0, arg1) }
    fun bvXor(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvXor(arg0, arg1) }
    fun bvNand(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvNand(arg0, arg1) }
    fun bvNor(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvNor(arg0, arg1) }
    fun bvXNor(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvXNor(arg0, arg1) }
    fun bvNeg(arg: YicesTerm) = mkTerm { Terms.bvNeg(arg) }
    fun bvAdd(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvAdd(arg0, arg1) }
    fun bvSub(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSub(arg0, arg1) }
    fun bvMul(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvMul(arg0, arg1) }
    fun bvDiv(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvDiv(arg0, arg1) }
    fun bvSDiv(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSDiv(arg0, arg1) }
    fun bvRem(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvRem(arg0, arg1) }
    fun bvSRem(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSRem(arg0, arg1) }
    fun bvSMod(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSMod(arg0, arg1) }
    fun bvLt(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvLt(arg0, arg1) }
    fun bvSLt(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSLt(arg0, arg1) }
    fun bvLe(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvLe(arg0, arg1) }
    fun bvSLe(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSLe(arg0, arg1) }
    fun bvGe(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvGe(arg0, arg1) }
    fun bvSGe(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSGe(arg0, arg1) }
    fun bvGt(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvGt(arg0, arg1) }
    fun bvSGt(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvSGt(arg0, arg1) }
    fun bvConcat(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.bvConcat(arg0, arg1) }
    fun bvExtract(arg: YicesTerm, low: Int, high: Int) = mkTerm { Terms.bvExtract(arg, low, high) }
    fun bvExtractBit(arg: YicesTerm, index: Int) = mkTerm { Terms.bvExtractBit(arg, index) }
    fun bvSignExtend(arg: YicesTerm, extensionSize: Int) = mkTerm {
        Terms.bvSignExtend(arg, extensionSize)
    }

    fun bvZeroExtend(arg: YicesTerm, extensionSize: Int) = mkTerm {
        Terms.bvZeroExtend(arg, extensionSize)
    }

    fun bvRepeat(arg: YicesTerm, repeatNumber: Int) = mkTerm { Terms.bvRepeat(arg, repeatNumber) }
    fun bvShl(arg: YicesTerm, shift: Int) = mkTerm { Terms.bvShl(arg, shift) }
    fun bvLshr(arg: YicesTerm, shift: Int) = mkTerm { Terms.bvLshr(arg, shift) }
    fun bvAshr(arg: YicesTerm, shift: Int) = mkTerm { Terms.bvAshr(arg, shift) }
    fun bvRotateLeft(arg: YicesTerm, rotationNumber: Int) = mkTerm {
        Terms.bvRotateLeft(arg, rotationNumber)
    }

    fun bvRotateRight(arg: YicesTerm, rotationNumber: Int) = mkTerm {
        Terms.bvRotateRight(arg, rotationNumber)
    }

    fun functionUpdate1(func: YicesTerm, arg: YicesTerm, value: YicesTerm) = mkTerm {
        Terms.functionUpdate1(func, arg, value)
    }

    fun lambda(bounds: List<YicesTerm>, body: YicesTerm) = mkTerm { Terms.lambda(bounds, body) }
    fun add(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.add(arg0, arg1) }
    fun add(args: List<YicesTerm>) = mkTerm { Terms.add(args) }
    fun mul(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.mul(arg0, arg1) }
    fun mul(args: List<YicesTerm>) = mkTerm { Terms.mul(args) }
    fun sub(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.sub(arg0, arg1) }
    fun neg(arg: YicesTerm) = mkTerm { Terms.neg(arg) }
    fun div(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.div(arg0, arg1) }
    fun power(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.power(arg0, arg1) }
    fun arithLt(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.arithLt(arg0, arg1) }
    fun arithLeq(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.arithLeq(arg0, arg1) }
    fun arithLeq0(arg: YicesTerm) = mkTerm { Terms.arithLeq0(arg) }
    fun arithGt(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.arithGt(arg0, arg1) }
    fun arithGeq(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.arithGeq(arg0, arg1) }
    fun idiv(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.idiv(arg0, arg1) }
    fun imod(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.imod(arg0, arg1) }
    fun intConst(value: Long) = mkTerm { Terms.intConst(value) }
    fun intConst(value: BigInteger) = mkTerm { Terms.intConst(value) }
    fun floor(arg: YicesTerm) = mkTerm { Terms.floor(arg) }
    fun isInt(arg: YicesTerm) = mkTerm { Terms.isInt(arg) }
    fun exists(bounds: List<YicesTerm>, body: YicesTerm) = mkTerm { Terms.exists(bounds, body) }
    fun forall(bounds: List<YicesTerm>, body: YicesTerm) = mkTerm { Terms.forall(bounds, body) }

    override fun close() {
        if (isClosed)
            return

        yicesTerms.forEach { Yices.yicesDecrefTerm(it) }
        yicesTypes.forEach { Yices.yicesDecrefType(it) }
        Yices.yicesGarbageCollect()

        isClosed = true
    }

    companion object {
        init {
            if (!Yices.isReady()) {
                NativeLibraryLoader.load { os ->
                    when (os) {
                        NativeLibraryLoader.OS.LINUX -> listOf("libgmp-10", "libyices", "libyices2java")
                        NativeLibraryLoader.OS.WINDOWS -> listOf("libgmp-10", "libyices", "libyices2java")
                        NativeLibraryLoader.OS.MACOS -> TODO("Mac os platform is not supported")
                    }
                }
                Yices.init()
                Yices.setReadyFlag(true)
            }
        }
    }
}
