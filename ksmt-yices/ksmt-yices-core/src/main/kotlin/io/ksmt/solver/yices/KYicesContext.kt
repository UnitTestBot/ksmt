package io.ksmt.solver.yices

import com.sri.yices.Terms
import com.sri.yices.Types
import com.sri.yices.Yices
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap
import it.unimi.dsi.fastutil.ints.IntOpenHashSet
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap
import io.ksmt.decl.KDecl
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.solver.KSolverUnsupportedFeatureException
import io.ksmt.solver.util.KExprIntInternalizerBase.Companion.NOT_INTERNALIZED
import io.ksmt.solver.yices.TermUtils.addTerm
import io.ksmt.solver.yices.TermUtils.andTerm
import io.ksmt.solver.yices.TermUtils.distinctTerm
import io.ksmt.solver.yices.TermUtils.funApplicationTerm
import io.ksmt.solver.yices.TermUtils.mulTerm
import io.ksmt.solver.yices.TermUtils.orTerm
import io.ksmt.sort.KSort
import io.ksmt.utils.library.NativeLibraryLoaderUtils
import java.math.BigInteger
import java.util.concurrent.atomic.AtomicInteger

open class KYicesContext : AutoCloseable {
    private var isClosed = false

    private val expressions = mkTermCache<KExpr<*>>()
    private val yicesExpressions = mkTermReverseCache<KExpr<*>>()

    private val sorts = mkSortCache<KSort>()
    private val yicesSorts = mkSortReverseCache<KSort>()

    private val decls = mkTermCache<KDecl<*>>()
    private val yicesDecls = mkTermReverseCache<KDecl<*>>()

    private val vars = mkTermCache<KDecl<*>>()
    private val yicesVars = mkTermReverseCache<KDecl<*>>()

    private val yicesTypes = mkSortSet()
    private val yicesTerms = mkTermSet()

    val isActive: Boolean
        get() = !isClosed

    fun findInternalizedExpr(expr: KExpr<*>): YicesTerm = expressions.getInt(expr)
    fun saveInternalizedExpr(expr: KExpr<*>, internalized: YicesTerm) {
        if (expressions.putIfAbsent(expr, internalized) == NOT_INTERNALIZED) {
            if (expr is KInterpretedValue<*> || expr is KConst<*>) {
                yicesExpressions.put(internalized, expr)
            }
        }
    }

    fun findInternalizedSort(sort: KSort): YicesSort = sorts.getInt(sort)
    fun saveInternalizedSort(sort: KSort, internalized: YicesSort) {
        saveWithReverseCache(sorts, yicesSorts, sort, internalized)
    }

    fun findInternalizedDecl(decl: KDecl<*>): YicesTerm = decls.getInt(decl)
    fun saveInternalizedDecl(decl: KDecl<*>, internalized: YicesTerm) {
        saveWithReverseCache(decls, yicesDecls, decl, internalized)
    }

    fun findInternalizedVar(decl: KDecl<*>): YicesTerm = vars.getInt(decl)
    fun saveInternalizedVar(decl: KDecl<*>, internalized: YicesTerm) {
        saveWithReverseCache(vars, yicesVars, decl, internalized)
    }

    fun findConvertedExpr(expr: YicesTerm): KExpr<*>? = yicesExpressions[expr]
    fun saveConvertedExpr(expr: YicesTerm, converted: KExpr<*>) {
        saveWithReverseCache(yicesExpressions, expressions, expr, converted)
    }

    fun findConvertedSort(sort: YicesSort): KSort? = yicesSorts[sort]
    fun saveConvertedSort(sort: YicesSort, converted: KSort) {
        saveWithReverseCache(yicesSorts, sorts, sort, converted)
    }

    fun findConvertedDecl(decl: YicesTerm): KDecl<*>? = yicesDecls[decl]
    fun saveConvertedDecl(decl: YicesTerm, converted: KDecl<*>) {
        saveWithReverseCache(yicesDecls, decls, decl, converted)
    }

    fun findConvertedVar(variable: YicesTerm): KDecl<*>? = yicesVars[variable]
    fun saveConvertedVar(variable: YicesTerm, converted: KDecl<*>) {
        saveWithReverseCache(yicesVars, vars, variable, converted)
    }

    inline fun internalizeSort(sort: KSort, internalizer: (KSort) -> YicesSort): YicesSort =
        findOrSave(::findInternalizedSort, ::saveInternalizedSort, sort) { internalizer(sort) }

    inline fun internalizeDecl(decl: KDecl<*>, internalizer: (KDecl<*>) -> YicesTerm): YicesTerm =
        findOrSave(::findInternalizedDecl, ::saveInternalizedDecl, decl) { internalizer(decl) }

    inline fun internalizeVar(decl: KDecl<*>, internalizer: (KDecl<*>) -> YicesTerm): YicesTerm =
        findOrSave(::findInternalizedVar, ::saveInternalizedVar, decl) { internalizer(decl) }

    inline fun convertSort(sort: YicesSort, converter: (YicesSort) -> KSort): KSort =
        findOrSave(::findConvertedSort, ::saveConvertedSort, sort) { converter(sort) }

    inline fun convertDecl(decl: YicesTerm, converter: (YicesTerm) -> KDecl<*>): KDecl<*> =
        findOrSave(::findConvertedDecl, ::saveConvertedDecl, decl) { converter(decl) }

    inline fun convertVar(variable: YicesTerm, converter: (YicesTerm) -> KDecl<*>): KDecl<*> =
        findOrSave(::findConvertedVar, ::saveConvertedVar, variable) { converter(variable) }

    private fun <V> saveWithReverseCache(
        cache: Int2ObjectOpenHashMap<V>,
        reverseCache: Object2IntOpenHashMap<V>,
        key: Int,
        value: V
    ) {
        if (cache.putIfAbsent(key, value) == null) {
            reverseCache.putIfAbsent(value, key)
        }
    }

    private fun <K> saveWithReverseCache(
        cache: Object2IntOpenHashMap<K>,
        reverseCache: Int2ObjectOpenHashMap<K>,
        key: K,
        value: Int
    ) {
        if (cache.putIfAbsent(key, value) == NOT_INTERNALIZED) {
            reverseCache.putIfAbsent(value, key)
        }
    }

    inline fun <V> findOrSave(
        find: (Int) -> V?,
        save: (Int, V) -> Unit,
        key: Int,
        computeValue: () -> V
    ): V {
        val currentValue = find(key)
        if (currentValue != null) return currentValue

        val value = computeValue()
        save(key, value)
        return value
    }

    inline fun <K> findOrSave(
        find: (K) -> Int,
        save: (K, Int) -> Unit,
        key: K,
        computeValue: () -> Int
    ): Int {
        val currentValue = find(key)
        if (currentValue != NOT_INTERNALIZED) return currentValue

        val value = computeValue()
        save(key, value)
        return value
    }

    val bool = Types.BOOL
    val int = Types.INT
    val real = Types.REAL

    private inline fun mkType(mk: () -> YicesSort): YicesSort = withGcGuard {
        val type = mk()

        if (yicesTypes.add(type)) {
            Yices.yicesIncrefType(type)
        }

        return type
    }

    fun bvType(sizeBits: UInt) = mkType { Types.bvType(sizeBits.toInt()) }
    fun functionType(domain: YicesSort, range: YicesSort) = mkType { Types.functionType(domain, range) }
    fun functionType(domain: YicesSortArray, range: YicesSort) = mkType { Types.functionType(domain, range) }
    fun newUninterpretedType(name: String) = mkType { Types.newUninterpretedType(name) }

    val zero = mkTerm { Terms.intConst(0L) }
    val one = mkTerm { Terms.intConst(1L) }
    val minusOne = mkTerm { Terms.intConst(-1L) }

    private inline fun mkTerm(mk: () -> YicesTerm): YicesTerm = withGcGuard {
        val term = mk()

        if (yicesTerms.add(term)) {
            Yices.yicesIncrefTerm(term)
        }

        return term
    }

    fun newUninterpretedTerm(name: String, type: YicesSort) = mkTerm {
        Terms.newUninterpretedTerm(name, type)
    }

    fun newVariable(type: YicesSort) = mkTerm { Terms.newVariable(type) }
    fun newVariable(name: String, type: YicesSort) = mkTerm { Terms.newVariable(name, type) }

    fun and(args: YicesTermArray) = mkTerm { andTerm(args) }
    fun or(args: YicesTermArray) = mkTerm { orTerm(args) }
    fun not(term: YicesTerm) = mkTerm { Terms.not(term) }
    fun implies(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.implies(arg0, arg1) }
    fun xor(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.xor(arg0, arg1) }
    fun mkTrue() = mkTerm(Terms::mkTrue)
    fun mkFalse() = mkTerm(Terms::mkFalse)
    fun eq(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.eq(arg0, arg1) }
    fun distinct(args: YicesTermArray) = mkTerm { distinctTerm(args) }
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

    fun funApplication(func: YicesTerm, index: YicesTerm) = mkTerm { Terms.funApplication(func, index) }
    fun funApplication(func: YicesTerm, args: YicesTermArray) = mkTerm { funApplicationTerm(func, args) }

    fun functionUpdate(func: YicesTerm, args: YicesTermArray, value: YicesTerm) = mkTerm {
        Terms.functionUpdate(func, args, value)
    }

    fun lambda(bounds: YicesTermArray, body: YicesTerm) = mkTerm { Terms.lambda(bounds, body) }

    fun add(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.add(arg0, arg1) }
    fun add(args: YicesTermArray) = mkTerm { addTerm(args) }
    fun mul(arg0: YicesTerm, arg1: YicesTerm) = mkTerm { Terms.mul(arg0, arg1) }
    fun mul(args: YicesTermArray) = mkTerm { mulTerm(args) }
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

    fun exists(bounds: YicesTermArray, body: YicesTerm) = mkTerm { Terms.exists(bounds, body) }
    fun forall(bounds: YicesTermArray, body: YicesTerm) = mkTerm { Terms.forall(bounds, body) }

    fun uninterpretedSortConst(sort: YicesSort, idx: Int) = mkTerm { Terms.mkConst(sort, idx) }

    private var maxValueIndex = 0

    /**
     * Yices can produce different values with the same index.
     * This situation happens if we create a value with index I (e.g. 2)
     * and Yices generates a value in the model with index I.
     * To overcome this, we shift our indices by some very huge value.
     * Since Yices generates values with a small indices (e.g. 0, 1, 2, ...)
     * this trick solves the issue.
     * */
    fun uninterpretedSortValueIndex(idx: Int): Int {
        if (idx !in UNINTERPRETED_SORT_VALUE_INDEX_RANGE) {
            throw KSolverUnsupportedFeatureException(
                "Yices solver requires value index to be in range: $UNINTERPRETED_SORT_VALUE_INDEX_RANGE"
            )
        }

        maxValueIndex = maxOf(maxValueIndex, idx)
        return idx + UNINTERPRETED_SORT_VALUE_SHIFT
    }

    fun convertUninterpretedSortValueIndex(internalIndex: Int): Int {
        // User provided value index
        if (internalIndex >= UNINTERPRETED_SORT_MIN_SHIFTED_VALUE) {
            return internalIndex - UNINTERPRETED_SORT_VALUE_SHIFT
        }

        // Create a new index that doesn't overlap with any other indices
        return ++maxValueIndex
    }

    fun substitute(term: YicesTerm, substituteFrom: YicesTermArray, substituteTo: YicesTermArray): YicesTerm =
        mkTerm { Terms.subst(term, substituteFrom, substituteTo) }

    override fun close() {
        if (isClosed) return
        isClosed = true

        yicesTerms.forEach { Yices.yicesDecrefTerm(it) }
        yicesTypes.forEach { Yices.yicesDecrefType(it) }

        performGc()
    }

    companion object {
        init {
            if (!Yices.isReady()) {
                NativeLibraryLoaderUtils.load<KYicesNativeLibraryLoader>()
                Yices.init()
                Yices.setReadyFlag(true)
            }
        }

        private const val UNINTERPRETED_SORT_VALUE_SHIFT = 1 shl 30
        private const val UNINTERPRETED_SORT_MAX_ALLOWED_VALUE = UNINTERPRETED_SORT_VALUE_SHIFT / 2
        private const val UNINTERPRETED_SORT_MIN_ALLOWED_VALUE = -UNINTERPRETED_SORT_MAX_ALLOWED_VALUE
        private const val UNINTERPRETED_SORT_MIN_SHIFTED_VALUE = UNINTERPRETED_SORT_MAX_ALLOWED_VALUE

        internal val UNINTERPRETED_SORT_VALUE_INDEX_RANGE =
            UNINTERPRETED_SORT_MIN_ALLOWED_VALUE..UNINTERPRETED_SORT_MAX_ALLOWED_VALUE

        internal fun <K> mkTermCache() = Object2IntOpenHashMap<K>().apply {
            defaultReturnValue(NOT_INTERNALIZED)
        }

        internal fun <V> mkTermReverseCache() = Int2ObjectOpenHashMap<V>()

        internal fun <K> mkSortCache() = Object2IntOpenHashMap<K>().apply {
            defaultReturnValue(NOT_INTERNALIZED)
        }

        internal fun <V> mkSortReverseCache() = Int2ObjectOpenHashMap<V>()

        internal fun mkTermSet() = IntOpenHashSet()

        internal fun mkSortSet() = IntOpenHashSet()

        private const val FREE = 0
        private const val ON_GC = -1000000

        @JvmStatic
        private val gcGuard = AtomicInteger(FREE)

        /**
         * Since Yices manages terms globally we must ensure that
         * there are no Yices GC operations between
         * the initiation of the term creation process
         * and the execution of `incRef` on the newly created term.
         * Otherwise, there is a scenario when the term is deleted before the `incRef`
         * and therefore remains invalid.
         *
         * According to the [gcGuard] possible values we have the following situations:
         * 1. [gcGuard] == [FREE] -- no currently performing operations
         * 2. [gcGuard] > [FREE] -- some term creation operations are performed
         * 3. [gcGuard] < [FREE] -- GC is performed
         *
         * All term operations are performed according to the following rules:
         * 1. We can create term only if [gcGuard] >= [FREE]
         * (no operations or other term creation operation. No GC operations).
         * 2. If we are on GC and we want to create a term we spin wait until
         * [gcGuard] >= [FREE].
         * 3. If we want to perform GC we spin wait until [gcGuard] == [FREE] (no operations).
         *
         * See also [performGc]
         * */
        private inline fun <T> withGcGuard(body: () -> T): T {
            // spin wait until [gcGuard] >= [FREE]
            while (true) {
                val status = gcGuard.getAndIncrement()
                if (status >= FREE) break
                gcGuard.getAndDecrement()
            }

            return try {
                body()
            } finally {
                gcGuard.getAndDecrement()
            }
        }

        private fun performGc() {
            // spin wait until [gcGuard] == [FREE]
            while (true) {
                if (gcGuard.compareAndSet(FREE, ON_GC)) {
                    break
                }
            }

            Yices.yicesGarbageCollect()

            gcGuard.getAndAdd(-ON_GC)
        }

    }
}
