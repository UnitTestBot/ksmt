package org.ksmt.solver.bitwuzla

import com.sun.jna.Pointer
import org.ksmt.cache.mkCache
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KSort
import java.lang.ref.WeakReference
import java.util.WeakHashMap
import kotlin.time.Duration
import kotlin.time.ExperimentalTime
import kotlin.time.TimeMark
import kotlin.time.TimeSource

open class KBitwuzlaContext : AutoCloseable {
    private var closed = false

    val bitwuzla = Native.bitwuzlaNew()

    val trueTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkTrue(bitwuzla) }
    val falseTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkFalse(bitwuzla) }
    val boolSort: BitwuzlaSort by lazy { Native.bitwuzlaMkBoolSort(bitwuzla) }

    private val expressions = WeakHashMap<KExpr<*>, BitwuzlaTerm>()
    private val bitwuzlaExpressions = WeakHashMap<BitwuzlaTerm, WeakReference<KExpr<*>>>()
    private val sorts = WeakHashMap<KSort, BitwuzlaSort>()
    private val bitwuzlaSorts = WeakHashMap<BitwuzlaSort, WeakReference<KSort>>()
    private val declSorts = WeakHashMap<KDecl<*>, BitwuzlaSort>()
    private val bitwuzlaConstants = HashMap<BitwuzlaTerm, KDecl<*>>()

    operator fun get(expr: KExpr<*>): BitwuzlaTerm? = expressions[expr]
    operator fun get(sort: KSort): BitwuzlaSort? = sorts[sort]

    fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> BitwuzlaTerm): BitwuzlaTerm =
        expressions.getOrPut(expr) {
            internalizer(expr) // don't reverse cache bitwuzla term since it may be rewrited
        }

    fun internalizeSort(sort: KSort, internalizer: (KSort) -> BitwuzlaSort): BitwuzlaSort =
        sorts.getOrPut(sort) {
            internalizer(sort).also {
                bitwuzlaSorts[it] = WeakReference(sort)
            }
        }

    fun internalizeDeclSort(decl: KDecl<*>, internalizer: (KDecl<*>) -> BitwuzlaSort): BitwuzlaSort =
        declSorts.getOrPut(decl) {
            internalizer(decl)
        }

    fun convertExpr(expr: BitwuzlaTerm, converter: (BitwuzlaTerm) -> KExpr<*>): KExpr<*> =
        convert(expressions, bitwuzlaExpressions, expr, converter)

    fun convertSort(sort: BitwuzlaSort, converter: (BitwuzlaSort) -> KSort): KSort =
        convert(sorts, bitwuzlaSorts, sort, converter)

    // constant is known only if it was previously internalized
    fun convertConstantIfKnown(term: BitwuzlaTerm): KDecl<*>? = bitwuzlaConstants[term]

    private val normalConstantScope: NormalConstantScope = NormalConstantScope()
    var currentConstantScope: ConstantScope = normalConstantScope

    fun declaredConstants(): Set<KDecl<*>> = normalConstantScope.constants.toSet()
    fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = currentConstantScope.mkConstant(decl, sort)

    /** Constant scope for quantifiers.
     *
     * Quantifier bound variables:
     * 1. may clash with constants from outer scope
     * 2. must be replaced with vars in quantifier body
     *
     * @see QuantifiedConstantScope
     * */
    inline fun <reified T> withConstantScope(body: QuantifiedConstantScope.() -> T): T {
        val oldScope = currentConstantScope
        val newScope = QuantifiedConstantScope(currentConstantScope)
        currentConstantScope = newScope
        val result = newScope.body()
        currentConstantScope = oldScope
        return result
    }

    @OptIn(ExperimentalTime::class)
    fun <T> withTimeout(timeout: Duration, body: () -> T): T {
        if (timeout == Duration.INFINITE) {
            Native.bitwuzlaResetTerminationCallback(bitwuzla)
            return body()
        }
        val currentTime = TimeSource.Monotonic.markNow()
        val finishTime = currentTime + timeout
        val timeoutTerminator = BitwuzlaTimeout(finishTime)
        try {
            Native.bitwuzlaSetTerminationCallback(bitwuzla, timeoutTerminator, null)
            return body()
        } finally {
            Native.bitwuzlaResetTerminationCallback(bitwuzla)
        }
    }

    inline fun <reified T> bitwuzlaTry(body: () -> T): T = try {
        ensureActive()
        body()
    } catch (ex: Native.BitwuzlaException) {
        throw KSolverException(ex)
    }

    override fun close() {
        closed = true
        sorts.clear()
        bitwuzlaSorts.clear()
        expressions.clear()
        bitwuzlaExpressions.clear()
        declSorts.clear()
        bitwuzlaConstants.clear()
        Native.bitwuzlaDelete(bitwuzla)
    }

    fun ensureActive() {
        check(!closed) { "context already closed" }
    }

    private inline fun <K, V> convert(
        cache: MutableMap<K, V>,
        reverseCache: MutableMap<V, WeakReference<K>>,
        key: V,
        converter: (V) -> K
    ): K {
        val current = reverseCache[key]?.get()

        if (current != null) return current

        val converted = converter(key)
        cache.putIfAbsent(converted, key)
        reverseCache[key] = WeakReference(converted)
        return converted
    }

    sealed interface ConstantScope {
        fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm
    }


    /** Produce normal constants for KSmt declarations.
     *
     *  Guarantee that if two declarations are equal
     *  then same Bitwuzla native pointer is returned.
     * */
    inner class NormalConstantScope : ConstantScope {
        val constants = hashSetOf<KDecl<*>>()
        private val constantCache = mkCache { decl: KDecl<*>, sort: BitwuzlaSort ->
            Native.bitwuzlaMkConst(bitwuzla, sort, decl.name).also {
                constants += decl
                bitwuzlaConstants[it] = decl
            }
        }

        override fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = constantCache.create(decl, sort)
    }

    /** Constant quantification.
     *
     * If constant declaration registered as quantified variable,
     * then the corresponding var is returned instead of constant.
     * */
    inner class QuantifiedConstantScope(private val parent: ConstantScope) : ConstantScope {
        private val constants = hashMapOf<Pair<KDecl<*>, BitwuzlaSort>, BitwuzlaTerm>()
        fun mkVar(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = constants.getOrPut(decl to sort) {
            Native.bitwuzlaMkVar(bitwuzla, sort, decl.name)
        }

        override fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm =
            constants[decl to sort] ?: parent.mkConstant(decl, sort)
    }

    @OptIn(ExperimentalTime::class)
    class BitwuzlaTimeout(private val finishTime: TimeMark) : Native.BitwuzlaTerminationCallback {
        override fun terminate(state: Pointer?): Int = if (finishTime.hasNotPassedNow()) 0 else 1
    }

}
