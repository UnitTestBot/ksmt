package org.ksmt.solver.bitwuzla

import org.ksmt.cache.mkCache
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaResult
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KSort
import java.lang.ref.WeakReference
import java.util.*

open class KBitwuzlaContext : AutoCloseable {
    private var closed = false

    val bitwuzla = Native.bitwuzla_new()

    private val expressions = WeakHashMap<KExpr<*>, BitwuzlaTerm>()
    private val bitwuzlaExpressions = WeakHashMap<BitwuzlaTerm, WeakReference<KExpr<*>>>()
    private val sorts = WeakHashMap<KSort, BitwuzlaSort>()
    private val bitwuzlaSorts = WeakHashMap<BitwuzlaSort, WeakReference<KSort>>()
    private val declSorts = WeakHashMap<KDecl<*>, BitwuzlaSort>()

    operator fun get(expr: KExpr<*>): BitwuzlaTerm? = expressions[expr]
    operator fun get(sort: KSort): BitwuzlaSort? = sorts[sort]

    fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> BitwuzlaTerm): BitwuzlaTerm =
        internalize(expressions, bitwuzlaExpressions, expr, internalizer)

    fun internalizeSort(sort: KSort, internalizer: (KSort) -> BitwuzlaSort): BitwuzlaSort =
        internalize(sorts, bitwuzlaSorts, sort, internalizer)

    fun internalizeDeclSort(decl: KDecl<*>, internalizer: (KDecl<*>) -> BitwuzlaSort): BitwuzlaSort =
        declSorts.getOrPut(decl) {
            internalizer(decl)
        }

    fun convertExpr(expr: BitwuzlaTerm, converter: (BitwuzlaTerm) -> KExpr<*>): KExpr<*> =
        convert(expressions, bitwuzlaExpressions, expr, converter)

    fun convertSort(sort: BitwuzlaSort, converter: (BitwuzlaSort) -> KSort): KSort =
        convert(sorts, bitwuzlaSorts, sort, converter)

    private var freshVarNumber = 0
    fun mkFreshVar(sort: BitwuzlaSort) = Native.bitwuzla_mk_var(bitwuzla, sort, "${freshVarNumber++}")

    val rootConstantScope: RootConstantScope = RootConstantScope()
    var currentConstantScope: ConstantScope = rootConstantScope

    fun declaredConstants(): Set<BitwuzlaTerm> = rootConstantScope.constants.toSet()
    fun mkConstant(name: String, sort: BitwuzlaSort): BitwuzlaTerm = currentConstantScope.mkConstant(name, sort)

    inline fun <reified T> withConstantScope(body: NestedConstantScope.() -> T): T {
        val oldScope = currentConstantScope
        val newScope = NestedConstantScope(currentConstantScope)
        currentConstantScope = newScope
        val result = newScope.body()
        currentConstantScope = oldScope
        return result
    }

    fun assert(term: BitwuzlaTerm) {
        ensureActive()
        Native.bitwuzla_assert(bitwuzla, term)
    }

    fun check(): BitwuzlaResult {
        ensureActive()
        return Native.bitwuzla_check_sat_helper(bitwuzla)
    }

    override fun close() {
        closed = true
        sorts.clear()
//        decls.clear()
        expressions.clear()
        Native.bitwuzla_delete(bitwuzla)
    }

    fun ensureActive() {
        check(!closed) { "context already closed" }
    }

    private inline fun <K, V> internalize(
        cache: MutableMap<K, V>,
        reverseCache: MutableMap<V, WeakReference<K>>,
        key: K,
        internalizer: (K) -> V
    ): V = cache.getOrPut(key) {
        internalizer(key).also { reverseCache[it] = WeakReference(key) }
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
        cache.getOrPut(converted) { key }
        reverseCache[key] = WeakReference(converted)
        return converted
    }

    sealed interface ConstantScope {
        fun mkConstant(name: String, sort: BitwuzlaSort): BitwuzlaTerm
    }

    inner class RootConstantScope : ConstantScope {
        val constants = hashSetOf<BitwuzlaTerm>()
        private val constantCache = mkCache { name: String, sort: BitwuzlaSort ->
            Native.bitwuzla_mk_const(bitwuzla, sort, name).also {
                constants += it
            }
        }

        override fun mkConstant(name: String, sort: BitwuzlaSort): BitwuzlaTerm = constantCache.create(name, sort)
    }

    inner class NestedConstantScope(val parent: ConstantScope) : ConstantScope {
        private val constants = hashMapOf<Pair<String, BitwuzlaSort>, BitwuzlaTerm>()
        fun mkFreshConstant(name: String, sort: BitwuzlaSort): BitwuzlaTerm = constants.getOrPut(name to sort) {
            Native.bitwuzla_mk_const(bitwuzla, sort, name)
        }

        override fun mkConstant(name: String, sort: BitwuzlaSort): BitwuzlaTerm =
            constants[name to sort] ?: parent.mkConstant(name, sort)
    }

}
