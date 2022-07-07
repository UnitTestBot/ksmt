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

    val bitwuzla = Native.bitwuzlaNew()

    val expressions = WeakHashMap<KExpr<*>, BitwuzlaTerm>()
    val bitwuzlaExpressions = WeakHashMap<BitwuzlaTerm, WeakReference<KExpr<*>>>()
    private val sorts = WeakHashMap<KSort, BitwuzlaSort>()
    private val bitwuzlaSorts = WeakHashMap<BitwuzlaSort, WeakReference<KSort>>()
    private val declSorts = WeakHashMap<KDecl<*>, BitwuzlaSort>()

    operator fun get(expr: KExpr<*>): BitwuzlaTerm? = expressions[expr]
    operator fun get(sort: KSort): BitwuzlaSort? = sorts[sort]

    fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> BitwuzlaTerm): BitwuzlaTerm =
        expressions.getOrPut(expr) {
            internalizer(expr) // don't reverse cache bitwuzla terms since it may be rewrited
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

    private var freshVarNumber = 0
    fun mkFreshVar(sort: BitwuzlaSort) = Native.bitwuzlaMkVar(bitwuzla, sort, "${freshVarNumber++}")

    val rootConstantScope: RootConstantScope = RootConstantScope()
    var currentConstantScope: ConstantScope = rootConstantScope

    fun declaredConstants(): Set<KDecl<*>> = rootConstantScope.constants.toSet()
    fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = currentConstantScope.mkConstant(decl, sort)

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
        Native.bitwuzlaAssert(bitwuzla, term)
    }

    fun check(): BitwuzlaResult {
        ensureActive()
        return Native.bitwuzlaCheckSat(bitwuzla)
    }

    override fun close() {
        closed = true
        sorts.clear()
        bitwuzlaSorts.clear()
        expressions.clear()
        bitwuzlaExpressions.clear()
        declSorts.clear()
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
        cache.getOrPut(converted) { key }
        reverseCache[key] = WeakReference(converted)
        return converted
    }

    sealed interface ConstantScope {
        fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm
    }

    inner class RootConstantScope : ConstantScope {
        val constants = hashSetOf<KDecl<*>>()
        private val constantCache = mkCache { decl: KDecl<*>, sort: BitwuzlaSort ->
            Native.bitwuzlaMkConst(bitwuzla, sort, decl.name).also {
                constants += decl
            }
        }

        override fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = constantCache.create(decl, sort)
    }

    inner class NestedConstantScope(val parent: ConstantScope) : ConstantScope {
        private val constants = hashMapOf<Pair<KDecl<*>, BitwuzlaSort>, BitwuzlaTerm>()
        fun mkFreshConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = constants.getOrPut(decl to sort) {
            Native.bitwuzlaMkConst(bitwuzla, sort, decl.name)
        }

        override fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm =
            constants[decl to sort] ?: parent.mkConstant(decl, sort)
    }

}
