package org.ksmt.solver.z3

import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.Sort
import java.lang.ref.WeakReference
import java.util.WeakHashMap
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

@Suppress("TooManyFunctions")
class KZ3InternalizationContext : AutoCloseable {
    private var closed = false
    private val expressions = WeakHashMap<KExpr<*>, Expr>()
    private val z3Expressions = WeakHashMap<Expr, WeakReference<KExpr<*>>>()
    private val sorts = WeakHashMap<KSort, Sort>()
    private val z3Sorts = WeakHashMap<Sort, WeakReference<KSort>>()
    private val decls = WeakHashMap<KDecl<*>, FuncDecl>()
    private val z3Decls = WeakHashMap<FuncDecl, WeakReference<KDecl<*>>>()

    val isActive: Boolean
        get() = !closed

    operator fun get(expr: KExpr<*>): Internalized<Expr> = expressions.getInternalized(expr)
    operator fun get(sort: KSort): Internalized<Sort> = sorts.getInternalized(sort)
    operator fun get(decl: KDecl<*>): Internalized<FuncDecl> = decls.getInternalized(decl)

    fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> Expr): Expr =
        internalize(expressions, z3Expressions, expr, internalizer)

    fun internalizeSort(sort: KSort, internalizer: (KSort) -> Sort): Sort =
        internalize(sorts, z3Sorts, sort, internalizer)

    fun internalizeDecl(decl: KDecl<*>, internalizer: (KDecl<*>) -> FuncDecl): FuncDecl =
        internalize(decls, z3Decls, decl, internalizer)

    fun convertExpr(expr: Expr, converter: (Expr) -> KExpr<*>): KExpr<*> =
        convert(expressions, z3Expressions, expr, converter)

    fun convertSort(sort: Sort, converter: (Sort) -> KSort): KSort =
        convert(sorts, z3Sorts, sort, converter)

    fun convertDecl(decl: FuncDecl, converter: (FuncDecl) -> KDecl<*>): KDecl<*> =
        convert(decls, z3Decls, decl, converter)

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

    override fun close() {
        closed = true
        sorts.clear()
        decls.clear()
        expressions.clear()
    }

    sealed interface Internalized<out T> {
        val isInternalized: Boolean
            get() = this is Value

        fun getOrError(): T = when (this) {
            is Value -> value
            NotInternalized -> error("not internalized")
        }

        class Value<T>(val value: T) : Internalized<T>
        object NotInternalized : Internalized<Nothing>
    }

    private fun <K, V> Map<K, V>.getInternalized(key: K): Internalized<V> =
        this[key]?.let { Internalized.Value(it) } ?: Internalized.NotInternalized
}
