package org.ksmt.solver.z3

import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.Sort
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort
import java.lang.ref.WeakReference
import java.util.WeakHashMap

@Suppress("TooManyFunctions")
class KZ3InternalizationContext : AutoCloseable {
    private var closed = false
    private val expressions = WeakHashMap<KExpr<*>, Expr<*>>()
    private val z3Expressions = WeakHashMap<Expr<*>, WeakReference<KExpr<*>>>()
    private val sorts = WeakHashMap<KSort, Sort>()
    private val z3Sorts = WeakHashMap<Sort, WeakReference<KSort>>()
    private val decls = WeakHashMap<KDecl<*>, FuncDecl<*>>()
    private val z3Decls = WeakHashMap<FuncDecl<*>, WeakReference<KDecl<*>>>()

    val isActive: Boolean
        get() = !closed

    fun findInternalizedExpr(expr: KExpr<*>): Expr<*>? = expressions[expr]

    fun findConvertedExpr(expr: Expr<*>): KExpr<*>? = z3Expressions[expr]?.get()

    fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> Expr<*>): Expr<*> =
        internalize(expressions, z3Expressions, expr, internalizer)

    fun internalizeSort(sort: KSort, internalizer: (KSort) -> Sort): Sort =
        internalize(sorts, z3Sorts, sort, internalizer)

    fun internalizeDecl(decl: KDecl<*>, internalizer: (KDecl<*>) -> FuncDecl<*>): FuncDecl<*> =
        internalize(decls, z3Decls, decl, internalizer)

    fun convertExpr(expr: Expr<*>, converter: (Expr<*>) -> KExpr<*>): KExpr<*> =
        convert(expressions, z3Expressions, expr, converter)

    fun convertSort(sort: Sort, converter: (Sort) -> KSort): KSort =
        convert(sorts, z3Sorts, sort, converter)

    fun convertDecl(decl: FuncDecl<*>, converter: (FuncDecl<*>) -> KDecl<*>): KDecl<*> =
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
}
