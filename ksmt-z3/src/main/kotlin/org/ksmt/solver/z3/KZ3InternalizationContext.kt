package org.ksmt.solver.z3

import com.microsoft.z3.Expr
import com.microsoft.z3.FuncDecl
import com.microsoft.z3.Sort
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

@Suppress("TooManyFunctions")
class KZ3InternalizationContext : AutoCloseable {
    private var isClosed = false
    private val expressions = HashMap<KExpr<*>, Expr<*>>()
    private val z3Expressions = HashMap<Expr<*>, KExpr<*>>()
    private val sorts = HashMap<KSort, Sort>()
    private val z3Sorts = HashMap<Sort, KSort>()
    private val decls = HashMap<KDecl<*>, FuncDecl<*>>()
    private val z3Decls = HashMap<FuncDecl<*>, KDecl<*>>()

    val isActive: Boolean
        get() = !isClosed

    fun findInternalizedExpr(expr: KExpr<*>): Expr<*>? = expressions[expr]

    fun findConvertedExpr(expr: Expr<*>): KExpr<*>? = z3Expressions[expr]

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

    override fun close() {
        isClosed = true
        sorts.clear()
        decls.clear()
        expressions.clear()
    }
}
