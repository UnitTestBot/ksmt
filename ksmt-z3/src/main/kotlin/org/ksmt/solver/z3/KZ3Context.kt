package org.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.decRefUnsafe
import com.microsoft.z3.incRefUnsafe
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

@Suppress("TooManyFunctions")
class KZ3Context(private val ctx: Context) : AutoCloseable {
    constructor() : this(Context())

    private var isClosed = false
    private val expressions = HashMap<KExpr<*>, Long>()
    private val z3Expressions = HashMap<Long, KExpr<*>>()
    private val sorts = HashMap<KSort, Long>()
    private val z3Sorts = HashMap<Long, KSort>()
    private val decls = HashMap<KDecl<*>, Long>()
    private val z3Decls = HashMap<Long, KDecl<*>>()
    private val tmpNativeObjects = HashSet<Long>()

    val nCtx: Long
        get() = ctx.nCtx()

    val nativeContext: Context
        get() = ctx

    val isActive: Boolean
        get() = !isClosed

    // expr
    fun findInternalizedExpr(expr: KExpr<*>): Long? = expressions[expr]

    fun findConvertedExpr(expr: Long): KExpr<*>? = z3Expressions[expr]

    fun saveInternalizedExpr(expr: KExpr<*>, internalized: Long): Long =
        internalizeAst(expressions, z3Expressions, expr) { internalized }

    fun saveConvertedExpr(expr: Long, converted: KExpr<*>): KExpr<*> =
        convertAst(expressions, z3Expressions, expr) { converted }

    // sort
    fun findInternalizedSort(sort: KSort): Long? = sorts[sort]

    fun findConvertedSort(sort: Long): KSort? = z3Sorts[sort]

    fun saveInternalizedSort(sort: KSort, internalized: Long): Long =
        internalizeAst(sorts, z3Sorts, sort) { internalized }

    fun saveConvertedSort(sort: Long, converted: KSort): KSort =
        convertAst(sorts, z3Sorts, sort) { converted }

    inline fun internalizeSort(sort: KSort, internalizer: () -> Long): Long =
        findOrSave(sort, internalizer, ::findInternalizedSort, ::saveInternalizedSort)

    inline fun convertSort(sort: Long, converter: () -> KSort): KSort =
        findOrSave(sort, converter, ::findConvertedSort, ::saveConvertedSort)

    // decl
    fun findInternalizedDecl(decl: KDecl<*>): Long? = decls[decl]

    fun findConvertedDecl(decl: Long): KDecl<*>? = z3Decls[decl]

    fun saveInternalizedDecl(decl: KDecl<*>, internalized: Long): Long =
        internalizeAst(decls, z3Decls, decl) { internalized }

    fun saveConvertedDecl(decl: Long, converted: KDecl<*>): KDecl<*> =
        convertAst(decls, z3Decls, decl) { converted }

    inline fun internalizeDecl(decl: KDecl<*>, internalizer: () -> Long): Long =
        findOrSave(decl, internalizer, ::findInternalizedDecl, ::saveInternalizedDecl)

    inline fun convertDecl(decl: Long, converter: () -> KDecl<*>): KDecl<*> =
        findOrSave(decl, converter, ::findConvertedDecl, ::saveConvertedDecl)

    /**
     * Keep reference for a native object, which has no mapped KSMT expression.
     * */
    fun temporaryAst(ast: Long): Long {
        if (tmpNativeObjects.add(ast)) {
            incRefUnsafe(nCtx, ast)
        }
        return ast
    }

    /**
     * Release native object, obtained with [temporaryAst]
     * */
    fun releaseTemporaryAst(ast: Long) {
        if (tmpNativeObjects.remove(ast)) {
            decRefUnsafe(nCtx, ast)
        }
    }

    inline fun <K, V> findOrSave(
        key: K,
        computeValue: () -> V,
        find: (K) -> V?,
        save: (K, V) -> V
    ): V {
        val value = find(key)
        if (value != null) return value
        return save(key, computeValue())
    }

    private inline fun <K> internalizeAst(
        cache: MutableMap<K, Long>,
        reverseCache: MutableMap<Long, K>,
        key: K,
        internalizer: () -> Long
    ): Long = cache.getOrPut(key) {
        internalizer().also {
            reverseCache[it] = key
            incRefUnsafe(nCtx, it)
        }
    }

    private inline fun <K> convertAst(
        cache: MutableMap<K, Long>,
        reverseCache: MutableMap<Long, K>,
        key: Long,
        converter: () -> K
    ): K {
        val current = reverseCache[key]

        if (current != null) return current

        val converted = converter()
        cache.getOrPut(converted) { key }
        reverseCache[key] = converted
        incRefUnsafe(nCtx, key)

        return converted
    }

    override fun close() {
        isClosed = true

        z3Expressions.keys.forEach { decRefUnsafe(nCtx, it) }
        expressions.clear()
        z3Expressions.clear()

        tmpNativeObjects.forEach { decRefUnsafe(nCtx, it) }
        tmpNativeObjects.clear()

        z3Sorts.keys.forEach { decRefUnsafe(nCtx, it) }
        sorts.clear()
        z3Sorts.clear()

        z3Decls.keys.forEach { decRefUnsafe(nCtx, it) }
        decls.clear()
        z3Decls.clear()

        ctx.close()
    }
}
