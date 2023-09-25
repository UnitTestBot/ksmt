package io.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Solver
import com.microsoft.z3.decRefUnsafe
import com.microsoft.z3.incRefUnsafe
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.util.KExprLongInternalizerBase.Companion.NOT_INTERNALIZED
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import it.unimi.dsi.fastutil.longs.LongOpenHashSet
import it.unimi.dsi.fastutil.longs.LongSet
import it.unimi.dsi.fastutil.objects.Object2LongOpenHashMap

@Suppress("TooManyFunctions")
class KZ3Context(
    ksmtCtx: KContext,
    private val ctx: Context
) : AutoCloseable {
    constructor(ksmtCtx: KContext) : this(ksmtCtx, Context())

    private var isClosed = false

    private val expressions = Object2LongOpenHashMap<KExpr<*>>().apply {
        defaultReturnValue(NOT_INTERNALIZED)
    }

    private val sorts = Object2LongOpenHashMap<KSort>().apply {
        defaultReturnValue(NOT_INTERNALIZED)
    }

    private val decls = Object2LongOpenHashMap<KDecl<*>>().apply {
        defaultReturnValue(NOT_INTERNALIZED)
    }

    private val z3Expressions = Long2ObjectOpenHashMap<KExpr<*>>()
    private val z3Sorts = Long2ObjectOpenHashMap<KSort>()
    private val z3Decls = Long2ObjectOpenHashMap<KDecl<*>>()
    private val tmpNativeObjects = LongOpenHashSet()
    private val converterNativeObjects = LongOpenHashSet()

    val uninterpretedValuesTracker = ExpressionUninterpretedValuesTracker(ksmtCtx, this)

    @JvmField
    val nCtx: Long = ctx.nCtx()

    val nativeContext: Context
        get() = ctx

    val isActive: Boolean
        get() = !isClosed

    /**
     * Find internalized expr.
     * Returns [NOT_INTERNALIZED] if expression was not found.
     * */
    fun findInternalizedExpr(expr: KExpr<*>): Long {
        val result = expressions.getLong(expr)
        if (result == NOT_INTERNALIZED) return NOT_INTERNALIZED

        uninterpretedValuesTracker.expressionUse(expr)

        return result
    }

    fun saveInternalizedExpr(expr: KExpr<*>, internalized: Long) {
        uninterpretedValuesTracker.expressionSave(expr)

        saveAst(internalized, expr, expressions, z3Expressions)
    }

    fun findConvertedExpr(expr: Long): KExpr<*>? = z3Expressions[expr]

    fun saveConvertedExpr(expr: Long, converted: KExpr<*>) {
        saveConvertedAst(expr, converted, expressions, z3Expressions)
    }

    /**
     * Find internalized sort.
     * Returns [NOT_INTERNALIZED] if sort was not found.
     * */
    fun findInternalizedSort(sort: KSort): Long = sorts.getLong(sort)

    fun findConvertedSort(sort: Long): KSort? = z3Sorts[sort]

    fun saveInternalizedSort(sort: KSort, internalized: Long): Long =
        saveAst(internalized, sort, sorts, z3Sorts)

    fun saveConvertedSort(sort: Long, converted: KSort): KSort =
        saveConvertedAst(sort, converted, sorts, z3Sorts)

    inline fun internalizeSort(sort: KSort, internalizer: () -> Long): Long =
        findOrSave(sort, internalizer, ::findInternalizedSort, ::saveInternalizedSort)

    inline fun convertSort(sort: Long, converter: () -> KSort): KSort =
        findOrSave(sort, converter, ::findConvertedSort, ::saveConvertedSort)

    /**
     * Find internalized decl.
     * Returns [NOT_INTERNALIZED] if decl was not found.
     * */
    fun findInternalizedDecl(decl: KDecl<*>): Long = decls.getLong(decl)

    fun findConvertedDecl(decl: Long): KDecl<*>? = z3Decls[decl]

    fun saveInternalizedDecl(decl: KDecl<*>, internalized: Long): Long =
        saveAst(internalized, decl, decls, z3Decls)

    fun saveConvertedDecl(decl: Long, converted: KDecl<*>): KDecl<*> =
        saveConvertedAst(decl, converted, decls, z3Decls)

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

    /**
     * Save reference to the converter local object.
     * */
    fun saveConverterNativeObject(ast: Long): Long {
        if (converterNativeObjects.add(ast)) {
            incRefUnsafe(nCtx, ast)
        }
        return ast
    }

    private val uninterpretedSortValueInterpreter = hashMapOf<KUninterpretedSort, Long>()

    private val uninterpretedSortValueDecls = Long2ObjectOpenHashMap<KUninterpretedSortValue>()
    private val uninterpretedSortValueInterpreters = LongOpenHashSet()

    fun saveUninterpretedSortValueDecl(decl: Long, value: KUninterpretedSortValue): Long {
        if (uninterpretedSortValueDecls.putIfAbsent(decl, value) == null) {
            incRefUnsafe(nCtx, decl)
        }
        return decl
    }

    fun saveUninterpretedSortValueInterpreter(decl: Long): Long {
        if (uninterpretedSortValueInterpreters.add(decl)) {
            incRefUnsafe(nCtx, decl)
        }
        return decl
    }

    inline fun registerUninterpretedSortValue(
        value: KUninterpretedSortValue,
        uniqueValueDescriptorExpr: Long,
        uninterpretedValueExpr: Long,
        mkInterpreter: () -> Long
    ) {
        val interpreter = getUninterpretedSortValueInterpreter(value.sort)
        if (interpreter == null) {
            registerUninterpretedSortValueInterpreter(value.sort, mkInterpreter())
        }

        uninterpretedValuesTracker.registerUninterpretedSortValue(
            value, uniqueValueDescriptorExpr, uninterpretedValueExpr
        )
    }

    fun pushAssertionLevel() {
        uninterpretedValuesTracker.pushAssertionLevel()
    }

    fun popAssertionLevel() {
        uninterpretedValuesTracker.popAssertionLevel()
    }

    fun assertPendingAxioms(solver: Solver) {
        uninterpretedValuesTracker.assertPendingUninterpretedValueConstraints(solver)
    }

    fun getUninterpretedSortValueInterpreter(sort: KUninterpretedSort): Long? =
        uninterpretedSortValueInterpreter[sort]

    fun registerUninterpretedSortValueInterpreter(sort: KUninterpretedSort, interpreter: Long) {
        uninterpretedSortValueInterpreter[sort] = interpreter
    }

    fun findInternalConstDeclAssociatedUninterpretedSortValue(decl: Long): KUninterpretedSortValue? =
        uninterpretedSortValueDecls.get(decl)

    fun isInternalFuncDecl(decl: Long): Boolean =
        uninterpretedSortValueInterpreters.contains(decl)

    inline fun <K> findOrSave(
        key: K,
        computeValue: () -> Long,
        find: (K) -> Long,
        save: (K, Long) -> Long
    ): Long {
        val value = find(key)
        if (value != NOT_INTERNALIZED) return value
        return save(key, computeValue())
    }

    inline fun <V> findOrSave(
        key: Long,
        computeValue: () -> V,
        find: (Long) -> V?,
        save: (Long, V) -> V
    ): V {
        val value = find(key)
        if (value != null) return value
        return save(key, computeValue())
    }

    private fun <T> saveConvertedAst(
        native: Long,
        ksmt: T,
        cache: Object2LongOpenHashMap<T>,
        reverseCache: Long2ObjectOpenHashMap<T>
    ): T {
        val reverseCached = reverseCache.putIfAbsent(native, ksmt)
        if (reverseCached != null) return reverseCached

        cache.putIfAbsent(ksmt, native)
        incRefUnsafe(nCtx, native)

        return ksmt
    }

    private fun <T> saveAst(
        native: Long,
        ksmt: T,
        cache: Object2LongOpenHashMap<T>,
        reverseCache: Long2ObjectOpenHashMap<T>
    ): Long {
        val cached = cache.putIfAbsent(ksmt, native)
        if (cached == NOT_INTERNALIZED) {
            incRefUnsafe(nCtx, native)
            reverseCache.put(native, ksmt)
            return native
        }
        return cached
    }

    override fun close() {
        if (isClosed) return
        isClosed = true

        uninterpretedSortValueInterpreter.clear()

        uninterpretedSortValueDecls.keys.decRefAll()
        uninterpretedSortValueDecls.clear()

        uninterpretedSortValueInterpreters.decRefAll()
        uninterpretedSortValueInterpreters.clear()

        converterNativeObjects.decRefAll()
        converterNativeObjects.clear()

        z3Expressions.keys.decRefAll()
        expressions.clear()
        z3Expressions.clear()

        tmpNativeObjects.decRefAll()
        tmpNativeObjects.clear()

        z3Decls.keys.decRefAll()
        decls.clear()
        z3Decls.clear()

        z3Sorts.keys.decRefAll()
        sorts.clear()
        z3Sorts.clear()

        ctx.close()
    }

    private fun LongSet.decRefAll() =
        longIterator().forEachRemaining {
            decRefUnsafe(nCtx, it)
        }
}
