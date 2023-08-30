package io.ksmt.solver.cvc5

import io.github.cvc5.CVC5ApiException
import io.github.cvc5.Solver
import io.github.cvc5.Sort
import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KConst
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.solver.KSolverException
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import java.util.TreeMap

/**
 * @param mkExprSolver used as "context" for creation of all native expressions, which are stored in [KCvc5Context].
 */
open class KCvc5Context(
    protected val solver: Solver,
    val mkExprSolver: Solver,
    protected val ctx: KContext
) : AutoCloseable {

    /**
     * Creates [KCvc5Context]. All native expressions will be created via [solver].
     */
    constructor(solver: Solver, ctx: KContext) : this(solver, solver, ctx)

    protected var isClosed = false

    private var exprCurrentLevelCacheRestorer = KCurrentScopeExprCacheRestorer(ctx)

    protected open val uninterpretedSorts: ScopedFrame<HashSet<KUninterpretedSort>> = ScopedArrayFrame(::HashSet)
    protected open val declarations: ScopedFrame<HashSet<KDecl<*>>> = ScopedArrayFrame(::HashSet)


    /**
     * We use double-scoped expression internalization cache:
     *  * current accumulated (before pop operation) - [currentAccumulatedScopeExpressions]
     *  * global (current + all previously met)- [expressions]
     *
     * Due to incremental collect of declarations and uninterpreted sorts for the model,
     * we collect them during internalizing.
     *
     * **problem**: However, after asserting the expression that previously was in the cache,
     * its uninterpreted sorts / declarations would not be collected repeatedly.
     * Then, there are two scenarios:
     *  1) we have sorts / decls for the expr on one previous push levels
     *  2) we popped the scope, and relevant declarations have been erased
     *
     *  **solution**: Recollect sorts / decls for each expression
     *  that is in global cache, but whose sorts / decls have been erased after pop()
     *  (and put this expr to the cache of current accumulated scope)
     */
    protected val currentAccumulatedScopeExpressions = HashMap<KExpr<*>, Term>()
    protected open val expressions = HashMap<KExpr<*>, Term>()

    /**
     * We can't use HashMap with Term and Sort (hashcode is not implemented)
     */
    protected open val cvc5Expressions = TreeMap<Term, KExpr<*>>()
    protected open val sorts = HashMap<KSort, Sort>()
    protected open val cvc5Sorts = TreeMap<Sort, KSort>()
    protected open val decls = HashMap<KDecl<*>, Term>()
    protected open val cvc5Decls = TreeMap<Term, KDecl<*>>()

    @Suppress("LeakingThis")
    open val uninterpretedValuesTracker = ExpressionUninterpretedValuesTracker(this)
    protected open val uninterpretedSortValueInterpreter = HashMap<KUninterpretedSort, Term>()
    protected open val uninterpretedSortValues =
        HashMap<KUninterpretedSort, MutableList<Pair<Term, KUninterpretedSortValue>>>()

    fun addUninterpretedSort(sort: KUninterpretedSort) {
        uninterpretedSorts.currentFrame += sort
    }

    fun uninterpretedSorts(): Set<KUninterpretedSort> = uninterpretedSorts.flatten { this += it }

    fun addDeclaration(decl: KDecl<*>) {
        declarations.currentFrame += decl
        uninterpretedValuesTracker.collectUninterpretedSorts(decl)
    }

    fun declarations(): Set<KDecl<*>> = declarations.flatten { this += it }

    val nativeSolver: Solver
        get() = solver

    val isActive: Boolean
        get() = !isClosed

    fun ensureActive() = check(isActive) { "The context is already closed." }

    fun push() {
        declarations.push()
        uninterpretedSorts.push()
        uninterpretedValuesTracker.pushAssertionLevel()
    }

    fun pop(n: UInt) {
        declarations.pop(n)
        uninterpretedSorts.pop(n)

        repeat(n.toInt()) { uninterpretedValuesTracker.popAssertionLevel() }

        currentAccumulatedScopeExpressions.clear()
        // recreate cache restorer to avoid KNonRecursiveTransformer cache
        exprCurrentLevelCacheRestorer = KCurrentScopeExprCacheRestorer(ctx)
    }

    fun findInternalizedExpr(expr: KExpr<*>): Term? = currentAccumulatedScopeExpressions[expr]
        ?: expressions[expr]?.also {
            /*
             * expr is not in cache of current accumulated scope, but in global cache.
             * Recollect declarations and uninterpreted sorts
             * and add entire expression tree to the current accumulated scope cache from the global
             * to avoid re-internalization
             */
            exprCurrentLevelCacheRestorer.apply(expr)
        }

    open fun <T : KSort> Term.convert(converter: KCvc5ExprConverter) = with(converter) { convertExpr<T>() }

    fun findConvertedExpr(expr: Term): KExpr<*>? = cvc5Expressions[expr]

    fun saveInternalizedExpr(expr: KExpr<*>, internalized: Term): Term =
        internalizeAst(currentAccumulatedScopeExpressions, cvc5Expressions, expr) { internalized }
            .also { expressions[expr] = internalized }

    /**
     * save expr, which is in global cache, to the current scope cache
     */
    fun saveInternalizedExprToCurrentAccumulatedScope(expr: KExpr<*>): Term =
        currentAccumulatedScopeExpressions.getOrPut(expr) { expressions[expr]!! }

    fun saveConvertedExpr(expr: Term, converted: KExpr<*>): KExpr<*> =
        convertAst(currentAccumulatedScopeExpressions, cvc5Expressions, expr) { converted }
            .also { expressions[converted] = expr }

    fun findInternalizedSort(sort: KSort): Sort? = sorts[sort]

    fun findConvertedSort(sort: Sort): KSort? = cvc5Sorts[sort]

    fun saveInternalizedSort(sort: KSort, internalized: Sort): Sort =
        internalizeAst(sorts, cvc5Sorts, sort) { internalized }

    fun saveConvertedSort(sort: Sort, converted: KSort): KSort =
        convertAst(sorts, cvc5Sorts, sort) { converted }

    inline fun internalizeSort(sort: KSort, internalizer: () -> Sort): Sort =
        findOrSave(sort, internalizer, ::findInternalizedSort, ::saveInternalizedSort)

    inline fun convertSort(sort: Sort, converter: () -> KSort): KSort =
        findOrSave(sort, converter, ::findConvertedSort, ::saveConvertedSort)

    fun findInternalizedDecl(decl: KDecl<*>): Term? = decls[decl]

    fun findConvertedDecl(decl: Term): KDecl<*>? = cvc5Decls[decl]

    fun saveInternalizedDecl(decl: KDecl<*>, internalized: Term): Term =
        internalizeAst(decls, cvc5Decls, decl) { internalized }

    fun saveConvertedDecl(decl: Term, converted: KDecl<*>): KDecl<*> =
        convertAst(decls, cvc5Decls, decl) { converted }

    inline fun internalizeDecl(decl: KDecl<*>, internalizer: () -> Term): Term =
        findOrSave(decl, internalizer, ::findInternalizedDecl, ::saveInternalizedDecl)

    inline fun convertDecl(decl: Term, converter: () -> KDecl<*>): KDecl<*> =
        findOrSave(decl, converter, ::findConvertedDecl, ::saveConvertedDecl)

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

    fun saveUninterpretedSortValue(nativeValue: Term, value: KUninterpretedSortValue): Term {
        val sortValues = uninterpretedSortValues.getOrPut(value.sort) { arrayListOf() }
        sortValues += nativeValue to value
        return nativeValue
    }

    inline fun registerUninterpretedSortValue(
        value: KUninterpretedSortValue,
        uniqueValueDescriptorTerm: Term,
        uninterpretedValueTerm: Term,
        mkInterpreter: () -> Term
    ) {
        val interpreter = getUninterpretedSortValueInterpreter(value.sort)
        if (interpreter == null) {
            registerUninterpretedSortValueInterpreter(value.sort, mkInterpreter())
        }

        uninterpretedValuesTracker.registerUninterpretedSortValue(
            value,
            uniqueValueDescriptorTerm,
            uninterpretedValueTerm
        )
    }

    fun assertPendingAxioms(solver: Solver) {
        uninterpretedValuesTracker.assertPendingUninterpretedValueConstraints(solver)
    }

    fun getUninterpretedSortValueInterpreter(sort: KUninterpretedSort): Term? =
        uninterpretedSortValueInterpreter[sort]

    fun registerUninterpretedSortValueInterpreter(sort: KUninterpretedSort, interpreter: Term) {
        uninterpretedSortValueInterpreter[sort] = interpreter
    }

    fun getRegisteredSortValues(sort: KUninterpretedSort): List<Pair<Term, KUninterpretedSortValue>> =
        uninterpretedSortValues[sort] ?: emptyList()

    private inline fun <K> internalizeAst(
        cache: MutableMap<K, Term>,
        reverseCache: MutableMap<Term, K>,
        key: K,
        internalizer: () -> Term
    ): Term = cache.getOrPut(key) {
        internalizer().also {
            reverseCache[it] = key
        }
    }

    private inline fun <K> internalizeAst(
        cache: MutableMap<K, Sort>,
        reverseCache: MutableMap<Sort, K>,
        key: K,
        internalizer: () -> Sort
    ): Sort = cache.getOrPut(key) {
        internalizer().also {
            reverseCache[it] = key
        }
    }

    private inline fun <K> convertAst(
        cache: MutableMap<K, Sort>,
        reverseCache: MutableMap<Sort, K>,
        key: Sort,
        converter: () -> K
    ): K {
        val current = reverseCache[key]

        if (current != null) return current

        val converted = converter()
        cache.getOrPut(converted) { key }
        reverseCache[key] = converted

        return converted
    }

    private inline fun <K> convertAst(
        cache: MutableMap<K, Term>,
        reverseCache: MutableMap<Term, K>,
        key: Term,
        converter: () -> K
    ): K {
        val current = reverseCache[key]

        if (current != null) return current

        val converted = converter()
        cache.getOrPut(converted) { key }
        reverseCache[key] = converted

        return converted
    }

    override fun close() {
        if (isClosed) return
        isClosed = true

        currentAccumulatedScopeExpressions.clear()
        expressions.clear()
        cvc5Expressions.clear()
        sorts.clear()
        cvc5Sorts.clear()
        decls.clear()
        cvc5Decls.clear()
        uninterpretedSortValueInterpreter.clear()
        uninterpretedSortValues.clear()
    }

    inline fun <reified T> cvc5Try(body: () -> T): T = try {
        ensureActive()
        body()
    } catch (ex: CVC5ApiException) {
        throw KSolverException(ex)
    }

    inner class KCurrentScopeExprCacheRestorer(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean =
            expr !in currentAccumulatedScopeExpressions

        override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = cacheIfNeed(expr) {
            this@KCvc5Context.addDeclaration(expr.decl)
            uninterpretedValuesTracker.collectUninterpretedSorts(expr.decl)
        }

        override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = cacheIfNeed(expr) {
            this@KCvc5Context.addDeclaration(expr.decl)
            uninterpretedValuesTracker.collectUninterpretedSorts(expr.decl)
            saveInternalizedExprToCurrentAccumulatedScope(expr)
        }

        override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> =
            cacheIfNeed(expr) {
                this@KCvc5Context.addDeclaration(expr.function)
                uninterpretedValuesTracker.collectUninterpretedSorts(expr.function)
            }

        override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
            cacheIfNeed(expr) {
                transformQuantifier(setOf(expr.indexVarDecl), expr.body)
            }

        override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> = cacheIfNeed(expr) {
            transformQuantifier(expr.bounds.toSet(), expr.body)
        }

        override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> = cacheIfNeed(expr) {
            transformQuantifier(expr.bounds.toSet(), expr.body)
            saveInternalizedExprToCurrentAccumulatedScope(expr)
        }

        private fun transformQuantifier(bounds: Set<KDecl<*>>, body: KExpr<*>) {
            val usedDecls = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(body) - bounds
            usedDecls.forEach(this@KCvc5Context::addDeclaration)
        }

        private fun <S : KSort, E : KExpr<S>> cacheIfNeed(expr: E, transform: E.() -> Unit): KExpr<S> {
            if (expr in currentAccumulatedScopeExpressions)
                return expr

            expr.transform()
            saveInternalizedExprToCurrentAccumulatedScope(expr)
            return expr
        }
    }
}
