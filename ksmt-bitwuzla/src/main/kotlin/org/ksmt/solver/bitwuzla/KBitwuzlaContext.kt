package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KConst
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.solver.KSolverException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaNativeException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort

open class KBitwuzlaContext(val ctx: KContext) : AutoCloseable {
    private var isClosed = false

    val bitwuzla = Native.bitwuzlaNew()

    val trueTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkTrue(bitwuzla) }
    val falseTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkFalse(bitwuzla) }
    val boolSort: BitwuzlaSort by lazy { Native.bitwuzlaMkBoolSort(bitwuzla) }

    private val exprGlobalCache = hashMapOf<KExpr<*>, BitwuzlaTerm>()
    private val bitwuzlaExpressions = hashMapOf<BitwuzlaTerm, KExpr<*>>()

    private val constantsGlobalCache = hashMapOf<KDecl<*>, BitwuzlaTerm>()
    private val bitwuzlaConstants = hashMapOf<BitwuzlaTerm, KDecl<*>>()

    private val sorts = hashMapOf<KSort, BitwuzlaSort>()
    private val declSorts = hashMapOf<KDecl<*>, BitwuzlaSort>()
    private val bitwuzlaSorts = hashMapOf<BitwuzlaSort, KSort>()

    private val bitwuzlaValues = hashMapOf<BitwuzlaTerm, KExpr<*>>()

    private var exprCurrentLevelCache = hashSetOf<KExpr<*>>()
    private val exprCacheLevel = hashMapOf<KExpr<*>, Int>()
    private val exprLeveledCache = arrayListOf(exprCurrentLevelCache)
    private var currentLevelExprMover = ExprMover()

    private val currentLevel: Int
        get() = exprLeveledCache.lastIndex

    private var currentLevelDeclarations = hashSetOf<KDecl<*>>()
    private val leveledDeclarations = arrayListOf(currentLevelDeclarations)

    private var currentLevelUninterpretedSorts = hashMapOf<KUninterpretedSort, HashSet<KDecl<*>>>()
    private var currentLevelUninterpretedSortRegisterer = UninterpretedSortRegisterer(currentLevelUninterpretedSorts)
    private val leveledUninterpretedSorts = arrayListOf(currentLevelUninterpretedSorts)


    /**
     * Search for expression term and
     * ensure correctness of currently known declarations.
     *
     * 1. Expression is in current level cache.
     * All declarations are already known.
     *
     * 2. Expression is not in global cache.
     * Expression is not internalized yet.
     *
     * 3. Expression is in global cache but not in current level cache.
     * Move expression and recollect declarations.
     * See [ExprMover].
     * */
    fun findExprTerm(expr: KExpr<*>): BitwuzlaTerm? {
        val term = exprGlobalCache[expr] ?: return null

        if (expr in exprCurrentLevelCache) return term

        currentLevelExprMover.apply(expr)

        return term
    }

    fun saveExprTerm(expr: KExpr<*>, term: BitwuzlaTerm) {
        if (exprCurrentLevelCache.add(expr)) {
            exprGlobalCache[expr] = term
            exprCacheLevel[expr] = currentLevel
        }
    }

    operator fun get(sort: KSort): BitwuzlaSort? = sorts[sort]

    fun internalizeSort(sort: KSort, internalizer: (KSort) -> BitwuzlaSort): BitwuzlaSort =
        sorts.getOrPut(sort) {
            internalizer(sort).also {
                bitwuzlaSorts[it] = sort
            }
        }

    fun internalizeDeclSort(decl: KDecl<*>, internalizer: (KDecl<*>) -> BitwuzlaSort): BitwuzlaSort =
        declSorts.getOrPut(decl) {
            internalizer(decl)
        }.also { registerDeclaration(decl) }

    /**
     * Internalize and reverse cache Bv value to support Bv values conversion.
     *
     * Since [Native.bitwuzlaGetBvValue] is only available after check-sat call
     * we must reverse cache Bv values to be able to convert all previously internalized
     * expressions.
     * */
    fun saveInternalizedValue(expr: KExpr<*>, term: BitwuzlaTerm) {
        bitwuzlaValues[term] = expr
    }

    fun findConvertedExpr(expr: BitwuzlaTerm): KExpr<*>? = bitwuzlaExpressions[expr]

    fun convertExpr(expr: BitwuzlaTerm, converter: (BitwuzlaTerm) -> KExpr<*>): KExpr<*> =
        convert(exprGlobalCache, bitwuzlaExpressions, expr, converter)

    fun convertSort(sort: BitwuzlaSort, converter: (BitwuzlaSort) -> KSort): KSort =
        convert(sorts, bitwuzlaSorts, sort, converter)

    fun convertValue(value: BitwuzlaTerm): KExpr<*>? = bitwuzlaValues[value]

    // Constant is known only if it was previously internalized
    fun convertConstantIfKnown(term: BitwuzlaTerm): KDecl<*>? = bitwuzlaConstants[term]

    // Find normal constant if it was previously internalized
    fun findConstant(decl: KDecl<*>): BitwuzlaTerm? = constantsGlobalCache[decl]

    fun declarations(): Set<KDecl<*>> =
        leveledDeclarations.flatMapTo(hashSetOf()) { it }

    fun uninterpretedSortsWithRelevantDecls(): Map<KUninterpretedSort, Set<KDecl<*>>> {
        val result = hashMapOf<KUninterpretedSort, MutableSet<KDecl<*>>>()

        leveledUninterpretedSorts.forEach { levelSorts ->
            levelSorts.forEach { entry ->
                val values = result.getOrPut(entry.key) { hashSetOf() }
                values.addAll(entry.value)
            }
        }

        return result
    }

    /**
     * Add declaration to the current declaration scope.
     * Also, if declaration sort is uninterpreted,
     * register this declaration as relevant to the sort.
     * */
    private fun registerDeclaration(decl: KDecl<*>) {
        if (currentLevelDeclarations.add(decl)) {
            currentLevelUninterpretedSortRegisterer.decl = decl
            decl.sort.accept(currentLevelUninterpretedSortRegisterer)
            if (decl is KFuncDecl<*>) {
                decl.argSorts.forEach { it.accept(currentLevelUninterpretedSortRegisterer) }
            }
        }
    }

    /**
     * Internalize constant declaration.
     * Since [Native.bitwuzlaMkConst] creates fresh constant on each invocation caches are used
     * to guarantee that if two constants are equal in ksmt they are also equal in Bitwuzla.
     * */
    fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = constantsGlobalCache.getOrPut(decl) {
        Native.bitwuzlaMkConst(bitwuzla, sort, decl.name).also {
            bitwuzlaConstants[it] = decl
        }
    }.also { registerDeclaration(decl) }

    /**
     * Create nested declaration scope to allow [popDeclarationScope].
     * Declarations scopes are used to manage set of currently asserted declarations
     * and must match to the corresponding assertion level ([KBitwuzlaSolver.push]).
     * */
    fun createNestedDeclarationScope() {
        exprCurrentLevelCache = hashSetOf()
        exprLeveledCache.add(exprCurrentLevelCache)
        currentLevelExprMover = ExprMover()

        currentLevelDeclarations = hashSetOf()
        leveledDeclarations.add(currentLevelDeclarations)

        currentLevelUninterpretedSorts = hashMapOf()
        leveledUninterpretedSorts.add(currentLevelUninterpretedSorts)
        currentLevelUninterpretedSortRegisterer = UninterpretedSortRegisterer(currentLevelUninterpretedSorts)
    }

    /**
     * Pop declaration scope to ensure that [declarations] match
     * the set of asserted declarations at the current assertion level ([KBitwuzlaSolver.pop]).
     *
     * We also invalidate expressions internalization cache, since it may contain
     * expressions with invalidated declarations.
     * */
    fun popDeclarationScope() {
        exprLeveledCache.removeLast()
        exprCurrentLevelCache = exprLeveledCache.last()
        currentLevelExprMover = ExprMover()

        leveledDeclarations.removeLast()
        currentLevelDeclarations = leveledDeclarations.last()

        leveledUninterpretedSorts.removeLast()
        currentLevelUninterpretedSorts = leveledUninterpretedSorts.last()
        currentLevelUninterpretedSortRegisterer = UninterpretedSortRegisterer(currentLevelUninterpretedSorts)
    }

    inline fun <reified T> bitwuzlaTry(body: () -> T): T = try {
        ensureActive()
        body()
    } catch (ex: BitwuzlaNativeException) {
        throw KSolverException(ex)
    }

    override fun close() {
        isClosed = true
        sorts.clear()
        bitwuzlaSorts.clear()

        exprGlobalCache.clear()
        bitwuzlaExpressions.clear()
        constantsGlobalCache.clear()

        exprLeveledCache.clear()
        exprCurrentLevelCache.clear()

        declSorts.clear()
        bitwuzlaConstants.clear()
        Native.bitwuzlaDelete(bitwuzla)
    }

    fun ensureActive() {
        check(!isClosed) { "The context is already closed." }
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
        cache.putIfAbsent(converted, key)
        reverseCache[key] = converted

        return converted
    }

    private class UninterpretedSortRegisterer(
        private val register: MutableMap<KUninterpretedSort, HashSet<KDecl<*>>>
    ) : KSortVisitor<Unit> {
        lateinit var decl: KDecl<*>

        override fun visit(sort: KUninterpretedSort) {
            val sortElements = register.getOrPut(sort) { hashSetOf() }
            sortElements += decl
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            sort.domain.accept(this)
            sort.range.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
            sort.domain0.accept(this)
            sort.domain1.accept(this)
            sort.range.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
            sort.domain0.accept(this)
            sort.domain1.accept(this)
            sort.domain2.accept(this)
            sort.range.accept(this)
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>) {
            sort.domainSorts.forEach { it.accept(this) }
            sort.range.accept(this)
        }

        override fun visit(sort: KBoolSort) {
        }

        override fun visit(sort: KIntSort) {
        }

        override fun visit(sort: KRealSort) {
        }

        override fun <S : KBvSort> visit(sort: S) {
        }

        override fun <S : KFpSort> visit(sort: S) {
        }

        override fun visit(sort: KFpRoundingModeSort) {
        }
    }

    /**
     * Move expressions from other cache level to the current cache level
     * and register declarations for all moved expressions.
     *
     * 1. If expression is valid on previous levels we don't need to move it.
     * Also, all expression declarations are known.
     *
     * 2. Otherwise, move expression to current level and recollect declarations.
     * */
    private inner class ExprMover : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> {
            if (!insideQuantifiedScope) {
                /**
                 *  Move expr to current level.
                 *
                 *  Don't move quantified expression since:
                 *  1. Body may contain vars which can't be moved correctly
                 *  2. Expression caches will remain correct regardless of body moved
                 *  */
                if (exprCurrentLevelCache.add(expr)) {
                    exprCacheLevel[expr] = currentLevel
                }
            }

            return super.transformExpr(expr)
        }

        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean {
            val cachedLevel = exprCacheLevel[expr]
            if (cachedLevel != null && cachedLevel < currentLevel) {
                val levelCache = exprLeveledCache[cachedLevel]
                // If expr is valid on its level we don't need to move it
                return expr !in levelCache
            }
            return super.exprTransformationRequired(expr)
        }

        private var currentlyIgnoredDeclarations: Set<KDecl<*>>? = null

        private fun registerDeclIfNotIgnored(decl: KDecl<*>) {
            if (currentlyIgnoredDeclarations?.contains(decl) == true) {
                return
            }
            registerDeclaration(decl)
        }

        override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> {
            registerDeclIfNotIgnored(expr.decl)
            return super.transform(expr)
        }

        override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> {
            registerDeclIfNotIgnored(expr.decl)
            return super.transform(expr)
        }

        override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> {
            registerDeclIfNotIgnored(expr.function)
            return super.transform(expr)
        }

        private val quantifiedVarsScopeOwner = arrayListOf<KExpr<*>>()
        private val quantifiedVarsScope = arrayListOf<Set<KDecl<*>>?>()

        private val insideQuantifiedScope: Boolean
            get() = quantifiedVarsScopeOwner.isNotEmpty()

        private fun <T : KSort> KExpr<T>.transformQuantifier(bounds: List<KDecl<*>>, body: KExpr<*>): KExpr<T> {
            if (quantifiedVarsScopeOwner.lastOrNull() != this) {
                quantifiedVarsScopeOwner.add(this)
                quantifiedVarsScope.add(currentlyIgnoredDeclarations)

                val ignoredDecls = currentlyIgnoredDeclarations?.toHashSet() ?: hashSetOf()
                ignoredDecls.addAll(bounds)
                currentlyIgnoredDeclarations = ignoredDecls
            }
            return transformExprAfterTransformed(this, body) {
                quantifiedVarsScopeOwner.removeLast()
                currentlyIgnoredDeclarations = quantifiedVarsScope.removeLast()
                transformExpr(this)
            }
        }

        override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
            expr.transformQuantifier(expr.indexVarDeclarations, expr.body)

        override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
            expr: KArray2Lambda<D0, D1, R>
        ): KExpr<KArray2Sort<D0, D1, R>> =
            expr.transformQuantifier(expr.indexVarDeclarations, expr.body)

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
            expr: KArray3Lambda<D0, D1, D2, R>
        ): KExpr<KArray3Sort<D0, D1, D2, R>> =
            expr.transformQuantifier(expr.indexVarDeclarations, expr.body)

        override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>> =
            expr.transformQuantifier(expr.indexVarDeclarations, expr.body)

        override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
            expr.transformQuantifier(expr.bounds, expr.body)

        override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort>  =
            expr.transformQuantifier(expr.bounds, expr.body)
    }
}
