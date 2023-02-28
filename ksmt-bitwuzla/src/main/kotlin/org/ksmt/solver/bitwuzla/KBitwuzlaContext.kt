package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KArrayLambda
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
import org.ksmt.sort.KArraySort
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
    private val bitwuzlaSorts = hashMapOf<BitwuzlaSort, KSort>()
    private val declSorts = hashMapOf<KDecl<*>, BitwuzlaSort>()

    private val bitwuzlaValues = hashMapOf<BitwuzlaTerm, KExpr<*>>()

    private var exprCurrentLevelCache = hashMapOf<KExpr<*>, BitwuzlaTerm>()
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
        val globalTerm = exprGlobalCache[expr] ?: return null

        val currentLevelTerm = exprCurrentLevelCache[expr]
        if (currentLevelTerm != null) return currentLevelTerm

        currentLevelExprMover.apply(expr)

        return globalTerm
    }

    fun saveExprTerm(expr: KExpr<*>, term: BitwuzlaTerm) {
        if (exprCurrentLevelCache.putIfAbsent(expr, term) == null) {
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
        exprCurrentLevelCache = hashMapOf()
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
            // Move expr to current level
            val term = exprGlobalCache.getValue(expr)
            exprCacheLevel[expr] = currentLevel
            exprCurrentLevelCache[expr] = term

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

        override fun <D : KSort, R : KSort> transform(expr: KFunctionAsArray<D, R>): KExpr<KArraySort<D, R>> {
            registerDeclIfNotIgnored(expr.function)
            return super.transform(expr)
        }

        private val quantifiedVarsScope = arrayListOf<Pair<KExpr<*>, Set<KDecl<*>>?>>()

        private fun <T : KSort> KExpr<T>.transformQuantifier(bounds: List<KDecl<*>>, body: KExpr<*>): KExpr<T> {
            if (quantifiedVarsScope.lastOrNull()?.first != this) {
                quantifiedVarsScope.add(this to currentlyIgnoredDeclarations)
                val ignoredDecls = currentlyIgnoredDeclarations?.toHashSet() ?: hashSetOf()
                ignoredDecls.addAll(bounds)
                currentlyIgnoredDeclarations = ignoredDecls
            }
            return transformExprAfterTransformed(this, body) {
                currentlyIgnoredDeclarations = quantifiedVarsScope.removeLast().second
                this
            }
        }

        override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>> =
            expr.transformQuantifier(listOf(expr.indexVarDecl), expr.body)

        override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
            expr.transformQuantifier(expr.bounds, expr.body)

        override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort>  =
            expr.transformQuantifier(expr.bounds, expr.body)
    }
}
