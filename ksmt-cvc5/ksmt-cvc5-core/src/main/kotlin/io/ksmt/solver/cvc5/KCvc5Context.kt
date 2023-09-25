package io.ksmt.solver.cvc5

import io.github.cvc5.Kind
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
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort
import java.util.TreeMap

class KCvc5Context(
    private val solver: Solver,
    private val ctx: KContext
) : AutoCloseable {
    private var isClosed = false

    private val uninterpretedSortCollector = KUninterpretedSortCollector(this)
    private var exprCurrentLevelCacheRestorer = KCurrentScopeExprCacheRestorer(uninterpretedSortCollector, ctx)

    /**
     * We use double-scoped expression internalization cache:
     *  * current (before pop operation) - [currentScopeExpressions]
     *  * global (after pop operation) - [expressions]
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
     *  (and put this expr to the current scope cache)
     */
    private val currentScopeExpressions = HashMap<KExpr<*>, Term>()
    private val expressions = HashMap<KExpr<*>, Term>()
    // we can't use HashMap with Term and Sort (hashcode is not implemented)
    private val cvc5Expressions = TreeMap<Term, KExpr<*>>()
    private val sorts = HashMap<KSort, Sort>()
    private val cvc5Sorts = TreeMap<Sort, KSort>()
    private val decls = HashMap<KDecl<*>, Term>()
    private val cvc5Decls = TreeMap<Term, KDecl<*>>()

    private var currentLevelUninterpretedSorts = hashSetOf<KUninterpretedSort>()
    private val uninterpretedSorts = mutableListOf(currentLevelUninterpretedSorts)

    private var currentLevelDeclarations = hashSetOf<KDecl<*>>()
    private val declarations = mutableListOf(currentLevelDeclarations)

    fun addUninterpretedSort(sort: KUninterpretedSort) { currentLevelUninterpretedSorts += sort }

    /**
     * uninterpreted sorts of active push-levels
     */
    fun uninterpretedSorts(): List<Set<KUninterpretedSort>> = uninterpretedSorts

    fun addDeclaration(decl: KDecl<*>) {
        currentLevelDeclarations += decl
        uninterpretedSortCollector.collect(decl)
    }

    /**
     * declarations of active push-levels
     */
    fun declarations(): List<Set<KDecl<*>>> = declarations

    val nativeSolver: Solver
        get() = solver

    val isActive: Boolean
        get() = !isClosed

    fun push() {
        currentLevelDeclarations = hashSetOf()
        declarations.add(currentLevelDeclarations)
        currentLevelUninterpretedSorts = hashSetOf()
        uninterpretedSorts.add(currentLevelUninterpretedSorts)

        pushAssertionLevel()
    }

    fun pop(n: UInt) {
        repeat(n.toInt()) {
            declarations.removeLast()
            uninterpretedSorts.removeLast()

            popAssertionLevel()
        }

        currentLevelDeclarations = declarations.last()
        currentLevelUninterpretedSorts = uninterpretedSorts.last()

        expressions += currentScopeExpressions
        currentScopeExpressions.clear()
        // recreate cache restorer to avoid KNonRecursiveTransformer cache
        exprCurrentLevelCacheRestorer = KCurrentScopeExprCacheRestorer(uninterpretedSortCollector, ctx)
    }

    // expr
    fun findInternalizedExpr(expr: KExpr<*>): Term? = currentScopeExpressions[expr]
        ?: expressions[expr]?.also {
            /*
             * expr is not in current scope cache, but in global cache.
             * Recollect declarations and uninterpreted sorts
             * and add entire expression tree to the current scope cache from the global
             * to avoid re-internalizing with native calls
             */
            exprCurrentLevelCacheRestorer.apply(expr)
        }

    fun findConvertedExpr(expr: Term): KExpr<*>? = cvc5Expressions[expr]

    fun saveInternalizedExpr(expr: KExpr<*>, internalized: Term): Term =
        internalizeAst(currentScopeExpressions, cvc5Expressions, expr) { internalized }

    /**
     * save expr, which is in global cache, to the current scope cache
     */
    fun savePreviouslyInternalizedExpr(expr: KExpr<*>): Term = saveInternalizedExpr(expr, expressions[expr]!!)

    fun saveConvertedExpr(expr: Term, converted: KExpr<*>): KExpr<*> =
        convertAst(currentScopeExpressions, cvc5Expressions, expr) { converted }

    // sort
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

    // decl
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


    @Suppress("ForbiddenComment")
    /**
     * Uninterpreted sort values distinct constraints management.
     *
     * 1. save/register uninterpreted value.
     * See [KUninterpretedSortValue] internalization for the details.
     * 2. Assert distinct constraints ([assertPendingAxioms]) that may be introduced during internalization.
     * Currently, we assert constraints for all the values we have ever internalized.
     *
     * todo: precise uninterpreted sort values tracking
     * */
    private data class UninterpretedSortValueDescriptor(
        val value: KUninterpretedSortValue,
        val nativeUniqueValueDescriptor: Term,
        val nativeValueTerm: Term
    )

    private var currentValueConstraintsLevel = 0
    private val assertedConstraintLevels = arrayListOf<Int>()
    private val uninterpretedSortValueDescriptors = arrayListOf<UninterpretedSortValueDescriptor>()
    private val uninterpretedSortValueInterpreter = hashMapOf<KUninterpretedSort, Term>()

    private val uninterpretedSortValues =
        hashMapOf<KUninterpretedSort, MutableList<Pair<Term, KUninterpretedSortValue>>>()

    fun saveUninterpretedSortValue(nativeValue: Term, value: KUninterpretedSortValue): Term {
        val sortValues = uninterpretedSortValues.getOrPut(value.sort) { arrayListOf() }
        sortValues.add(nativeValue to value)
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

        registerUninterpretedSortValue(value, uniqueValueDescriptorTerm, uninterpretedValueTerm)
    }

    fun assertPendingAxioms(solver: Solver) {
        assertPendingUninterpretedValueConstraints(solver)
    }

    fun getUninterpretedSortValueInterpreter(sort: KUninterpretedSort): Term? =
        uninterpretedSortValueInterpreter[sort]

    fun registerUninterpretedSortValueInterpreter(sort: KUninterpretedSort, interpreter: Term) {
        uninterpretedSortValueInterpreter[sort] = interpreter
    }

    fun registerUninterpretedSortValue(
        value: KUninterpretedSortValue,
        uniqueValueDescriptorTerm: Term,
        uninterpretedValueTerm: Term
    ) {
        uninterpretedSortValueDescriptors += UninterpretedSortValueDescriptor(
            value = value,
            nativeUniqueValueDescriptor = uniqueValueDescriptorTerm,
            nativeValueTerm = uninterpretedValueTerm
        )
    }

    fun getRegisteredSortValues(sort: KUninterpretedSort): List<Pair<Term, KUninterpretedSortValue>> =
        uninterpretedSortValues[sort] ?: emptyList()

    private fun pushAssertionLevel() {
        assertedConstraintLevels.add(currentValueConstraintsLevel)
    }

    private fun popAssertionLevel() {
        currentValueConstraintsLevel = assertedConstraintLevels.removeLast()
    }

    private fun assertPendingUninterpretedValueConstraints(solver: Solver) {
        while (currentValueConstraintsLevel < uninterpretedSortValueDescriptors.size) {
            assertUninterpretedSortValueConstraint(
                solver,
                uninterpretedSortValueDescriptors[currentValueConstraintsLevel]
            )
            currentValueConstraintsLevel++
        }
    }

    private fun assertUninterpretedSortValueConstraint(solver: Solver, value: UninterpretedSortValueDescriptor) {
        val interpreter = uninterpretedSortValueInterpreter[value.value.sort]
            ?: error("Interpreter was not registered for sort: ${value.value.sort}")

        val constraintLhs = solver.mkTerm(Kind.APPLY_UF, arrayOf(interpreter, value.nativeValueTerm))
        val constraint = constraintLhs.eqTerm(value.nativeUniqueValueDescriptor)

        solver.assertFormula(constraint)
    }

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

        currentScopeExpressions.clear()
        expressions.clear()
        cvc5Expressions.clear()

        uninterpretedSorts.clear()
        currentLevelUninterpretedSorts.clear()

        declarations.clear()
        currentLevelDeclarations.clear()

        sorts.clear()
        cvc5Sorts.clear()
        decls.clear()
        cvc5Decls.clear()
    }

    class KUninterpretedSortCollector(private val cvc5Ctx: KCvc5Context) : KSortVisitor<Unit> {
        override fun visit(sort: KBoolSort) = Unit

        override fun visit(sort: KIntSort) = Unit

        override fun visit(sort: KRealSort) = Unit

        override fun <S : KBvSort> visit(sort: S) = Unit

        override fun <S : KFpSort> visit(sort: S) = Unit

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            sort.domain.accept(this)
            sort.range.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
            sort.domainSorts.forEach { it.accept(this) }
            sort.range.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
            sort.domainSorts.forEach { it.accept(this) }
            sort.range.accept(this)
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>) {
            sort.domainSorts.forEach { it.accept(this) }
            sort.range.accept(this)
        }

        override fun visit(sort: KFpRoundingModeSort) = Unit

        override fun visit(sort: KUninterpretedSort) = cvc5Ctx.addUninterpretedSort(sort)

        fun collect(decl: KDecl<*>) {
            decl.argSorts.map { it.accept(this) }
            decl.sort.accept(this)
        }
    }

    inner class KCurrentScopeExprCacheRestorer(
        private val uninterpretedSortCollector: KUninterpretedSortCollector,
        ctx: KContext
    ) : KNonRecursiveTransformer(ctx) {

        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean = expr !in currentScopeExpressions

        override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = cacheIfNeed(expr) {
            this@KCvc5Context.addDeclaration(expr.decl)
            uninterpretedSortCollector.collect(expr.decl)
        }

        override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = cacheIfNeed(expr) {
            this@KCvc5Context.addDeclaration(expr.decl)
            uninterpretedSortCollector.collect(expr.decl)
            this@KCvc5Context.savePreviouslyInternalizedExpr(expr)
        }

        override fun <A : KArraySortBase<R>, R : KSort> transform(expr: KFunctionAsArray<A, R>): KExpr<A> =
            cacheIfNeed(expr) {
                this@KCvc5Context.addDeclaration(expr.function)
                uninterpretedSortCollector.collect(expr.function)
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
            this@KCvc5Context.savePreviouslyInternalizedExpr(expr)
        }

        private fun transformQuantifier(bounds: Set<KDecl<*>>, body: KExpr<*>) {
            val usedDecls = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(body) - bounds
            usedDecls.forEach(this@KCvc5Context::addDeclaration)
        }

        private fun <S : KSort, E : KExpr<S>> cacheIfNeed(expr: E, transform: E.() -> Unit): KExpr<S> {
            if (expr in currentScopeExpressions)
                return expr

            expr.transform()
            this@KCvc5Context.savePreviouslyInternalizedExpr(expr)
            return expr
        }
    }
}
