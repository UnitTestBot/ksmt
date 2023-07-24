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

class KCvc5Context private constructor(
    private val solver: Solver,
    private val ctx: KContext,
    parent: KCvc5Context?,
    isForking: Boolean
) : AutoCloseable {
    constructor(solver: Solver, ctx: KContext, isForking: Boolean = false) : this(solver, ctx, null, isForking)

    private var isClosed = false
    private val isChild = parent != null

    private val uninterpretedSortCollector = KUninterpretedSortCollector(this)
    private var exprCurrentLevelCacheRestorer = KCurrentScopeExprCacheRestorer(uninterpretedSortCollector, ctx)

    private val uninterpretedSorts: ScopedFrame<HashSet<KUninterpretedSort>>
    private val declarations: ScopedFrame<HashSet<KDecl<*>>>


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
    private val currentAccumulatedScopeExpressions: HashMap<KExpr<*>, Term>
    private val expressions: HashMap<KExpr<*>, Term>

    // we can't use HashMap with Term and Sort (hashcode is not implemented)
    private val cvc5Expressions: TreeMap<Term, KExpr<*>>
    private val sorts: HashMap<KSort, Sort>
    private val cvc5Sorts: TreeMap<Sort, KSort>
    private val decls: HashMap<KDecl<*>, Term>
    private val cvc5Decls: TreeMap<Term, KDecl<*>>

    private val uninterpretedSortValueDescriptors: ArrayList<UninterpretedSortValueDescriptor>
    private val uninterpretedSortValueInterpreter: HashMap<KUninterpretedSort, Term>

    /**
     * Uninterpreted sort values and universe are shared for whole forking hierarchy (from parent to children)
     * due to shared expressions cache,
     * that's why once [registerUninterpretedSortValue] and [saveUninterpretedSortValue] are called,
     * each solver in hierarchy should assert newly internalized uninterpreted sort values via [assertPendingAxioms]
     *
     * @see KCvc5Model.uninterpretedSortUniverse
     */
    private val uninterpretedSortValues: HashMap<KUninterpretedSort, MutableList<Pair<Term, KUninterpretedSortValue>>>

    init {
        if (isForking) {
            uninterpretedSorts = (parent?.uninterpretedSorts as? ScopedLinkedFrame)?.fork()
                ?: ScopedLinkedFrame(::HashSet, ::HashSet)
            declarations = (parent?.declarations as? ScopedLinkedFrame)?.fork()
                ?: ScopedLinkedFrame(::HashSet, ::HashSet)
        } else {
            uninterpretedSorts = ScopedArrayFrame(::HashSet)
            declarations = ScopedArrayFrame(::HashSet)
        }

        if (parent != null) {
            currentAccumulatedScopeExpressions = parent.currentAccumulatedScopeExpressions.toMap(HashMap())
            expressions = parent.expressions
            cvc5Expressions = parent.cvc5Expressions
            sorts = parent.sorts
            cvc5Sorts = parent.cvc5Sorts
            decls = parent.decls
            cvc5Decls = parent.cvc5Decls
            uninterpretedSortValueDescriptors = parent.uninterpretedSortValueDescriptors
            uninterpretedSortValueInterpreter = parent.uninterpretedSortValueInterpreter
            uninterpretedSortValues = parent.uninterpretedSortValues
        } else {
            currentAccumulatedScopeExpressions = HashMap()
            expressions = HashMap()
            cvc5Expressions = TreeMap()
            sorts = HashMap()
            cvc5Sorts = TreeMap()
            decls = HashMap()
            cvc5Decls = TreeMap()
            uninterpretedSortValueDescriptors = arrayListOf()
            uninterpretedSortValueInterpreter = hashMapOf()
            uninterpretedSortValues = hashMapOf()
        }
    }

    fun addUninterpretedSort(sort: KUninterpretedSort) {
        uninterpretedSorts.currentFrame += sort
    }

    fun uninterpretedSorts(): Set<KUninterpretedSort> = uninterpretedSorts.flatten { this += it }

    fun addDeclaration(decl: KDecl<*>) {
        declarations.currentFrame += decl
        uninterpretedSortCollector.collect(decl)
    }

    fun declarations(): Set<KDecl<*>> = declarations.flatten { this += it }

    val nativeSolver: Solver
        get() = solver

    val isActive: Boolean
        get() = !isClosed

    fun fork(solver: Solver): KCvc5Context = KCvc5Context(solver, ctx, this, true).also { forkCtx ->
        repeat(assertedConstraintLevels.size) {
            forkCtx.pushAssertionLevel()
        }
    }

    fun push() {
        declarations.push()
        uninterpretedSorts.push()

        pushAssertionLevel()
    }

    fun pop(n: UInt) {
        declarations.pop(n)
        uninterpretedSorts.pop(n)

        repeat(n.toInt()) { popAssertionLevel() }

        currentAccumulatedScopeExpressions.clear()
        // recreate cache restorer to avoid KNonRecursiveTransformer cache
        exprCurrentLevelCacheRestorer = KCurrentScopeExprCacheRestorer(uninterpretedSortCollector, ctx)
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

    /**
     * Uninterpreted sort value axioms will not be lost for [KCvc5ForkingSolver] on [fork].
     *
     * On child initialization, "[currentValueConstraintsLevel] = 0"
     * will be pushed to [assertedConstraintLevels] for each push-level ([currentValueConstraintsLevel] times).
     * At the first call of [assertPendingAxioms] each descriptor from [uninterpretedSortValueDescriptors]
     * will be asserted to the child [KCvc5ForkingSolver]
     */
    private var currentValueConstraintsLevel = 0
    private val assertedConstraintLevels = arrayListOf<Int>()

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
        assertedConstraintLevels += currentValueConstraintsLevel
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

        currentAccumulatedScopeExpressions.clear()

        if (isChild) {
            expressions.clear()
            cvc5Expressions.clear()

            sorts.clear()
            cvc5Sorts.clear()

            decls.clear()
            cvc5Decls.clear()
        }
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

        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean =
            expr !in currentAccumulatedScopeExpressions

        override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = cacheIfNeed(expr) {
            this@KCvc5Context.addDeclaration(expr.decl)
            uninterpretedSortCollector.collect(expr.decl)
        }

        override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = cacheIfNeed(expr) {
            this@KCvc5Context.addDeclaration(expr.decl)
            uninterpretedSortCollector.collect(expr.decl)
            saveInternalizedExprToCurrentAccumulatedScope(expr)
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
