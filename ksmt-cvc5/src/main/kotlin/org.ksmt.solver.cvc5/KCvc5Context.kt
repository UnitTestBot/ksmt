package org.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.github.cvc5.Sort
import io.github.cvc5.Term
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import java.util.TreeMap

class KCvc5Context(private val solver: Solver) : AutoCloseable {
    private var isClosed = false

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
    fun uninterpretedSorts(): Set<KUninterpretedSort> = uninterpretedSorts.flatten().toSet()

    fun addDeclaration(sort: KDecl<*>) { currentLevelDeclarations += sort }
    fun declarations(): Set<KDecl<*>> = declarations.flatten().toSet()

    val nativeSolver: Solver
        get() = solver

    val isActive: Boolean
        get() = !isClosed

    fun push() {
        currentLevelDeclarations = hashSetOf()
        declarations.add(currentLevelDeclarations)
        currentLevelUninterpretedSorts = hashSetOf()
        uninterpretedSorts.add(currentLevelUninterpretedSorts)
    }

    fun pop(n: UInt) {
        repeat(n.toInt()) {
            declarations.removeLast()
            uninterpretedSorts.removeLast()
        }

        currentLevelDeclarations = declarations.last()
        currentLevelUninterpretedSorts = uninterpretedSorts.last()

        expressions += currentScopeExpressions
        currentScopeExpressions.clear()
    }

    // expr
    fun findInternalizedExpr(expr: KExpr<*>): Term? = expressions[expr]

    fun findCurrentScopeInternalizedExpr(expr: KExpr<*>): Term? = currentScopeExpressions[expr]

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
}
