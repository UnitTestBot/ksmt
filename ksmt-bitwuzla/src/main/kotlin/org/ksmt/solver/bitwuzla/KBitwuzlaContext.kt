package org.ksmt.solver.bitwuzla

import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KExpr
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

open class KBitwuzlaContext : AutoCloseable {
    private var isClosed = false

    val bitwuzla = Native.bitwuzlaNew()

    val trueTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkTrue(bitwuzla) }
    val falseTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkFalse(bitwuzla) }
    val boolSort: BitwuzlaSort by lazy { Native.bitwuzlaMkBoolSort(bitwuzla) }

    private var expressions = hashMapOf<KExpr<*>, BitwuzlaTerm>()
    private val bitwuzlaExpressions = hashMapOf<BitwuzlaTerm, KExpr<*>>()
    private val sorts = hashMapOf<KSort, BitwuzlaSort>()
    private val bitwuzlaSorts = hashMapOf<BitwuzlaSort, KSort>()
    private val declSorts = hashMapOf<KDecl<*>, BitwuzlaSort>()

    private val bitwuzlaValues = hashMapOf<BitwuzlaTerm, KExpr<*>>()

    private var constants = hashMapOf<KDecl<*>, BitwuzlaTerm>()
    private val bitwuzlaConstants = hashMapOf<BitwuzlaTerm, KDecl<*>>()

    private var declarations = hashSetOf<KDecl<*>>()
    private var uninterpretedSorts = hashMapOf<KUninterpretedSort, HashSet<KDecl<*>>>()
    private var uninterpretedSortRegisterer = UninterpretedSortRegisterer(uninterpretedSorts)

    private var declarationScope = DeclarationScope(
        expressions = expressions,
        constants = constants,
        declarations = declarations,
        uninterpretedSorts = uninterpretedSorts,
        parentScope = null
    )

    operator fun get(expr: KExpr<*>): BitwuzlaTerm? = expressions[expr]
    operator fun get(sort: KSort): BitwuzlaSort? = sorts[sort]

    /**
     * Internalize ksmt expr into [BitwuzlaTerm] and cache internalization result to avoid
     * internalization of already internalized expressions.
     *
     * [internalizer] must use special functions to internalize BitVec values ([internalizeBvValue])
     * and constants ([mkConstant]).
     * */
    fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> BitwuzlaTerm): BitwuzlaTerm =
        expressions.getOrPut(expr) {
            internalizer(expr) // don't reverse cache bitwuzla term since it may be rewrote
        }

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
        convert(expressions, bitwuzlaExpressions, expr, converter)

    fun convertSort(sort: BitwuzlaSort, converter: (BitwuzlaSort) -> KSort): KSort =
        convert(sorts, bitwuzlaSorts, sort, converter)

    fun convertValue(value: BitwuzlaTerm): KExpr<*>? = bitwuzlaValues[value]

    // Constant is known only if it was previously internalized
    fun convertConstantIfKnown(term: BitwuzlaTerm): KDecl<*>? = bitwuzlaConstants[term]

    // Find normal constant if it was previously internalized
    fun findConstant(decl: KDecl<*>): BitwuzlaTerm? = constants[decl]

    fun declarations(): Set<KDecl<*>> = declarations

    fun uninterpretedSortsWithRelevantDecls(): Map<KUninterpretedSort, Set<KDecl<*>>> = uninterpretedSorts

    /**
     * Add declaration to the current declaration scope.
     * Also, if declaration sort is uninterpreted,
     * register this declaration as relevant to the sort.
     * */
    private fun registerDeclaration(decl: KDecl<*>) {
        if (declarations.add(decl)) {
            uninterpretedSortRegisterer.decl = decl
            decl.sort.accept(uninterpretedSortRegisterer)
            if (decl is KFuncDecl<*>) {
                decl.argSorts.forEach { it.accept(uninterpretedSortRegisterer) }
            }
        }
    }

    /**
     * Internalize constant.
     *  Since [Native.bitwuzlaMkConst] creates fresh constant on each invocation caches are used
     *  to guarantee that if two constants are equal in ksmt they are also equal in Bitwuzla.
     * */
    fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = constants.getOrPut(decl) {
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
        expressions = expressions.toMap(hashMapOf())
        constants = constants.toMap(hashMapOf())
        declarations = declarations.toHashSet()
        uninterpretedSorts = uninterpretedSorts.mapValuesTo(hashMapOf()) { (_, decls) -> decls.toHashSet() }
        uninterpretedSortRegisterer = UninterpretedSortRegisterer(uninterpretedSorts)
        declarationScope = DeclarationScope(expressions, constants, declarations, uninterpretedSorts, declarationScope)
    }

    /**
     * Pop declaration scope to ensure that [declarations] match
     * the set of asserted declarations at the current assertion level ([KBitwuzlaSolver.pop]).
     *
     * We also invalidate [expressions] internalization cache, since it may contain
     * expressions with invalidated declarations.
     * */
    fun popDeclarationScope() {
        declarationScope = declarationScope.parentScope
            ?: error("Can't pop root declaration scope")

        expressions = declarationScope.expressions
        constants = declarationScope.constants
        declarations = declarationScope.declarations
        uninterpretedSorts = declarationScope.uninterpretedSorts
        uninterpretedSortRegisterer = UninterpretedSortRegisterer(uninterpretedSorts)
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
        expressions.clear()
        bitwuzlaExpressions.clear()
        constants.clear()
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

    private class DeclarationScope(
        val expressions: HashMap<KExpr<*>, BitwuzlaTerm>,
        val constants: HashMap<KDecl<*>, BitwuzlaTerm>,
        val declarations: HashSet<KDecl<*>>,
        val uninterpretedSorts: HashMap<KUninterpretedSort, HashSet<KDecl<*>>>,
        val parentScope: DeclarationScope?
    )

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
}
