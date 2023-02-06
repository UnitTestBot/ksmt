package org.ksmt.solver.bitwuzla

import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaNativeException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KSort
import java.util.Collections

open class KBitwuzlaContext : AutoCloseable {
    private var isClosed = false

    val bitwuzla = Native.bitwuzlaNew()

    val trueTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkTrue(bitwuzla) }
    val falseTerm: BitwuzlaTerm by lazy { Native.bitwuzlaMkFalse(bitwuzla) }
    val boolSort: BitwuzlaSort by lazy { Native.bitwuzlaMkBoolSort(bitwuzla) }

    private val expressions = hashMapOf<KExpr<*>, BitwuzlaTerm>()
    private val bitwuzlaExpressions = hashMapOf<BitwuzlaTerm, KExpr<*>>()
    private val sorts = hashMapOf<KSort, BitwuzlaSort>()
    private val bitwuzlaSorts = hashMapOf<BitwuzlaSort, KSort>()
    private val declSorts = hashMapOf<KDecl<*>, BitwuzlaSort>()
    private val bitwuzlaConstants = hashMapOf<BitwuzlaTerm, KDecl<*>>()
    private val bitwuzlaValues = hashMapOf<BitwuzlaTerm, KExpr<*>>()

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
        }

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

    private val normalConstantScope: NormalConstantScope = NormalConstantScope()
    var currentConstantScope: ConstantScope = normalConstantScope

    fun declaredConstants(): Set<KDecl<*>> = Collections.unmodifiableSet(normalConstantScope.constants)

    /**
     * Internalize constant.
     *  1. Since [Native.bitwuzlaMkConst] creates fresh constant on each invocation caches are used
     *  to guarantee that if two constants are equal in ksmt they are also equal in Bitwuzla;
     *  2. Scoping is used to support quantifier bound variables (see [withConstantScope]).
     * */
    fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = currentConstantScope.mkConstant(decl, sort)

    /**
     * Constant scope for quantifiers.
     *
     * Quantifier bound variables:
     * 1. may clash with constants from outer scope;
     * 2. must be replaced with vars in quantifier body.
     *
     * @see QuantifiedConstantScope
     * */
    inline fun <reified T> withConstantScope(body: QuantifiedConstantScope.() -> T): T {
        val oldScope = currentConstantScope
        val newScope = QuantifiedConstantScope(currentConstantScope)

        currentConstantScope = newScope
        val result = newScope.body()
        currentConstantScope = oldScope

        return result
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

    sealed interface ConstantScope {
        fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm
    }


    /**
     * Produce normal constants for KSMT declarations.
     *
     * Guarantee that if two declarations are equal
     * then the same Bitwuzla native pointer is returned.
     * */
    inner class NormalConstantScope : ConstantScope {
        val constants = hashSetOf<KDecl<*>>()
        private val constantCache = hashMapOf<Pair<KDecl<*>, BitwuzlaSort>, BitwuzlaTerm>()

        override fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm =
            constantCache.getOrPut(decl to sort) {
                Native.bitwuzlaMkConst(bitwuzla, sort, decl.name).also {
                    constants += decl
                    bitwuzlaConstants[it] = decl
                }
            }
    }

    /**
     * Constant quantification.
     *
     * If constant declaration is registered as quantified variable,
     * then the corresponding var is returned instead of constant.
     * */
    inner class QuantifiedConstantScope(private val parent: ConstantScope) : ConstantScope {
        private val constants = hashMapOf<Pair<KDecl<*>, BitwuzlaSort>, BitwuzlaTerm>()
        fun mkVar(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm =
            constants.getOrPut(decl to sort) {
                Native.bitwuzlaMkVar(bitwuzla, sort, decl.name)
            }

        override fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm =
            constants[decl to sort] ?: parent.mkConstant(decl, sort)
    }
}
