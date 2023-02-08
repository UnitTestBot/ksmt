package org.ksmt.solver.bitwuzla

import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolverException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaNativeException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KSort

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

    private val bitwuzlaValues = hashMapOf<BitwuzlaTerm, KExpr<*>>()

    private val normalConstantsOnly = hashMapOf<KDecl<*>, BitwuzlaTerm>()
    private val constantsAndVars = hashMapOf<KDecl<*>, BitwuzlaTerm>()
    private val bitwuzlaNormalConstants = hashMapOf<BitwuzlaTerm, KDecl<*>>()

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
    fun convertConstantIfKnown(term: BitwuzlaTerm): KDecl<*>? = bitwuzlaNormalConstants[term]

    // Find normal constant if it was previously internalized
    fun findNormalConstant(decl: KDecl<*>): BitwuzlaTerm? = normalConstantsOnly[decl]

    fun declaredConstants(): Set<KDecl<*>> = normalConstantsOnly.keys

    /**
     * Internalize constant.
     *  1. Since [Native.bitwuzlaMkConst] creates fresh constant on each invocation caches are used
     *  to guarantee that if two constants are equal in ksmt they are also equal in Bitwuzla.
     *  2. Returns [Native.bitwuzlaMkVar] for previously registered quantified variables (see [registerVar]).
     * */
    fun mkConstant(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm = constantsAndVars.getOrPut(decl) {
        Native.bitwuzlaMkConst(bitwuzla, sort, decl.name).also {
            normalConstantsOnly[decl] = it
            bitwuzlaNormalConstants[it] = decl
        }
    }

    /**
     * Register quantified variable. Provided [decl] must be unique (e.g. fresh)
     * and shouldn't overlap with any other variable.
     * */
    fun registerVar(decl: KDecl<*>, sort: BitwuzlaSort): BitwuzlaTerm {
        check(decl !in constantsAndVars) { "Vars must be unique" }
        val bitwuzlaVar = Native.bitwuzlaMkVar(bitwuzla, sort, decl.name)
        constantsAndVars[decl] = bitwuzlaVar
        return bitwuzlaVar
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
        constantsAndVars.clear()
        normalConstantsOnly.clear()
        declSorts.clear()
        bitwuzlaNormalConstants.clear()
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
}
