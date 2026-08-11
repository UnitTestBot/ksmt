package io.ksmt.solver.z3

import com.microsoft.z3.Params
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.Array
import io.ksmt.solver.KTheory.BV
import io.ksmt.solver.KTheory.FP
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.KTheory.UF
import io.ksmt.solver.smtLib2String

interface KZ3SolverConfiguration : KSolverConfiguration {
    fun setZ3Option(option: String, value: Boolean)
    fun setZ3Option(option: String, value: Int)
    fun setZ3Option(option: String, value: Double)
    fun setZ3Option(option: String, value: String)

    override fun setBoolParameter(param: String, value: Boolean) {
        setZ3Option(param, value)
    }

    override fun setIntParameter(param: String, value: Int) {
        setZ3Option(param, value)
    }

    override fun setStringParameter(param: String, value: String) {
        setZ3Option(param, value)
    }

    override fun setDoubleParameter(param: String, value: Double) {
        setZ3Option(param, value)
    }
}

sealed class KZ3SolverConfigurationImpl(val params: Params) : KZ3SolverConfiguration {
    override fun setZ3Option(option: String, value: Boolean) {
        params.add(option, value)
    }

    override fun setZ3Option(option: String, value: Int) {
        params.add(option, value)
    }

    override fun setZ3Option(option: String, value: Double) {
        params.add(option, value)
    }

    override fun setZ3Option(option: String, value: String) {
        params.add(option, value)
    }
}

/**
 * Sort of the descriptor values ksmt uses to keep uninterpreted sort values distinct
 * (see `KZ3ExprInternalizer.transform(KUninterpretedSortValue)`).
 * */
enum class KZ3UninterpretedValueDescriptor { INT, BV }

/**
 * A logic to create the Z3 solver for, and the descriptor sort that logic permits.
 * A `null` [logic] means the general solver, which imposes no restriction.
 * */
data class KZ3ResolvedLogic(val logic: String?, val valueDescriptor: KZ3UninterpretedValueDescriptor)

class KZ3SolverLazyConfiguration(params: Params) : KZ3SolverConfigurationImpl(params) {
    private var theories: Set<KTheory>? = null
    private var quantifiersAllowed: Boolean = false

    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        this.theories = theories
        this.quantifiersAllowed = quantifiersAllowed
    }

    /**
     * ksmt keeps uninterpreted sort values distinct through values of a descriptor sort, so every
     * query constrains that sort whether or not the caller declared its theory. A logic that
     * forbids it makes Z3 answer UNKNOWN or expose the sort in the model, so the two are chosen
     * together. Combinations with no specialized solver fall back to the general one.
     * */
    fun resolveLogic(): KZ3ResolvedLogic {
        val requestedTheories = theories
        if (requestedTheories.isNullOrEmpty()) return GENERAL_SOLVER

        val hasIntegerArithmetic = LIA in requestedTheories || NIA in requestedTheories

        if (BV in requestedTheories && !hasIntegerArithmetic) {
            return resolve(requestedTheories, KZ3UninterpretedValueDescriptor.BV)
        }

        val theoriesWithDescriptor = if (hasIntegerArithmetic) requestedTheories else requestedTheories + LIA
        return resolve(theoriesWithDescriptor, KZ3UninterpretedValueDescriptor.INT)
    }

    private fun resolve(theories: Set<KTheory>, descriptor: KZ3UninterpretedValueDescriptor) =
        if (supportedLogicCombination(theories, quantifiersAllowed)) {
            KZ3ResolvedLogic(theories.smtLib2String(quantifiersAllowed), descriptor)
        } else {
            GENERAL_SOLVER
        }

    /**
     * Z3 provide special solver only for the following theory combinations
     * */
    private fun supportedLogicCombination(theories: Set<KTheory>, quantifiersAllowed: Boolean): Boolean =
        if (quantifiersAllowed) {
            theories in supportedTheoriesWithQuantifiers
        } else {
            theories in supportedQuantifierFreeTheories
        }

    companion object {
        private val GENERAL_SOLVER = KZ3ResolvedLogic(logic = null, KZ3UninterpretedValueDescriptor.INT)

        private fun l(vararg theories: KTheory) = theories.toSet()

        private val supportedTheoriesWithQuantifiers = setOf(
            l(Array, BV),
            l(Array, LIA),
            l(Array, UF, BV),
            l(Array, UF, LIA),
            l(Array, UF, LIA, LRA),
            l(Array, UF, NIA),
            l(Array, UF, NIA, NRA),
            l(BV),
            l(FP),
            l(LIA),
            l(LRA),
            l(NIA),
            l(NRA),
            l(UF),
            l(UF, BV),
            l(UF, LIA),
            l(UF, LRA),
            l(UF, NIA),
            l(UF, NIA, NRA),
            l(UF, NRA),
        )

        private val supportedQuantifierFreeTheories = setOf(
            l(Array),
            l(Array, BV),
            l(Array, LIA),
            l(Array, NIA),
            l(Array, UF, BV),
            l(Array, UF, LIA),
            l(Array, UF, LIA, LRA),
            l(Array, UF, NIA),
            l(Array, UF, NIA, NRA),
            l(BV),
            l(BV, FP),
            l(FP),
            l(FP, LRA),
            l(LIA),
            l(LIA, LRA),
            l(LRA),
            l(NIA),
            l(NIA, NRA),
            l(NRA),
            l(UF),
            l(UF, BV),
            l(UF, LIA),
            l(UF, LRA),
            l(UF, NIA),
            l(UF, NIA, NRA),
            l(UF, NRA),
        )
    }
}

class KZ3SolverParamsConfiguration(params: Params) : KZ3SolverConfigurationImpl(params) {
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        throw KSolverException("Solver logic already configured")
    }
}

class KZ3SolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KZ3SolverConfiguration {
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        builder.buildOptimizeForTheories(theories, quantifiersAllowed)
    }

    override fun setZ3Option(option: String, value: Boolean) {
        builder.buildBoolParameter(option, value)
    }

    override fun setZ3Option(option: String, value: Int) {
        builder.buildIntParameter(option, value)
    }

    override fun setZ3Option(option: String, value: Double) {
        builder.buildDoubleParameter(option, value)
    }

    override fun setZ3Option(option: String, value: String) {
        builder.buildStringParameter(option, value)
    }
}
