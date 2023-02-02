package org.ksmt.solver.cvc5

import io.github.cvc5.Solver
import org.ksmt.solver.KSolverConfiguration

interface KCvc5SolverConfiguration : KSolverConfiguration {
    fun setCvc5Option(option: String, value: String)
}

class KCvc5SolverConfigurationImpl(val solver: Solver) : KCvc5SolverConfiguration {
    override fun setCvc5Option(option: String, value: String) {
        solver.setOption(option, value)
    }
}
