package org.ksmt.solver.bitwuzla

import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.bitwuzla.bindings.Bitwuzla
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.Native

interface KBitwuzlaSolverConfiguration : KSolverConfiguration {
    fun setBitwuzlaOption(option: BitwuzlaOption, value: Int)
    fun setBitwuzlaOption(option: BitwuzlaOption, value: String)
}

class KBitwuzlaSolverConfigurationImpl(private val bitwuzla: Bitwuzla) : KBitwuzlaSolverConfiguration {
    override fun setBitwuzlaOption(option: BitwuzlaOption, value: Int) {
        Native.bitwuzlaSetOption(bitwuzla, option, value)
    }

    override fun setBitwuzlaOption(option: BitwuzlaOption, value: String) {
        Native.bitwuzlaSetOptionStr(bitwuzla, option, value)
    }
}
