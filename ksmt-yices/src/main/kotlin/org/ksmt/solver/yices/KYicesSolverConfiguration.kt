package org.ksmt.solver.yices

import com.sri.yices.Config
import org.ksmt.solver.KSolverConfiguration

interface KYicesSolverConfiguration : KSolverConfiguration {
    fun setYicesOption(option: String, value: String)
}

class KYicesSolverConfigurationImpl(private val config: Config) : KYicesSolverConfiguration {
    override fun setYicesOption(option: String, value: String) {
        config.set(option, value)
    }
}
