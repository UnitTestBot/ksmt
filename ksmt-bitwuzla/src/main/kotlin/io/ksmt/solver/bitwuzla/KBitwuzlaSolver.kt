package io.ksmt.solver.bitwuzla

import io.ksmt.KContext

open class KBitwuzlaSolver(ctx: KContext) : KBitwuzlaSolverBase(ctx) {

    override fun configure(configurator: KBitwuzlaSolverConfiguration.() -> Unit) {
        KBitwuzlaSolverConfigurationImpl(bitwuzlaCtx.bitwuzla).configurator()
    }
}
