package io.ksmt.solver.maxsat

import io.ksmt.solver.maxsat.KMaxSATContext.Strategy.PrimalDualMaxRes

class KMaxSATContext(
    val strategy: Strategy = PrimalDualMaxRes,
    val preferLargeWeightConstraintsForCores: Boolean = true,
    val minimizeCores: Boolean = true,
    val getMultipleCores: Boolean = true,
) {

    enum class Strategy {
        PrimalMaxRes,
        PrimalDualMaxRes,
    }
}
