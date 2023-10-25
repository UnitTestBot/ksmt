package io.ksmt.solver.maxsmt

import io.ksmt.solver.maxsmt.KMaxSMTContext.Strategy.PrimalDualMaxRes

class KMaxSMTContext(
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
