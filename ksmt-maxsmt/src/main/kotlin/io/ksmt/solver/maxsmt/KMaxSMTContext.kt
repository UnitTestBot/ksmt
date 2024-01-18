package io.ksmt.solver.maxsmt

import io.ksmt.solver.maxsmt.KMaxSMTContext.Strategy.PrimalDualMaxRes

class KMaxSMTContext(
    val strategy: Strategy = PrimalDualMaxRes,
    val preferLargeWeightConstraintsForCores: Boolean = false,
    val minimizeCores: Boolean = false,
    val getMultipleCores: Boolean = false,
) {

    enum class Strategy {
        PrimalMaxRes,
        PrimalDualMaxRes,
    }
}
