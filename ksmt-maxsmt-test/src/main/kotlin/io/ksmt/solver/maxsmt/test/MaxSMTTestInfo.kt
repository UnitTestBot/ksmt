package io.ksmt.solver.maxsmt.test

data class MaxSMTTestInfo(
    val softConstraintsWeights: List<UInt>,
    val softConstraintsWeightsSum: ULong,
    val satSoftConstraintsWeightsSum: ULong
)
