package io.ksmt.solver.maxsat.test

data class MaxSMTTestInfo(
    val softConstraintsWeights: List<UInt>,
    val softConstraintsWeightsSum: ULong,
    val satSoftConstraintsWeightsSum: ULong
)
