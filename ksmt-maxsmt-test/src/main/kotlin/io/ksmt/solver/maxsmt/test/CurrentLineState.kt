package io.ksmt.solver.maxsmt.test

enum class CurrentLineState {
    COMMENT,
    HARD_CONSTRAINT,
    SOFT_CONSTRAINT,
    ERROR,
}
