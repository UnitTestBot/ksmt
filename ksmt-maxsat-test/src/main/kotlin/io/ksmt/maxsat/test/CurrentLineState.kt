package io.ksmt.maxsat.test

enum class CurrentLineState {
    COMMENT,
    HARD_CONSTRAINT,
    SOFT_CONSTRAINT,
    ERROR,
}
