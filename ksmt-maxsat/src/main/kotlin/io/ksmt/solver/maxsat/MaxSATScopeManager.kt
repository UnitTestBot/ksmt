package io.ksmt.solver.maxsat

class MaxSATScopeManager {
    private var currentScope = 0u

    private val prevScopes = mutableListOf<MaxSATScope>()

    private var scopeAddedSoftConstraints = 0

    fun incrementSoft() {
        if (currentScope != 0u) {
            scopeAddedSoftConstraints++
        }
    }

    fun push() {
        if (currentScope != 0u) {
            prevScopes.add(MaxSATScope(scopeAddedSoftConstraints))
            scopeAddedSoftConstraints = 0
        }

        currentScope++
    }

    fun pop(n: UInt, softConstraints: MutableList<SoftConstraint>): MutableList<SoftConstraint> {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }
        if (n == 0u) {
            return softConstraints
        }

        repeat(n.toInt()) {
            val size = softConstraints.size
            softConstraints.subList(size - scopeAddedSoftConstraints, size).clear()

            if (prevScopes.isNotEmpty()) {
                scopeAddedSoftConstraints = prevScopes.last().scopeAddedSoftConstraints
                prevScopes.removeLast()
            } else {
                scopeAddedSoftConstraints = 0
            }
        }

        currentScope -= n

        return softConstraints
    }
}
