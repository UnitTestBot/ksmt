package io.ksmt.solver.cvc5

/**
 * An uninterpreted sort values tracker with ability to fork,
 * preserving all registered descriptors ([uninterpretedSortValueDescriptors]).
 * In a newly-forked tracker all known axioms will be asserted at
 * the nearest call of [assertPendingUninterpretedValueConstraints]
 */
class ExpressionUninterpretedValuesForkingTracker : ExpressionUninterpretedValuesTracker {
    constructor(cvc5Ctx: KCvc5Context) : super(cvc5Ctx)
    private constructor(
        cvc5Ctx: KCvc5Context,
        uninterpretedSortValueDescriptors: ArrayList<UninterpretedSortValueDescriptor>
    ) : super(cvc5Ctx, uninterpretedSortValueDescriptors)

    fun fork(childCvc5Ctx: KCvc5Context) =
        ExpressionUninterpretedValuesForkingTracker(childCvc5Ctx, uninterpretedSortValueDescriptors).also { child ->
            repeat(assertedConstraintLevels.size) {
                child.pushAssertionLevel()
            }
        }
}
