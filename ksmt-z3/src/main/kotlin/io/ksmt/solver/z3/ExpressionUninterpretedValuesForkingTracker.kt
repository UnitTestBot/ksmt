package io.ksmt.solver.z3

import io.ksmt.KContext
import io.ksmt.expr.KUninterpretedSortValue

/**
 * Uninterpreted sort values tracker with ability to fork.
 * On child tracker creation ([fork]), it will have the same [registeredUninterpretedSortValues] as its parent
 * to prevent descriptors loss. [expressionLevels] and [valueTrackerFrames] are copied
 * to restore parental state of caching.
 * Also, all axioms are asserted lazily on [assertPendingUninterpretedValueConstraints]
 */
class ExpressionUninterpretedValuesForkingTracker : ExpressionUninterpretedValuesTracker {
    private constructor(
        ctx: KContext,
        z3Ctx: KZ3Context,
        registeredUninterpretedSortValues: HashMap<KUninterpretedSortValue, UninterpretedSortValueDescriptor>
    ) : super(ctx, z3Ctx, registeredUninterpretedSortValues)

    constructor(ctx: KContext, z3Ctx: KZ3Context) : super(ctx, z3Ctx)

    fun fork(z3Ctx: KZ3Context) = ExpressionUninterpretedValuesForkingTracker(
        ctx, z3Ctx, registeredUninterpretedSortValues
    ).also { child ->
        child.expressionLevels += expressionLevels

        var isFirstFrame = true
        valueTrackerFrames.forEach { frame ->
            if (!isFirstFrame) {
                child.pushAssertionLevel()
            }

            if (frame.initialized) {
                child.currentFrame.ensureInitialized()
                child.currentFrame.currentLevelUninterpretedValues += frame.currentLevelUninterpretedValues
                child.currentFrame.currentLevelExpressions += frame.currentLevelExpressions
            }

            isFirstFrame = false
        }
    }
}
