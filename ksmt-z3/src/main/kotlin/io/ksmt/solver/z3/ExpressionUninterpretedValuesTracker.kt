package io.ksmt.solver.z3

import com.microsoft.z3.Native
import com.microsoft.z3.Solver
import com.microsoft.z3.solverAssert
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort

/**
 * Uninterpreted sort values distinct constraints management.
 *
 * 1. save/register uninterpreted value.
 * See [KUninterpretedSortValue] internalization for the details.
 * 2. Assert distinct constraints ([assertPendingUninterpretedValueConstraints])
 * that may be introduced during internalization.
 * */
class ExpressionUninterpretedValuesTracker(val ctx: KContext, val z3Ctx: KZ3Context) {
    private val expressionLevels = Object2IntOpenHashMap<KExpr<*>>().apply {
        defaultReturnValue(Int.MAX_VALUE) // Level which is greater than any possible level
    }

    private var currentFrame = ValueTrackerAssertionFrame(
        ctx, this, expressionLevels,
        level = 0,
        notAssertedConstraintsFromPreviousLevels = 0
    )

    private val valueTrackerFrames = arrayListOf(currentFrame)

    private val registeredUninterpretedSortValues =
        hashMapOf<KUninterpretedSortValue, UninterpretedSortValueDescriptor>()

    /**
     * Skip any value tracking related actions until
     * we have uninterpreted values registered.
     *
     * Current [ValueTrackerAssertionFrame] correctly
     * handles situations when expression is belongs to
     * some assertion level lower than current level
     * but was not registered on that level.
     * */
    private inline fun ifTrackingEnabled(body: () -> Unit) {
        if (registeredUninterpretedSortValues.isEmpty()) return
        body()
    }

    fun expressionUse(expr: KExpr<*>) = ifTrackingEnabled {
        currentFrame.analyzeUsedExpression(expr)
    }

    fun expressionSave(expr: KExpr<*>) = ifTrackingEnabled {
        currentFrame.addExpression(expr)
    }

    fun registerUninterpretedSortValue(
        value: KUninterpretedSortValue,
        uniqueValueDescriptorExpr: Long,
        uninterpretedValueExpr: Long
    ) {
        val descriptor = UninterpretedSortValueDescriptor(
            value = value,
            nativeUniqueValueDescriptor = uniqueValueDescriptorExpr,
            nativeValueExpr = uninterpretedValueExpr
        )
        if (registeredUninterpretedSortValues.putIfAbsent(value, descriptor) == null) {
            currentFrame.addRegisteredValueToCurrentLevel(descriptor)
        }
    }

    fun pushAssertionLevel() {
        currentFrame = currentFrame.nextFrame()
        valueTrackerFrames.add(currentFrame)
    }

    fun popAssertionLevel() = ifTrackingEnabled {
        valueTrackerFrames.removeLast()
        currentFrame = valueTrackerFrames.last()

        currentFrame.cleanupAfterPop()
    }

    fun assertPendingUninterpretedValueConstraints(solver: Solver) {
        // Assert constraints into solver and mark them as asserted
        currentFrame.assertUnassertedConstraints {
            assertUninterpretedSortValueConstraint(solver, it)
        }

        var frame = currentFrame
        while (frame.notAssertedConstraintsFromPreviousLevels != 0 && frame.level > 0) {
            frame = valueTrackerFrames[frame.level - 1]

            /**
             * Assert constraints into solver but DON'T mark them as asserted
             * because these constraints belongs to the lower levels.
             * */
            frame.visitUnassertedConstraints {
                assertUninterpretedSortValueConstraint(solver, it)
            }
        }
    }

    private fun assertUninterpretedSortValueConstraint(solver: Solver, value: UninterpretedSortValueDescriptor) {
        val interpreter = z3Ctx.getUninterpretedSortValueInterpreter(value.value.sort)
            ?: error("Interpreter was not registered for sort: ${value.value.sort}")

        val constraintLhs = z3Ctx.temporaryAst(
            Native.mkApp(z3Ctx.nCtx, interpreter, 1, longArrayOf(value.nativeValueExpr))
        )
        val constraint = z3Ctx.temporaryAst(
            Native.mkEq(z3Ctx.nCtx, constraintLhs, value.nativeUniqueValueDescriptor)
        )

        solver.solverAssert(constraint)

        z3Ctx.releaseTemporaryAst(constraint)
        z3Ctx.releaseTemporaryAst(constraintLhs)
    }

    private data class UninterpretedSortValueDescriptor(
        val value: KUninterpretedSortValue,
        val nativeUniqueValueDescriptor: Long,
        val nativeValueExpr: Long
    )

    private class ValueTrackerAssertionFrame(
        val ctx: KContext,
        val tracker: ExpressionUninterpretedValuesTracker,
        val expressionLevels: Object2IntOpenHashMap<KExpr<*>>,
        val level: Int,
        val notAssertedConstraintsFromPreviousLevels: Int
    ) {
        private var initialized = false
        private var lastAssertedConstraint = 0

        lateinit var currentLevelExpressions: MutableSet<KExpr<*>>
        lateinit var currentLevelUninterpretedValues: MutableList<UninterpretedSortValueDescriptor>
        lateinit var currentLevelExprAnalyzer: ExprUninterpretedValuesAnalyzer

        /**
         * Delay initialization to reduce memory consumption
         * since we might not have any uninterpreted values on
         * a current assertion level.
         * */
        private fun ensureInitialized() {
            if (initialized) return

            currentLevelExpressions = hashSetOf()
            currentLevelUninterpretedValues = arrayListOf()
            currentLevelExprAnalyzer = ExprUninterpretedValuesAnalyzer(ctx, this)

            initialized = true
        }

        private val numberOfConstraints: Int
            get() = if (!initialized) 0 else currentLevelUninterpretedValues.size

        fun nextFrame(): ValueTrackerAssertionFrame {
            val notAssertedConstraints = numberOfConstraints - lastAssertedConstraint
            val nextLevelRemainingConstraints = notAssertedConstraintsFromPreviousLevels + notAssertedConstraints
            return ValueTrackerAssertionFrame(
                ctx, tracker, expressionLevels,
                level = level + 1,
                notAssertedConstraintsFromPreviousLevels = nextLevelRemainingConstraints
            )
        }

        inline fun assertUnassertedConstraints(action: (UninterpretedSortValueDescriptor) -> Unit) {
            // Was not initialized --> has no constraints
            if (!initialized) return

            visitUnassertedConstraints { action(it) }

            lastAssertedConstraint = currentLevelUninterpretedValues.size
        }

        inline fun visitUnassertedConstraints(action: (UninterpretedSortValueDescriptor) -> Unit) {
            // Was not initialized --> has no constraints
            if (!initialized) return

            for (i in lastAssertedConstraint until currentLevelUninterpretedValues.size) {
                action(currentLevelUninterpretedValues[i])
            }
        }

        fun cleanupAfterPop() {
            if (!initialized) return

            // Recreate analyzer after pop to reset transformer transformation caches
            currentLevelExprAnalyzer = ExprUninterpretedValuesAnalyzer(ctx, this)
        }

        fun analyzeUsedExpression(expr: KExpr<*>) {
            ensureInitialized()

            if (expr in currentLevelExpressions) return

            /**
             * We use an expression that doesn't belong to the current level.
             * Therefore, we must analyze it for used uninterpreted values.
             * */
            currentLevelExprAnalyzer.apply(expr)
        }

        fun addExpression(expr: KExpr<*>) {
            ensureInitialized()

            if (currentLevelExpressions.add(expr)) {
                expressionLevels.put(expr, level)
            }
        }

        fun addRegisteredValueToCurrentLevel(value: KUninterpretedSortValue) {
            val descriptor = tracker.registeredUninterpretedSortValues[value]
                ?: error("Value $value was not registered")
            addRegisteredValueToCurrentLevel(descriptor)
        }

        fun addRegisteredValueToCurrentLevel(descriptor: UninterpretedSortValueDescriptor) {
            ensureInitialized()

            currentLevelUninterpretedValues.add(descriptor)
        }

        fun getFrame(level: Int) = tracker.valueTrackerFrames[level]
    }

    private class ExprUninterpretedValuesAnalyzer(
        ctx: KContext,
        val frame: ValueTrackerAssertionFrame
    ) : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = with(frame) {
            if (currentLevelExpressions.add(expr)) {
                expressionLevels.put(expr, level)
            }
            return super.transformExpr(expr)
        }

        override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> {
            frame.addRegisteredValueToCurrentLevel(expr)
            return super.transform(expr)
        }

        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean = with(frame) {
            val frameLevel = expressionLevels.getInt(expr)
            if (frameLevel < level) {
                val levelFrame = getFrame(frameLevel)
                // If expr is valid on its level we don't need to move it
                return expr !in levelFrame.currentLevelExpressions
            }
            return super.exprTransformationRequired(expr)
        }
    }
}
