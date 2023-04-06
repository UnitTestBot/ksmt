package org.ksmt.solver.z3

import com.microsoft.z3.Native
import com.microsoft.z3.Solver
import com.microsoft.z3.solverAssert
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort

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
        currentLevel = 0,
        notAssertedConstraintsFromPreviousLevels = 0
    )

    private val valueTrackerFrames = arrayListOf(currentFrame)

    private val registeredUninterpretedSortValues =
        hashMapOf<KUninterpretedSortValue, UninterpretedSortValueDescriptor>()

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
        for (frameIdx in valueTrackerFrames.lastIndex downTo 0) {
            val frame = valueTrackerFrames[frameIdx]

            while (frame.notAssertedConstraints > 0) {
                val valueConstraint = frame.assertNextConstraint()
                assertUninterpretedSortValueConstraint(solver, valueConstraint)
            }

            if (frame.notAssertedConstraintsFromPreviousLevels == 0) break
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
        val currentLevel: Int,
        val notAssertedConstraintsFromPreviousLevels: Int
    ) {
        private var initailized = false
        private var lastAssertedConstraint = 0

        lateinit var currentLevelExpressions: MutableSet<KExpr<*>>
        lateinit var currentLevelUninterpretedValues: MutableList<UninterpretedSortValueDescriptor>
        lateinit var currentLevelExprAnalyzer: ExprUninterpretedValuesAnalyzer

        private fun ensureInitialized() {
            if (initailized) return

            currentLevelExpressions = hashSetOf()
            currentLevelUninterpretedValues = arrayListOf()
            currentLevelExprAnalyzer = ExprUninterpretedValuesAnalyzer(ctx, this)

            initailized = true
        }

        private val numberOfConstraints: Int
            get() = if (!initailized) 0 else currentLevelUninterpretedValues.size

        val notAssertedConstraints: Int
            get() = numberOfConstraints - lastAssertedConstraint

        fun nextFrame(): ValueTrackerAssertionFrame {
            val nextLevelRemainingConstraints = notAssertedConstraintsFromPreviousLevels + notAssertedConstraints
            return ValueTrackerAssertionFrame(
                ctx, tracker, expressionLevels,
                currentLevel = currentLevel + 1,
                notAssertedConstraintsFromPreviousLevels = nextLevelRemainingConstraints
            )
        }

        fun assertNextConstraint(): UninterpretedSortValueDescriptor =
            currentLevelUninterpretedValues[lastAssertedConstraint++]

        fun cleanupAfterPop() {
            if (!initailized) return

            currentLevelExprAnalyzer = ExprUninterpretedValuesAnalyzer(ctx, this)
        }

        fun analyzeUsedExpression(expr: KExpr<*>) {
            ensureInitialized()

            if (expr in currentLevelExpressions) return
            currentLevelExprAnalyzer.apply(expr)
        }

        fun addExpression(expr: KExpr<*>) {
            ensureInitialized()

            if (currentLevelExpressions.add(expr)) {
                expressionLevels.put(expr, currentLevel)
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
                expressionLevels.put(expr, currentLevel)
            }
            return super.transformExpr(expr)
        }

        override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> {
            frame.addRegisteredValueToCurrentLevel(expr)
            return super.transform(expr)
        }

        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean = with(frame) {
            val frameLevel = expressionLevels.getInt(expr)
            if (frameLevel < currentLevel) {
                val levelFrame = getFrame(frameLevel)
                // If expr is valid on its level we don't need to move it
                return expr !in levelFrame.currentLevelExpressions
            }
            return super.exprTransformationRequired(expr)
        }
    }
}
