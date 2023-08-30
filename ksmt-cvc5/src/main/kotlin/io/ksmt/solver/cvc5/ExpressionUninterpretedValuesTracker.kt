package io.ksmt.solver.cvc5

import io.github.cvc5.Kind
import io.github.cvc5.Solver
import io.github.cvc5.Term
import io.ksmt.decl.KDecl
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort

open class ExpressionUninterpretedValuesTracker protected constructor(
    private val cvc5Ctx: KCvc5Context,
    protected val uninterpretedSortValueDescriptors: ArrayList<UninterpretedSortValueDescriptor>
) {
    constructor(cvc5Ctx: KCvc5Context) : this(cvc5Ctx, arrayListOf())

    private var currentValueConstraintsLevel = 0
    protected val assertedConstraintLevels = arrayListOf<Int>()

    private val uninterpretedSortCollector = KUninterpretedSortCollector(cvc5Ctx)

    fun collectUninterpretedSorts(decl: KDecl<*>) {
        uninterpretedSortCollector.collect(decl)
    }

    fun pushAssertionLevel() {
        assertedConstraintLevels += currentValueConstraintsLevel
    }

    fun popAssertionLevel() {
        currentValueConstraintsLevel = assertedConstraintLevels.removeLast()
    }

    fun registerUninterpretedSortValue(
        value: KUninterpretedSortValue,
        uniqueValueDescriptorTerm: Term,
        uninterpretedValueTerm: Term
    ) {
        uninterpretedSortValueDescriptors += UninterpretedSortValueDescriptor(
            value = value,
            nativeUniqueValueDescriptor = uniqueValueDescriptorTerm,
            nativeValueTerm = uninterpretedValueTerm
        )
    }

    fun assertPendingUninterpretedValueConstraints(solver: Solver) {
        while (currentValueConstraintsLevel < uninterpretedSortValueDescriptors.size) {
            assertUninterpretedSortValueConstraint(
                solver,
                uninterpretedSortValueDescriptors[currentValueConstraintsLevel]
            )
            currentValueConstraintsLevel++
        }
    }

    private fun assertUninterpretedSortValueConstraint(solver: Solver, value: UninterpretedSortValueDescriptor) {
        val interpreter = cvc5Ctx.getUninterpretedSortValueInterpreter(value.value.sort)
            ?: error("Interpreter was not registered for sort: ${value.value.sort}")

        val constraintLhs = solver.mkTerm(Kind.APPLY_UF, arrayOf(interpreter, value.nativeValueTerm))
        val constraint = constraintLhs.eqTerm(value.nativeUniqueValueDescriptor)

        solver.assertFormula(constraint)
    }

    @Suppress("ForbiddenComment")
    /**
     * Uninterpreted sort values distinct constraints management.
     *
     * 1. save/register uninterpreted value.
     * See [KUninterpretedSortValue] internalization for the details.
     * 2. Assert distinct constraints ([assertPendingUninterpretedValueConstraints]) that may be introduced
     * during internalization.
     * Currently, we assert constraints for all the values we have ever internalized.
     *
     * todo: precise uninterpreted sort values tracking
     * */
    protected data class UninterpretedSortValueDescriptor(
        val value: KUninterpretedSortValue,
        val nativeUniqueValueDescriptor: Term,
        val nativeValueTerm: Term
    )

    class KUninterpretedSortCollector(private val cvc5Ctx: KCvc5Context) : KSortVisitor<Unit> {
        override fun visit(sort: KBoolSort) = Unit

        override fun visit(sort: KIntSort) = Unit

        override fun visit(sort: KRealSort) = Unit

        override fun <S : KBvSort> visit(sort: S) = Unit

        override fun <S : KFpSort> visit(sort: S) = Unit

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            sort.domain.accept(this)
            sort.range.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
            sort.domainSorts.forEach { it.accept(this) }
            sort.range.accept(this)
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
            sort.domainSorts.forEach { it.accept(this) }
            sort.range.accept(this)
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>) {
            sort.domainSorts.forEach { it.accept(this) }
            sort.range.accept(this)
        }

        override fun visit(sort: KFpRoundingModeSort) = Unit

        override fun visit(sort: KUninterpretedSort) = cvc5Ctx.addUninterpretedSort(sort)

        fun collect(decl: KDecl<*>) {
            decl.argSorts.map { it.accept(this) }
            decl.sort.accept(this)
        }
    }
}
