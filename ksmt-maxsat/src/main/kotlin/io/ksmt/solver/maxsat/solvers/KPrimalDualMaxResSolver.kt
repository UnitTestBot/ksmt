package io.ksmt.solver.maxsat.solvers

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.KSolverStatus.UNKNOWN
import io.ksmt.solver.KSolverStatus.UNSAT
import io.ksmt.solver.maxsat.KMaxSATResult
import io.ksmt.solver.maxsat.constraints.SoftConstraint
import io.ksmt.solver.maxsat.solvers.utils.MinimalUnsatCore
import io.ksmt.solver.maxsat.utils.CoreUtils
import io.ksmt.solver.maxsat.utils.ModelUtils
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

class KPrimalDualMaxResSolver<T : KSolverConfiguration>(private val ctx: KContext, private val solver: KSolver<T>) :
    KMaxResSolver<T>(ctx, solver) {
    private var _lower: UInt = 0u // Current lower frontier
    private var _upper: UInt = 0u // Current upper frontier
    private var _maxUpper = 0u // Max possible upper frontier
    private var _correctionSetSize: Int = 0 // Current corrections set size
    private val _maxCoreSize = 3
    private var _correctionSetModel: KModel? = null
    private var _model: KModel? = null
    private var _minimalUnsatCore = MinimalUnsatCore(ctx, solver)

    private data class WeightedCore(val expressions: List<KExpr<KBoolSort>>, val weight: UInt)

    override fun checkMaxSAT(timeout: Duration): KMaxSATResult {
        val hardConstraintsStatus = solver.check()

        if (hardConstraintsStatus == UNSAT || softConstraints.isEmpty()) {
            return KMaxSATResult(listOf(), hardConstraintsStatus, true)
        } else if (hardConstraintsStatus == UNKNOWN) {
            return KMaxSATResult(listOf(), hardConstraintsStatus, false)
        }

        solver.push()
        initMaxSAT()

        val assumptions = softConstraints.toMutableList()
        unionSoftConstraintsWithSameExpressions(assumptions)

        while (_lower < _upper) {
            val status = checkSat(assumptions)

            when (status) {
                SAT -> {
                    val correctionSet = getCorrectionSet(solver.model(), assumptions)
                    if (correctionSet.isEmpty()) {
                        if (_model != null) {
                            // Feasible optimum is found by the moment.
                            _lower = _upper
                        }
                    } else {
                        processSat(correctionSet, assumptions)
                    }
                }

                UNSAT -> {
                    val processUnsatStatus = processUnsat(assumptions)
                    if (processUnsatStatus == UNSAT) {
                        _lower = _upper
                    } else if (processUnsatStatus == UNKNOWN) {
                        solver.pop()
                        return KMaxSATResult(listOf(), SAT, false)
                    }
                }

                UNKNOWN -> {
                    solver.pop()
                    return KMaxSATResult(listOf(), SAT, false)
                }
            }
        }

        _lower = _upper

        val result = KMaxSATResult(getSatSoftConstraintsByModel(_model!!), SAT, true)

        solver.pop()

        return result
    }

    private fun processSat(correctionSet: List<SoftConstraint>, assumptions: MutableList<SoftConstraint>) {
        removeCoreAssumptions(correctionSet, assumptions)
        val (minWeight, _) = splitCore(correctionSet, assumptions)
        correctionSetMaxResolve(correctionSet, assumptions, minWeight)

        _correctionSetModel = null
        _correctionSetSize = 0
    }

    private fun processUnsat(assumptions: MutableList<SoftConstraint>): KSolverStatus {
        val (status, cores) = getCores(assumptions)

        if (status != SAT) {
            return status
        }

        if (cores.isEmpty()) {
            return UNSAT
        }

        for (core in cores) {
            processUnsatCore(core, assumptions)
        }

        return SAT
    }

    private fun processUnsatCore(weightedCore: WeightedCore, assumptions: MutableList<SoftConstraint>) = with(ctx) {
        val core = weightedCore.expressions

        require(core.isNotEmpty()) { "Core should not be empty here" }

        maxResolve(weightedCore, assumptions)

        val fml = !(core.reduce { x, y -> x and y })
        assert(fml)

        _lower += weightedCore.weight
        _lower = minOf(_lower, _upper)

        if (_correctionSetModel != null && _correctionSetSize > 0) {
            // This estimate can overshoot for weighted soft constraints.
            --_correctionSetSize
        }

        // Here we also prefer a smaller correction set to core.
        if (_correctionSetModel != null && _correctionSetSize < core.size) {
            val correctionSet = getCorrectionSet(_correctionSetModel!!, assumptions)
            if (correctionSet.size >= core.size) {
                return
            }

            var weight = 0u
            for (asm in assumptions) {
                val weight1 = asm.weight
                if (weight != 0u && weight1 != weight) {
                    return
                }

                weight = weight1
            }

            processSat(correctionSet, assumptions)
        }
    }

    private fun getCores(assumptions: MutableList<SoftConstraint>): Pair<KSolverStatus, List<WeightedCore>> {
        val cores = mutableListOf<WeightedCore>()
        var status = UNSAT

        while (status == UNSAT) {
            val minimalUnsatCore = minimizeCore(assumptions)
            updateMinimalUnsatCoreModel(assumptions)

            if (minimalUnsatCore.isEmpty()) {
                cores.clear()
                _lower = _upper
                return Pair(SAT, cores)
            }

            // 1. remove all core literals from assumptions
            // 2. re-add literals of higher weight than min-weight.
            // 3. 'core' stores the core literals that are re-encoded as assumptions afterward
            cores.add(WeightedCore(minimalUnsatCore.map { it.expression }, CoreUtils.getCoreWeight(minimalUnsatCore)))

            removeCoreAssumptions(minimalUnsatCore, assumptions)
            splitCore(minimalUnsatCore, assumptions)

            if (minimalUnsatCore.size >= _maxCoreSize) {
                return Pair(SAT, cores)
            }

            status = checkSat(assumptions)
        }

        return Pair(status, cores)
    }

    private fun minimizeCore(assumptions: List<SoftConstraint>): List<SoftConstraint> =
        _minimalUnsatCore.tryGetMinimalUnsatCore(assumptions)

    private fun updateMinimalUnsatCoreModel(assumptions: List<SoftConstraint>) {
        val (model, weight) = _minimalUnsatCore.getBestModel()

        if (model != null && _upper > weight) {
            updateAssignment(model, assumptions)
        }
    }

    private fun updateAssignment(model: KModel, assumptions: List<SoftConstraint>) {
        var correctionSetSize = 0
        for (constr in assumptions) {
            if (ModelUtils.expressionIsFalse(ctx, model, constr.expression)) {
                ++correctionSetSize
            }
        }

        if (_correctionSetModel == null || correctionSetSize < _correctionSetSize) {
            _correctionSetModel = model
            _correctionSetSize = correctionSetSize
        }

        val upper = ModelUtils.getModelCost(ctx, model, softConstraints)

        if (upper > _upper) {
            return
        }

        _model = model

        _upper = upper
    }

    private fun maxResolve(weightedCore: WeightedCore, assumptions: MutableList<SoftConstraint>) = with(ctx) {
        val core = weightedCore.expressions

        require(core.isNotEmpty()) { "Core should not be empty here" }

        // d_0 := true
        // d_i := b_{i-1} and d_{i-1}    for i = 1...sz-1
        // soft (b_i or !d_i)
        //   == (b_i or !(!b_{i-1} or d_{i-1}))
        //   == (b_i or b_0 & b_1 & ... & b_{i-1})
        //
        // Soft constraint is satisfied if the previous soft constraint
        // holds or if it is the first soft constraint to fail.
        //
        // The Soundness of this rule can be established using MaxRes.

        lateinit var d: KExpr<KBoolSort>
        var dd: KExpr<KBoolSort>
        var fml: KExpr<KBoolSort>

        for (index in 1..core.lastIndex) {
            val b_i = core[index - 1]
            val b_i1 = core[index]

            when (index) {
                1 -> {
                    d = b_i
                }

                2 -> {
                    d = b_i and d
                }

                else -> {
                    dd = mkFreshConst("d", this.boolSort)
                    fml = mkImplies(dd, d)
                    assert(fml)
                    fml = mkImplies(dd, b_i)
                    assert(fml)
                    fml = d and b_i
                    // TODO: process!!!
                    // update_model(dd, fml);
                    // I use the string below instead!!!
                    assert(dd eq fml)
                    d = dd
                }
            }

            val asm = mkFreshConst("a", this.boolSort)
            val cls = b_i1 or d
            fml = mkImplies(asm, cls)
            assumptions.add(SoftConstraint(asm, weightedCore.weight))
            // TODO: process!!!
            // update_model(asm, cls);
            // I use the string below instead!!!
            assert(asm eq cls)
            assert(fml)
        }
    }

    private fun correctionSetMaxResolve(
        correctionSet: List<SoftConstraint>,
        assumptions: MutableList<SoftConstraint>,
        weight: UInt,
    ) = with(ctx) {
        if (correctionSet.isEmpty()) {
            return
        }

        var d: KExpr<KBoolSort> = falseExpr
        var fml: KExpr<KBoolSort>
        var asm: KExpr<KBoolSort>
        //
        // d_0 := false
        // d_i := b_{i-1} or d_{i-1}    for i = 1...sz-1
        // soft (b_i and d_i)
        //   == (b_i and (b_0 or b_1 or ... or b_{i-1}))
        //
        // asm => b_i
        // asm => d_{i-1} or b_{i-1}
        // d_i => d_{i-1} or b_{i-1}
        //
        for (i in 1..correctionSet.lastIndex) {
            val b_i = correctionSet[i - 1].expression
            val b_i1 = correctionSet[i].expression
            val cls = b_i or d

            if (i > 2) {
                d = mkFreshConst("d", this.boolSort)
                fml = mkImplies(d, cls)

                // TODO: process!!!
                // update_model(d, cls);
                // I use the string below instead:
                assert(d eq cls)

                assert(fml)
            } else {
                d = cls
            }

            asm = mkFreshConst("a", this.boolSort)
            fml = mkImplies(asm, b_i1)
            assert(fml)
            fml = mkImplies(asm, cls)
            assert(fml)
            assumptions.add(SoftConstraint(asm, weight))

            fml = b_i1 and cls
            // TODO: process!!!  update_model(asm, fml);
            // I use the string below instead:
            assert(asm eq fml)
        }

        fml = correctionSet
            .map { it.expression }
            .reduce { x, y -> x or y }
        assert(fml)
    }

    private fun initMaxSAT() {
        _lower = 0u
        _upper = softConstraints.sumOf { it.weight }

        _maxUpper = _upper
        _correctionSetSize = 0

        _model = null
        _correctionSetModel = null
        _minimalUnsatCore.reset()
    }

    private fun getCorrectionSet(model: KModel, assumptions: List<SoftConstraint>): List<SoftConstraint> {
        updateAssignment(model, assumptions)

        return ModelUtils.getCorrectionSet(ctx, model, assumptions)
    }

    private fun checkSat(assumptions: List<SoftConstraint>): KSolverStatus {
        val status = solver.checkWithAssumptions(assumptions.map { it.expression })

        if (status == SAT) {
            updateAssignment(solver.model().detach(), assumptions)
        }

        return status
    }
}
