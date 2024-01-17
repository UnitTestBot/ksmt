package io.ksmt.solver.maxsmt.solvers

import io.github.oshai.kotlinlogging.KotlinLogging
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.KSolverStatus.UNKNOWN
import io.ksmt.solver.KSolverStatus.UNSAT
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.KMaxSMTContext.Strategy.PrimalDualMaxRes
import io.ksmt.solver.maxsmt.KMaxSMTContext.Strategy.PrimalMaxRes
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.solvers.utils.MinimalUnsatCore
import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.solver.maxsmt.utils.CoreUtils
import io.ksmt.solver.maxsmt.utils.ModelUtils
import io.ksmt.solver.maxsmt.utils.TimerUtils
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration
import kotlin.time.TimeSource.Monotonic.markNow

class KPrimalDualMaxResSolver<T : KSolverConfiguration>(
    private val ctx: KContext,
    private val solver: KSolver<out T>,
    private val maxSmtCtx: KMaxSMTContext,
) :
    KMaxResSolver<T>(ctx, solver) {
    private var _lower: UInt = 0u // Current lower frontier
    private var _upper: UInt = 0u // Current upper frontier
    private var _maxUpper = 0u // Max possible upper frontier
    private var _correctionSetSize: Int = 0 // Current corrections set size
    private val _maxCoreSize = if (maxSmtCtx.getMultipleCores) 3 else 1
    private var _correctionSetModel: KModel? = null
    private var _model: KModel? = null
    private var _minimalUnsatCore = MinimalUnsatCore(ctx, solver)
    private var collectStatistics = false
    private var _iteration = 0
    private val logger = KotlinLogging.logger {}
    private var markLoggingPoint = markNow()

    private data class WeightedCore(val expressions: List<KExpr<KBoolSort>>, val weight: UInt)

    override fun checkMaxSMT(timeout: Duration, collectStatistics: Boolean): KMaxSMTResult {
        val markCheckMaxSMTStart = markNow()
        markLoggingPoint = markCheckMaxSMTStart

        if (TimerUtils.timeoutExceeded(timeout)) {
            error("Timeout must be positive but was [${timeout.inWholeSeconds} s]")
        }

        this.collectStatistics = collectStatistics

        if (this.collectStatistics) {
            maxSMTStatistics = KMaxSMTStatistics(maxSmtCtx)
            maxSMTStatistics.timeoutMs = timeout.inWholeMilliseconds
        }

        val markHardConstraintsCheckStart = markNow()
        val hardConstraintsStatus = solver.check(timeout)
        if (this.collectStatistics) {
            maxSMTStatistics.queriesToSolverNumber++
            maxSMTStatistics.timeInSolverQueriesMs += markHardConstraintsCheckStart.elapsedNow().inWholeMilliseconds
        }

        if (hardConstraintsStatus == UNSAT || softConstraints.isEmpty()) {
            if (collectStatistics) {
                maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
            }
            return KMaxSMTResult(listOf(), hardConstraintsStatus, true)
        } else if (hardConstraintsStatus == UNKNOWN) {
            if (collectStatistics) {
                maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
            }
            return KMaxSMTResult(listOf(), hardConstraintsStatus, false)
        }

        solver.push()
        initMaxSMT()

        val assumptions = softConstraints.toMutableList()

        while (_lower < _upper) {
            logger.info {
                "[${markLoggingPoint.elapsedNow().inWholeMicroseconds} mcs] Iteration number: $_iteration"
            }
            markLoggingPoint = markNow()

            val softConstraintsCheckRemainingTime = TimerUtils.computeRemainingTime(timeout, markCheckMaxSMTStart)
            if (TimerUtils.timeoutExceeded(softConstraintsCheckRemainingTime)) {
                if (collectStatistics) {
                    maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
                }
                throw NotImplementedError()
            }

            val status = checkSatHillClimb(assumptions, timeout)

            when (status) {
                SAT -> {
                    when (maxSmtCtx.strategy) {
                        PrimalMaxRes -> _upper = _lower

                        PrimalDualMaxRes -> {
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
                    }
                }

                UNSAT -> {
                    val remainingTime = TimerUtils.computeRemainingTime(timeout, markCheckMaxSMTStart)
                    if (TimerUtils.timeoutExceeded(remainingTime)) {
                        solver.pop()
                        if (collectStatistics) {
                            maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
                        }
                        throw NotImplementedError()
                    }

                    val processUnsatStatus = processUnsat(assumptions, remainingTime)
                    if (processUnsatStatus == UNSAT) {
                        _lower = _upper
                        // TODO: process this case as it can happen when timeout exceeded
                        solver.pop()
                        if (collectStatistics) {
                            maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
                        }
                        throw NotImplementedError()
                    } else if (processUnsatStatus == UNKNOWN) {
                        solver.pop()
                        if (collectStatistics) {
                            maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
                        }
                        throw NotImplementedError()
                    }
                }

                UNKNOWN -> {
                    solver.pop()
                    if (collectStatistics) {
                        maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
                    }
                    throw NotImplementedError()
                }
            }

            ++_iteration
        }

        _lower = _upper

        val result = KMaxSMTResult(getSatSoftConstraintsByModel(_model!!), SAT, true)
        logger.info {
            "[${markLoggingPoint.elapsedNow().inWholeMicroseconds} mcs] --- returning MaxSMT result"
        }

        solver.pop()

        if (collectStatistics) {
            maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
        }
        return result
    }

    private fun processSat(correctionSet: List<SoftConstraint>, assumptions: MutableList<SoftConstraint>) {
        removeCoreAssumptions(correctionSet, assumptions)
        val (minWeight, _) = splitCore(correctionSet, assumptions)
        correctionSetMaxResolve(correctionSet, assumptions, minWeight)

        _correctionSetModel = null
        _correctionSetSize = 0
    }

    private fun processUnsat(assumptions: MutableList<SoftConstraint>, timeout: Duration): KSolverStatus {
        val (status, cores) = getCores(assumptions, timeout)

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
        logger.info { "processing unsat core --- started" }

        val core = weightedCore.expressions

        require(core.isNotEmpty()) { "Core should not be empty here" }

        maxResolve(weightedCore, assumptions)

        val fml = !(core.reduce { x, y -> x and y })
        assert(fml)

        _lower += weightedCore.weight

        if (maxSmtCtx.strategy == PrimalDualMaxRes) {
            _lower = minOf(_lower, _upper)
        }

        if (_correctionSetModel != null && _correctionSetSize > 0) {
            // This estimate can overshoot for weighted soft constraints.
            --_correctionSetSize
        }

        // Here we also prefer a smaller correction set to core.
        if (_correctionSetModel != null && _correctionSetSize < core.size) {
            val correctionSet = getCorrectionSet(_correctionSetModel!!, assumptions)
            if (correctionSet.size >= core.size) {
                logger.info { "processing unsat core --- ended" }
                return
            }

            var weight = 0u
            for (asm in assumptions) {
                val weight1 = asm.weight
                if (weight != 0u && weight1 != weight) {
                    logger.info { "processing unsat core --- ended" }
                    return
                }

                weight = weight1
            }

            processSat(correctionSet, assumptions)
        }

        logger.info { "processing unsat core --- ended" }
    }

    private fun getCores(
        assumptions: MutableList<SoftConstraint>,
        timeout: Duration,
    ): Pair<KSolverStatus, List<WeightedCore>> {
        val markStart = markNow()

        val cores = mutableListOf<WeightedCore>()
        var status = UNSAT

        while (status == UNSAT) {
            var unsatCore: List<SoftConstraint>

            if (maxSmtCtx.minimizeCores) {
                val minimizeCoreRemainingTime = TimerUtils.computeRemainingTime(timeout, markStart)
                if (TimerUtils.timeoutExceeded(minimizeCoreRemainingTime)) {
                    return Pair(SAT, cores) // TODO: is this status Ok?
                }

                unsatCore = minimizeCore(assumptions, minimizeCoreRemainingTime)
                updateMinimalUnsatCoreModel(assumptions)
            } else {
                unsatCore = CoreUtils.coreToSoftConstraints(solver.unsatCore(), assumptions)
            }

            if (unsatCore.isEmpty()) {
                cores.clear()
                _lower = _upper
                return Pair(SAT, cores)
            }

            // 1. remove all core literals from assumptions
            // 2. re-add literals of higher weight than min-weight.
            // 3. 'core' stores the core literals that are re-encoded as assumptions afterward
            cores.add(WeightedCore(unsatCore.map { it.expression }, CoreUtils.getCoreWeight(unsatCore)))

            removeCoreAssumptions(unsatCore, assumptions)
            splitCore(unsatCore, assumptions)

            if (unsatCore.size >= _maxCoreSize) {
                return Pair(SAT, cores)
            }

            val checkSatRemainingTime = TimerUtils.computeRemainingTime(timeout, markStart)
            if (TimerUtils.timeoutExceeded(checkSatRemainingTime)) {
                return Pair(SAT, cores) // TODO: is this status Ok?
            }

            status = checkSatHillClimb(assumptions, checkSatRemainingTime)
        }

        return Pair(status, cores)
    }

    private fun minimizeCore(assumptions: List<SoftConstraint>, timeout: Duration): List<SoftConstraint> {
        val minimalUnsatCore = _minimalUnsatCore.tryGetMinimalUnsatCore(assumptions, timeout, collectStatistics)
        if (collectStatistics) {
            val minimalCoreStatistics = _minimalUnsatCore.collectStatistics()
            maxSMTStatistics.queriesToSolverNumber += minimalCoreStatistics.queriesToSolverNumber
            maxSMTStatistics.timeInSolverQueriesMs += minimalCoreStatistics.timeInSolverQueriesMs
        }

        return minimalUnsatCore
    }

    private fun updateMinimalUnsatCoreModel(assumptions: List<SoftConstraint>) {
        val (model, weight) = _minimalUnsatCore.getBestModel()

        if (model != null && _upper > weight) {
            updateAssignment(model, assumptions)
        }
    }

    private fun updateAssignment(model: KModel, assumptions: List<SoftConstraint>) {
        var correctionSetSize = 0
        for (constr in assumptions) {
            if (ModelUtils.expressionIsNotTrue(ctx, model, constr.expression)) {
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

        logger.info {
            "[${markLoggingPoint.elapsedNow().inWholeMicroseconds} mcs] (lower bound: $_lower, upper bound: $_upper) --- model is updated"
        }
        markLoggingPoint = markNow()
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

    private fun initMaxSMT() {
        _lower = 0u
        _upper = softConstraints.sumOf { it.weight }

        _maxUpper = _upper
        _correctionSetSize = 0

        _iteration = 0

        logger.info {
            "[${markLoggingPoint.elapsedNow().inWholeMicroseconds} mcs] (lower bound: $_lower, upper bound: $_upper) --- model is initialized with null"
        }
        markLoggingPoint = markNow()

        _model = null
        _correctionSetModel = null
        _minimalUnsatCore.reset()
    }

    private fun getCorrectionSet(model: KModel, assumptions: List<SoftConstraint>): List<SoftConstraint> {
        updateAssignment(model, assumptions)

        return ModelUtils.getCorrectionSet(ctx, model, assumptions)
    }

    private fun checkSatHillClimb(assumptions: MutableList<SoftConstraint>, timeout: Duration): KSolverStatus {
        logger.info { "checking formula on satisfiability --- started" }

        var status = SAT

        if (maxSmtCtx.preferLargeWeightConstraintsForCores && assumptions.isNotEmpty()) {
            val markStart = markNow()

            // Give preference to cores that have large minimal values.
            assumptions.sortByDescending { it.weight }

            var lastIndex = 0
            var index = 0

            while (index < assumptions.size && status == SAT) {
                while (assumptions.size > 20 * (index - lastIndex) && index < assumptions.size) {
                    index = getNextIndex(assumptions, index)
                }
                lastIndex = index

                val assumptionsToCheck = assumptions.subList(0, index)

                val remainingTime = TimerUtils.computeRemainingTime(timeout, markStart)
                if (TimerUtils.timeoutExceeded(remainingTime)) {
                    logger.info { "checking formula on satisfiability --- ended --- solver returned UNKNOWN" }
                    return UNKNOWN
                }

                val markCheckAssumptionsStart = markNow()
                status = checkSat(assumptionsToCheck, assumptionsToCheck.size == assumptions.size, remainingTime)
                if (collectStatistics) {
                    maxSMTStatistics.queriesToSolverNumber++
                    maxSMTStatistics.timeInSolverQueriesMs += markCheckAssumptionsStart.elapsedNow().inWholeMilliseconds
                }
            }
        } else {
            val markCheckStart = markNow()
            status = checkSat(assumptions, true, timeout)
            if (collectStatistics) {
                maxSMTStatistics.queriesToSolverNumber++
                maxSMTStatistics.timeInSolverQueriesMs += markCheckStart.elapsedNow().inWholeMilliseconds
            }
        }

        logger.info { "checking formula on satisfiability --- ended" }
        return status
    }

    private fun getNextIndex(assumptions: List<SoftConstraint>, index: Int): Int {
        var currentIndex = index

        if (currentIndex < assumptions.size) {
            val weight = assumptions[currentIndex].weight
            ++currentIndex
            while (currentIndex < assumptions.size && weight == assumptions[currentIndex].weight) {
                ++currentIndex
            }
        }
        return currentIndex
    }

    private fun checkSat(
        assumptions: List<SoftConstraint>,
        passedAllAssumptions: Boolean,
        timeout: Duration,
    ): KSolverStatus {
        val status = solver.checkWithAssumptions(assumptions.map { it.expression }, timeout)

        if (passedAllAssumptions && status == SAT) {
            updateAssignment(solver.model().detach(), assumptions)
        }

        return status
    }
}
