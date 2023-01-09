package org.ksmt.solver.runner

import com.jetbrains.rd.framework.IProtocol
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.ChildProcessBase
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.models.generated.CheckResult
import org.ksmt.runner.models.generated.ModelEntry
import org.ksmt.runner.models.generated.ModelFuncInterpEntry
import org.ksmt.runner.models.generated.ModelResult
import org.ksmt.runner.models.generated.ModelUninterpretedSortUniverse
import org.ksmt.runner.models.generated.ReasonUnknownResult
import org.ksmt.runner.models.generated.SolverProtocolModel
import org.ksmt.runner.models.generated.SolverType
import org.ksmt.runner.models.generated.UnsatCoreResult
import org.ksmt.runner.models.generated.solverProtocolModel
import org.ksmt.runner.serializer.AstSerializationCtx
import org.ksmt.solver.KSolver
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.bitwuzla.KBitwuzlaSolverConfiguration
import org.ksmt.solver.yices.KYicesSolver
import org.ksmt.solver.yices.KYicesSolverConfiguration
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.solver.z3.KZ3SolverConfiguration
import org.ksmt.sort.KBoolSort
import kotlin.time.Duration.Companion.milliseconds

class KSolverWorkerProcess : ChildProcessBase<SolverProtocolModel>() {
    private var workerCtx: KContext? = null
    private var workerSolver: KSolver<*>? = null

    private val ctx: KContext
        get() = workerCtx ?: error("Solver is not initialized")

    private val solver: KSolver<*>
        get() = workerSolver ?: error("Solver is not initialized")

    override fun parseArgs(args: Array<String>) = KsmtWorkerArgs.fromList(args.toList())

    override fun initProtocolModel(protocol: IProtocol): SolverProtocolModel =
        protocol.solverProtocolModel

    @Suppress("LongMethod")
    override fun SolverProtocolModel.setup(astSerializationCtx: AstSerializationCtx) {
        initSolver.measureExecutionForTermination { params ->
            check(workerCtx == null) { "Solver is initialized" }
            workerCtx = KContext()
            astSerializationCtx.initCtx(ctx)
            workerSolver = when (params.type) {
                SolverType.Z3 -> KZ3Solver(ctx)
                SolverType.Bitwuzla -> KBitwuzlaSolver(ctx)
                SolverType.Yices -> KYicesSolver(ctx)
            }
        }
        deleteSolver.measureExecutionForTermination {
            solver.close()
            ctx.close()
            astSerializationCtx.resetCtx()
            workerSolver = null
            workerCtx = null
        }
        configure.measureExecutionForTermination { config ->
            solver.configure {
                when (this) {
                    is KZ3SolverConfiguration -> config.forEach { addUniversalParam(it) }
                    is KBitwuzlaSolverConfiguration -> config.forEach { addUniversalParam(it) }
                    is KYicesSolverConfiguration -> config.forEach { addUniversalParam(it) }
                    else -> error("Unexpected configuration: ${this::class}")
                }
            }
        }
        assert.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            solver.assert(params.expression as KExpr<KBoolSort>)
        }
        assertAndTrack.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            solver.assertAndTrack(params.expression as KExpr<KBoolSort>, params.trackVar as KConstDecl<KBoolSort>)
        }
        push.measureExecutionForTermination {
            solver.push()
        }
        pop.measureExecutionForTermination { params ->
            solver.pop(params.levels)
        }
        check.measureExecutionForTermination { params ->
            val timeout = params.timeout.milliseconds
            val status = solver.check(timeout)
            CheckResult(status)
        }
        checkWithAssumptions.measureExecutionForTermination { params ->
            val timeout = params.timeout.milliseconds

            @Suppress("UNCHECKED_CAST")
            val status = solver.checkWithAssumptions(params.assumptions as List<KExpr<KBoolSort>>, timeout)

            CheckResult(status)
        }
        model.measureExecutionForTermination {
            val model = solver.model().detach()
            val declarations = model.declarations.toList()
            val interpretations = declarations.map {
                val interp = model.interpretation(it) ?: error("No interpretation for model declaration $it")
                val interpEntries = interp.entries.map { ModelFuncInterpEntry(it.args, it.value) }
                ModelEntry(interp.decl, interp.vars, interpEntries, interp.default)
            }
            val uninterpretedSortUniverse = model.uninterpretedSorts.map { sort ->
                val universe = model.uninterpretedSortUniverse(sort)
                    ?: error("No universe for uninterpreted sort $it")
                ModelUninterpretedSortUniverse(sort, universe.toList())
            }
            ModelResult(declarations, interpretations, uninterpretedSortUniverse)
        }
        unsatCore.measureExecutionForTermination {
            val core = solver.unsatCore()
            UnsatCoreResult(core)
        }
        reasonOfUnknown.measureExecutionForTermination {
            val reason = solver.reasonOfUnknown()
            ReasonUnknownResult(reason)
        }
        interrupt.measureExecutionForTermination {
            solver.interrupt()
        }
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            KSolverWorkerProcess().start(args)
        }
    }
}
