package io.ksmt.solver.runner

import com.jetbrains.rd.framework.IProtocol
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.runner.core.ChildProcessBase
import io.ksmt.runner.core.KsmtWorkerArgs
import io.ksmt.runner.generated.createInstance
import io.ksmt.runner.generated.createSolverConstructor
import io.ksmt.runner.generated.models.CheckResult
import io.ksmt.runner.generated.models.ContextSimplificationMode
import io.ksmt.runner.generated.models.ModelEntry
import io.ksmt.runner.generated.models.ModelFuncInterpEntry
import io.ksmt.runner.generated.models.ModelResult
import io.ksmt.runner.generated.models.ModelUninterpretedSortUniverse
import io.ksmt.runner.generated.models.ReasonUnknownResult
import io.ksmt.runner.generated.models.SolverProtocolModel
import io.ksmt.runner.generated.models.SolverType
import io.ksmt.runner.generated.models.UnsatCoreResult
import io.ksmt.runner.generated.models.solverProtocolModel
import io.ksmt.runner.serializer.AstSerializationCtx
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpEntry
import io.ksmt.solver.model.KFuncInterpEntryWithVars
import io.ksmt.solver.KSolver
import io.ksmt.solver.model.KFuncInterpWithVars
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration.Companion.milliseconds

class KSolverWorkerProcess : ChildProcessBase<SolverProtocolModel>() {
    private var workerCtx: KContext? = null
    private var workerSolver: KSolver<*>? = null
    private val customSolverCreators = hashMapOf<String, (KContext) -> KSolver<*>>()

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

            val simplificationMode = when (params.contextSimplificationMode) {
                ContextSimplificationMode.SIMPLIFY -> KContext.SimplificationMode.SIMPLIFY
                ContextSimplificationMode.NO_SIMPLIFY -> KContext.SimplificationMode.NO_SIMPLIFY
            }
            workerCtx = KContext(simplificationMode = simplificationMode)

            astSerializationCtx.initCtx(ctx)

            workerSolver = if (params.type != SolverType.Custom) {
                params.type.createInstance(ctx)
            } else {
                val solverName = params.customSolverQualifiedName
                    ?: error("Custom solver name was not provided")

                val solverCreator = customSolverCreators.getOrPut(solverName) {
                    createSolverConstructor(solverName)
                }

                solverCreator(ctx)
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
                config.forEach { addUniversalParam(it) }
            }
        }
        assert.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            solver.assert(params.expression as KExpr<KBoolSort>)
        }
        bulkAssert.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            solver.assert(params.expressions as List<KExpr<KBoolSort>>)
        }
        assertAndTrack.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            solver.assertAndTrack(params.expression as KExpr<KBoolSort>)
        }
        bulkAssertAndTrack.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            solver.assertAndTrack(params.expressions as List<KExpr<KBoolSort>>)
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
                serializeFunctionInterpretation(interp)
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

    private fun serializeFunctionInterpretation(interp: KFuncInterp<*>): ModelEntry {
        val interpEntries = interp.entries.map { serializeFunctionInterpretationEntry(it) }
        val interpVars = if (interp is KFuncInterpWithVars) interp.vars else null
        return ModelEntry(interp.decl, interpVars, interpEntries, interp.default)
    }

    private fun serializeFunctionInterpretationEntry(entry: KFuncInterpEntry<*>) =
        ModelFuncInterpEntry(
            hasVars = entry is KFuncInterpEntryWithVars<*>,
            args = entry.args,
            value = entry.value
        )

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            KSolverWorkerProcess().start(args)
        }
    }
}
