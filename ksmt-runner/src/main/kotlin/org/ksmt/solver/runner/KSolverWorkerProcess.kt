package org.ksmt.solver.runner

import com.jetbrains.rd.framework.IProtocol
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.ChildProcessBase
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.generated.createInstance
import org.ksmt.runner.generated.createSolverConstructor
import org.ksmt.runner.generated.models.CheckResult
import org.ksmt.runner.generated.models.ContextSimplificationMode
import org.ksmt.runner.generated.models.ModelEntry
import org.ksmt.runner.generated.models.ModelFuncInterpEntry
import org.ksmt.runner.generated.models.ModelResult
import org.ksmt.runner.generated.models.ModelUninterpretedSortUniverse
import org.ksmt.runner.generated.models.ReasonUnknownResult
import org.ksmt.runner.generated.models.SolverProtocolModel
import org.ksmt.runner.generated.models.SolverType
import org.ksmt.runner.generated.models.UnsatCoreResult
import org.ksmt.runner.generated.models.solverProtocolModel
import org.ksmt.runner.serializer.AstSerializationCtx
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.sort.KBoolSort
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

    private fun serializeFunctionInterpretation(interp: KModel.KFuncInterp<*>): ModelEntry {
        val interpEntries = interp.entries.map { serializeFunctionInterpretationEntry(it) }
        val interpVars = if (interp is KModel.KFuncInterpEntryWithVars<*>) interp.vars else null
        return ModelEntry(interp.decl, interpVars, interpEntries, interp.default)
    }

    private fun serializeFunctionInterpretationEntry(entry: KModel.KFuncInterpEntry<*>) =
        ModelFuncInterpEntry(
            hasVars = entry is KModel.KFuncInterpEntryWithVars<*>,
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
