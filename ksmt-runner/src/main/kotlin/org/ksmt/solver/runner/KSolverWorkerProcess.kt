package org.ksmt.solver.runner

import com.jetbrains.rd.framework.IProtocol
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.ChildProcessBase
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.generated.AssertAndTrackResult
import org.ksmt.runner.generated.CheckResult
import org.ksmt.runner.generated.ModelEntry
import org.ksmt.runner.generated.ModelFuncInterpEntry
import org.ksmt.runner.generated.ModelResult
import org.ksmt.runner.generated.ReasonUnknownResult
import org.ksmt.runner.generated.SolverProtocolModel
import org.ksmt.runner.generated.SolverType
import org.ksmt.runner.generated.UnsatCoreResult
import org.ksmt.runner.generated.solverProtocolModel
import org.ksmt.runner.serializer.AstSerializationCtx
import org.ksmt.solver.KSolver
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import kotlin.time.Duration.Companion.milliseconds

class KSolverWorkerProcess : ChildProcessBase<SolverProtocolModel>() {
    override fun IProtocol.protocolModel(): SolverProtocolModel = solverProtocolModel

    private var ctx: KContext? = null
    private var solver: KSolver? = null

    private fun ensureInitialized() {
        check(ctx != null) { "Solver is not initialized" }
    }

    override fun parseArgs(args: Array<String>) = KsmtWorkerArgs.fromList(args.toList())

    @Suppress("LongMethod")
    override fun SolverProtocolModel.setup(astSerializationCtx: AstSerializationCtx, onStop: () -> Unit) {
        initSolver.measureExecutionForTermination { params ->
            check(ctx == null) { "Solver is initialized" }
            ctx = KContext()
            astSerializationCtx.initCtx(ctx!!)
            solver = when (params.type) {
                SolverType.Z3 -> KZ3Solver(ctx!!)
                SolverType.Bitwuzla -> KBitwuzlaSolver(ctx!!)
            }
        }
        deleteSolver.measureExecutionForTermination {
            ensureInitialized()
            solver?.close()
            ctx?.close()
            astSerializationCtx.resetCtx()
            solver = null
            ctx = null
        }
        assert.measureExecutionForTermination { params ->
            ensureInitialized()

            @Suppress("UNCHECKED_CAST")
            solver!!.assert(params.expression as KExpr<KBoolSort>)
        }
        assertAndTrack.measureExecutionForTermination { params ->
            ensureInitialized()

            @Suppress("UNCHECKED_CAST")
            val track = solver!!.assertAndTrack(params.expression as KExpr<KBoolSort>)

            AssertAndTrackResult(track)
        }
        push.measureExecutionForTermination {
            ensureInitialized()
            solver!!.push()
        }
        pop.measureExecutionForTermination { params ->
            ensureInitialized()
            solver!!.pop(params.levels)
        }
        check.measureExecutionForTermination { params ->
            ensureInitialized()
            val timeout = params.timeout.milliseconds
            val status = solver!!.check(timeout)
            CheckResult(status)
        }
        checkWithAssumptions.measureExecutionForTermination { params ->
            ensureInitialized()
            val timeout = params.timeout.milliseconds

            @Suppress("UNCHECKED_CAST")
            val status = solver!!.checkWithAssumptions(params.assumptions as List<KExpr<KBoolSort>>, timeout)

            CheckResult(status)
        }
        model.measureExecutionForTermination {
            ensureInitialized()
            val model = solver!!.model().detach()
            val declarations = model.declarations.toList()
            val interpretations = declarations.map {
                val interp = model.interpretation(it) ?: error("No interpretation for model declaration $it")
                val interpEntries = interp.entries.map { ModelFuncInterpEntry(it.args, it.value) }
                ModelEntry(interp.sort, interp.vars, interpEntries, interp.default)
            }
            ModelResult(declarations, interpretations)
        }
        unsatCore.measureExecutionForTermination {
            ensureInitialized()
            val core = solver!!.unsatCore()
            UnsatCoreResult(core)
        }
        reasonOfUnknown.measureExecutionForTermination {
            ensureInitialized()
            val reason = solver!!.reasonOfUnknown()
            ReasonUnknownResult(reason)
        }
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            KSolverWorkerProcess().start(args)
        }
    }
}
