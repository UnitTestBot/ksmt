package org.ksmt.runner.models

import com.jetbrains.rd.generator.nova.Ext
import com.jetbrains.rd.generator.nova.PredefinedType
import com.jetbrains.rd.generator.nova.async
import com.jetbrains.rd.generator.nova.call
import com.jetbrains.rd.generator.nova.field
import com.jetbrains.rd.generator.nova.immutableList
import com.jetbrains.rd.generator.nova.nullable

@Suppress("unused")
object SolverProtocolModel : Ext(ProtocolRoot) {
    private val kastType = kastType()
    private val statusType = solverStatusType()

    private val createSolverParams = structdef {
        field("type", enum("SolverType") {
            +"Z3"
            +"Bitwuzla"
        })
    }

    private val assertParams = structdef {
        field("expression", kastType)
    }

    private val assertAndTrackResult = structdef {
        field("expression", kastType)
    }

    private val popParams = structdef {
        field("levels", PredefinedType.uint)
    }

    private val checkParams = structdef {
        field("timeout", PredefinedType.long)
    }

    private val checkResult = structdef {
        field("status", statusType)
    }

    private val checkWithAssumptionsParams = structdef {
        field("assumptions", immutableList(kastType))
        field("timeout", PredefinedType.long)
    }

    private val unsatCoreResult = structdef {
        field("core", immutableList(kastType))
    }

    private val reasonUnknownResult = structdef {
        field("reasonUnknown", PredefinedType.string)
    }

    private val modelFuncInterpEntry = structdef {
        field("args", immutableList(kastType))
        field("value", kastType)
    }

    private val modelEntry = structdef {
        field("sort", kastType)
        field("vars", immutableList(kastType))
        field("entries", immutableList(modelFuncInterpEntry))
        field("default", kastType.nullable)
    }

    private val modelResult = structdef {
        field("declarations", immutableList(kastType))
        field("interpretations", immutableList(modelEntry))
    }

    init {
        call("initSolver", createSolverParams, PredefinedType.void).apply {
            async
            documentation = "Initialize solver"
        }
        call("deleteSolver", PredefinedType.void, PredefinedType.void).apply {
            async
            documentation = "Delete solver"
        }
        call("assert", assertParams, PredefinedType.void).apply {
            async
            documentation = "Assert expression"
        }
        call("assertAndTrack", assertParams, assertAndTrackResult).apply {
            async
            documentation = "Assert and track expression"
        }
        call("push", PredefinedType.void, PredefinedType.void).apply {
            async
            documentation = "Solver push"
        }
        call("pop", popParams, PredefinedType.void).apply {
            async
            documentation = "Solver pop"
        }
        call("check", checkParams, checkResult).apply {
            async
            documentation = "Check SAT"
        }
        call("checkWithAssumptions", checkWithAssumptionsParams, checkResult).apply {
            async
            documentation = "Check SAT with assumptions"
        }
        call("model", PredefinedType.void, modelResult).apply {
            async
            documentation = "Get model"
        }
        call("unsatCore", PredefinedType.void, unsatCoreResult).apply {
            async
            documentation = "Get unsat core"
        }
        call("reasonOfUnknown", PredefinedType.void, reasonUnknownResult).apply {
            async
            documentation = "Get reason of unknown"
        }
    }
}