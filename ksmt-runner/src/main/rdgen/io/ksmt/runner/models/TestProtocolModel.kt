package io.ksmt.runner.models

import com.jetbrains.rd.generator.nova.Root
import com.jetbrains.rd.generator.nova.Ext
import com.jetbrains.rd.generator.nova.PredefinedType
import com.jetbrains.rd.generator.nova.async
import com.jetbrains.rd.generator.nova.call
import com.jetbrains.rd.generator.nova.field
import com.jetbrains.rd.generator.nova.immutableList
import com.jetbrains.rd.generator.nova.nullable

object TestProtocolRoot : Root()

@Suppress("unused")
object TestProtocolModel : Ext(TestProtocolRoot) {
    private val kastType = kastType()
    private val statusType = solverStatusType()

    private val softConstraint = structdef {
        field("expression", kastType)
        field("weight", PredefinedType.uint)
    }

    private val equalityCheckParams = structdef {
        field("solver", PredefinedType.int)
        field("actual", kastType)
        field("expected", PredefinedType.long)
    }

    private val equalityCheckAssumptionsParams = structdef {
        field("solver", PredefinedType.int)
        field("assumption", kastType)
    }

    private val testAssertParams = structdef {
        field("solver", PredefinedType.int)
        field("expr", PredefinedType.long)
    }

    private val testCheckResult = structdef {
        field("status", statusType)
    }

    private val testCheckMaxSMTParams = structdef {
        field("timeout", PredefinedType.long)
        field("collectStatistics", PredefinedType.bool)
    }

    private val testCheckMaxSMTResult = structdef {
        field("satSoftConstraints", immutableList(softConstraint))
        field("hardConstraintsSatStatus", statusType)
        field("maxSMTSucceeded", PredefinedType.bool)
    }

    private val testCollectMaxSMTStatisticsResult = structdef {
        field("timeoutMs", PredefinedType.long)
        field("elapsedTimeMs", PredefinedType.long)
        field("timeInSolverQueriesMs", PredefinedType.long)
        field("queriesToSolverNumber", PredefinedType.int)
    }

    private val testConversionResult = structdef {
        field("expressions", immutableList(kastType))
    }

    private val testInternalizeAndConvertParams = structdef {
        field("expressions", immutableList(kastType))
    }

    init {
        call("create", PredefinedType.void, PredefinedType.void).apply {
            async
            documentation = "Create context"
        }
        call("delete", PredefinedType.void, PredefinedType.void).apply {
            async
            documentation = "Delete context"
        }
        call("parseFile", PredefinedType.string, immutableList(PredefinedType.long)).apply {
            async
            documentation = "Parse smt-lib2 file"
        }
        call("convertAssertions", immutableList(PredefinedType.long), testConversionResult).apply {
            async
            documentation = "Convert native solver expression into KSMT"
        }
        call("internalizeAndConvertBitwuzla", testInternalizeAndConvertParams, testConversionResult).apply {
            async
            documentation = "Internalize and convert expressions using Bitwuzla converter/internalizer"
        }
        call("internalizeAndConvertYices", testInternalizeAndConvertParams, testConversionResult).apply {
            async
            documentation = "Internalize and convert expressions using Yices converter/internalizer"
        }
        call("internalizeAndConvertCvc5", testInternalizeAndConvertParams, testConversionResult).apply {
            async
            documentation = "Internalize and convert expressions using cvc5 converter/internalizer"
        }
        call("createSolver", PredefinedType.int, PredefinedType.int).apply {
            async
            documentation = "Create solver"
        }
        call("assert", testAssertParams, PredefinedType.void).apply {
            async
            documentation = "Assert expr"
        }
        call("assertSoft", softConstraint, PredefinedType.void).apply {
            async
            documentation = "Assert expression softly"
        }
        call("check", PredefinedType.int, testCheckResult).apply {
            async
            documentation = "Check-sat"
        }
        call("checkMaxSMT", testCheckMaxSMTParams, testCheckMaxSMTResult).apply {
            async
            documentation = "Check MaxSMT"
        }
        call("checkSubOptMaxSMT", testCheckMaxSMTParams, testCheckMaxSMTResult).apply {
            async
            documentation = "Check SubOptMaxSMT"
        }
        call("collectMaxSMTStatistics", PredefinedType.void, testCollectMaxSMTStatisticsResult).apply {
            documentation = "Collect MaxSMT statistics"
        }
        call("exprToString", PredefinedType.long, PredefinedType.string).apply {
            async
            documentation = "Expression to string"
        }
        call("getReasonUnknown", PredefinedType.int, PredefinedType.string).apply {
            async
            documentation = "Get reason unknown"
        }
        call("addEqualityCheck", equalityCheckParams, PredefinedType.void).apply {
            async
            documentation = "Add equality check"
        }
        call("addEqualityCheckAssumption", equalityCheckAssumptionsParams, PredefinedType.void).apply {
            async
            documentation = "Add assumptions for the subsequent equality check"
        }
        call("checkEqualities", PredefinedType.int, testCheckResult).apply {
            async
            documentation = "Check added equalities"
        }
        call("findFirstFailedEquality", PredefinedType.int, PredefinedType.int.nullable).apply {
            async
            documentation = "Find first failed equality check"
        }
        call("mkTrueExpr", PredefinedType.void, PredefinedType.long).apply {
            async
            documentation = "Create true expression"
        }
    }
}