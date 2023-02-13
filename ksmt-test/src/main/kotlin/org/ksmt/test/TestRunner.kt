package org.ksmt.test

import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.withTimeout
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.KsmtWorkerSession
import org.ksmt.runner.models.generated.EqualityCheckAssumptionsParams
import org.ksmt.runner.models.generated.EqualityCheckParams
import org.ksmt.runner.models.generated.TestAssertParams
import org.ksmt.runner.models.generated.TestInternalizeAndConvertParams
import org.ksmt.runner.models.generated.TestProtocolModel
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.sort.KBoolSort
import java.nio.file.Path
import kotlin.time.Duration
import kotlin.time.DurationUnit

class TestRunner(
    val ctx: KContext,
    private val hardTimeout: Duration,
    private val worker: KsmtWorkerSession<TestProtocolModel>,
) {
    suspend fun init() = withTimeoutAndExceptionHandling {
        worker.protocolModel.create.startSuspending(worker.lifetime, Unit)
    }

    suspend fun delete() = withTimeoutAndExceptionHandling {
        if (!worker.isAlive) return@withTimeoutAndExceptionHandling
        worker.protocolModel.delete.startSuspending(worker.lifetime, Unit)
    }

    suspend fun parseFile(path: Path): List<Long> = withTimeoutAndExceptionHandling {
        worker.protocolModel.parseFile.startSuspending(worker.lifetime, path.toFile().absolutePath)
    }

    suspend fun convertAssertions(nativeAssertions: List<Long>): List<KExpr<KBoolSort>> =
        withTimeoutAndExceptionHandling {
            val result = worker.protocolModel.convertAssertions.startSuspending(worker.lifetime, nativeAssertions)
            result.expressions.map {
                @Suppress("UNCHECKED_CAST")
                it as KExpr<KBoolSort>
            }
        }

    suspend fun internalizeAndConvertBitwuzla(assertions: List<KExpr<KBoolSort>>): List<KExpr<KBoolSort>> =
        withTimeoutAndExceptionHandling {
            val params = TestInternalizeAndConvertParams(assertions)
            val result = worker.protocolModel.internalizeAndConvertBitwuzla.startSuspending(worker.lifetime, params)
            result.expressions.map {
                @Suppress("UNCHECKED_CAST")
                it as KExpr<KBoolSort>
            }
        }

    suspend fun internalizeAndConvertYices(assertions: List<KExpr<KBoolSort>>): List<KExpr<KBoolSort>> =
        withTimeoutAndExceptionHandling {
            val params = TestInternalizeAndConvertParams(assertions)
            val result = worker.protocolModel.internalizeAndConvertYices.startSuspending(worker.lifetime, params)
            result.expressions.map {
                @Suppress("UNCHECKED_CAST")
                it as KExpr<KBoolSort>
            }
        }

    suspend fun createSolver(timeout: Duration): Int = withTimeoutAndExceptionHandling {
        val timeoutValue = timeout.toInt(DurationUnit.MILLISECONDS)
        worker.protocolModel.createSolver.startSuspending(worker.lifetime, timeoutValue)
    }

    suspend fun assert(solver: Int, expr: Long) = withTimeoutAndExceptionHandling {
        worker.protocolModel.assert.startSuspending(worker.lifetime, TestAssertParams(solver, expr))
    }

    suspend fun check(solver: Int): KSolverStatus = withTimeoutAndExceptionHandling {
        worker.protocolModel.check.startSuspending(worker.lifetime, solver).status
    }

    suspend fun addEqualityCheck(solver: Int, actual: KExpr<*>, expected: Long) = withTimeoutAndExceptionHandling {
        val params = EqualityCheckParams(solver, actual, expected)
        worker.protocolModel.addEqualityCheck.startSuspending(worker.lifetime, params)
    }

    suspend fun addEqualityCheckAssumption(solver: Int, assumption: KExpr<KBoolSort>) =
        withTimeoutAndExceptionHandling {
            val params = EqualityCheckAssumptionsParams(solver, assumption)
            worker.protocolModel.addEqualityCheckAssumption.startSuspending(worker.lifetime, params)
        }

    suspend fun checkEqualities(solver: Int): KSolverStatus = withTimeoutAndExceptionHandling {
        worker.protocolModel.checkEqualities.startSuspending(worker.lifetime, solver).status
    }

    suspend fun findFirstFailedEquality(solver: Int): Int? = withTimeoutAndExceptionHandling {
        worker.protocolModel.findFirstFailedEquality.startSuspending(worker.lifetime, solver)
    }

    suspend fun exprToString(expr: Long): String = withTimeoutAndExceptionHandling {
        worker.protocolModel.exprToString.startSuspending(worker.lifetime, expr)
    }

    suspend fun getReasonUnknown(solver: Int): String = withTimeoutAndExceptionHandling {
        worker.protocolModel.getReasonUnknown.startSuspending(worker.lifetime, solver)
    }

    suspend fun mkTrueExpr(): Long = withTimeoutAndExceptionHandling {
        worker.protocolModel.mkTrueExpr.startSuspending(worker.lifetime, Unit)
    }

    @Suppress("TooGenericExceptionCaught", "SwallowedException", "ThrowsCount")
    private suspend inline fun <T> withTimeoutAndExceptionHandling(crossinline body: suspend () -> T): T {
        try {
            return withTimeout(hardTimeout) {
                body()
            }
        } catch (ex: RdFault) {
            throw rdExceptionCause(ex) ?: ex
        } catch (ex: Exception) {
            worker.terminate()
            throw ex
        }
    }

    fun rdExceptionCause(ex: RdFault): Throwable? = when (ex.reasonTypeFqn) {
        NotImplementedError::class.simpleName ->
            NotImplementedError(ex.reasonMessage)
        KSolverUnsupportedFeatureException::class.simpleName ->
            KSolverUnsupportedFeatureException(ex.reasonMessage)
        SmtLibParseError::class.simpleName ->
            SmtLibParseError(ex)
        else -> null
    }
}
