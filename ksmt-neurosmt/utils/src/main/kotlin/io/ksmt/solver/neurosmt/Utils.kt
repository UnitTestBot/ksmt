package io.ksmt.solver.neurosmt

import com.jetbrains.rd.framework.SerializationCtx
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.UnsafeBuffer
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.runner.serializer.AstSerializationCtx
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.uncheckedCast
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Path
import kotlin.time.Duration

fun getAnswerForTest(path: Path): KSolverStatus {
    File(path.toUri()).useLines { lines ->
        for (line in lines) {
            when (line) {
                "(set-info :status sat)" -> return KSolverStatus.SAT
                "(set-info :status unsat)" -> return KSolverStatus.UNSAT
                "(set-info :status unknown)" -> return KSolverStatus.UNKNOWN
            }
        }
    }

    return KSolverStatus.UNKNOWN
}

fun getAnswerForTest(ctx: KContext, formula: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
    return KZ3Solver(ctx).use { solver ->
        for (clause in formula) {
            solver.assert(clause)
        }

        solver.check(timeout = timeout)
    }
}

fun serialize(ctx: KContext, expressions: List<KExpr<KBoolSort>>, outputStream: OutputStream) {
    val serializationCtx = AstSerializationCtx().apply { initCtx(ctx) }
    val marshaller = AstSerializationCtx.marshaller(serializationCtx)
    val emptyRdSerializationCtx = SerializationCtx(Serializers())

    val buffer = UnsafeBuffer(ByteArray(100_000))

    expressions.forEach { expr ->
        marshaller.write(emptyRdSerializationCtx, buffer, expr)
    }

    outputStream.write(buffer.getArray())
    outputStream.flush()
}

fun deserialize(ctx: KContext, inputStream: InputStream): List<KExpr<KBoolSort>> {
    val srcSerializationCtx = AstSerializationCtx().apply { initCtx(ctx) }
    val srcMarshaller = AstSerializationCtx.marshaller(srcSerializationCtx)
    val emptyRdSerializationCtx = SerializationCtx(Serializers())

    val buffer = UnsafeBuffer(inputStream.readBytes())
    val expressions: MutableList<KExpr<KBoolSort>> = mutableListOf()

    while (true) {
        try {
            expressions.add(srcMarshaller.read(emptyRdSerializationCtx, buffer).uncheckedCast())
        } catch (e : IllegalStateException) {
            break
        }
    }

    return expressions
}