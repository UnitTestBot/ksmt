package org.ksmt.test

import com.jetbrains.rd.framework.IMarshaller
import com.jetbrains.rd.framework.SerializationCtx
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.createAbstractBuffer
import com.jetbrains.rd.framework.readList
import com.jetbrains.rd.framework.writeList
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.runner.serializer.AstSerializationCtx
import java.nio.file.Path
import kotlin.test.assertEquals

class SerializerBenchmarksBasedTest : BenchmarksBasedTest() {

    @Execution(ExecutionMode.CONCURRENT)
    @ParameterizedTest(name = "{0}")
    @MethodSource("serializerTestData")
    fun testSerializer(
        name: String,
        samplePath: Path
    ) {
        val ctx1 = KContext()
        val ctx2 = KContext()
        testWorkers.withWorker(ctx1) { worker ->
            val assertions = worker.parseFile(samplePath)
            val convertedAssertions = worker.convertAssertions(assertions)

            convertedAssertions.forEachIndexed { idx, it ->
                SortChecker(ctx1).apply(it)
            }

            val (serialized, deserialized) = serializeAndDeserialize(ctx1, ctx2, convertedAssertions)

            assertEquals(convertedAssertions, deserialized)

            val (restored, _) = serializeAndDeserialize(ctx2, ctx1, serialized)

            assertEquals(convertedAssertions, restored)
        }
    }

    private fun serializeAndDeserialize(
        sourceCtx: KContext,
        targetCtx: KContext,
        expressions: List<KExpr<*>>,
    ): Pair<List<KExpr<*>>, List<KExpr<*>>> {
        val srcSerializationCtx = AstSerializationCtx().apply { initCtx(sourceCtx) }
        val srcMarshaller = AstSerializationCtx.marshaller(srcSerializationCtx)

        val targetSerializationCtx = AstSerializationCtx().apply { initCtx(targetCtx) }
        val targetMarshaller = AstSerializationCtx.marshaller(targetSerializationCtx)

        val serialized = serializeAndDeserialize(expressions, srcMarshaller, targetMarshaller)
        val deserialized = serializeAndDeserialize(serialized, targetMarshaller, srcMarshaller)
        return serialized to deserialized
    }


    private fun serializeAndDeserialize(
        expressions: List<KExpr<*>>,
        ctx1Marshaller: IMarshaller<KAst>,
        ctx2Marshaller: IMarshaller<KAst>
    ): List<KExpr<*>> {

        val emptyRdSerializationCtx = SerializationCtx(Serializers())
        val buffer = createAbstractBuffer()

        buffer.writeList(expressions) { expr ->
            ctx1Marshaller.write(emptyRdSerializationCtx, buffer, expr)
        }

        buffer.rewind()

        val deserializedExpressions = buffer.readList {
            ctx2Marshaller.read(emptyRdSerializationCtx, buffer) as KExpr<*>
        }

        return deserializedExpressions

    }

    companion object {

        @JvmStatic
        fun serializerTestData() = testData
    }
}
