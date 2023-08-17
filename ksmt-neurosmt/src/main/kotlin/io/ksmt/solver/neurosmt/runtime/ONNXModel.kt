package io.ksmt.solver.neurosmt.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.exp
import kotlin.streams.asSequence

class ONNXModel(env: OrtEnvironment, modelPath: String) : AutoCloseable {
    val session = env.createSession(modelPath)

    fun forward(input: Map<String, OnnxTensor>): OnnxTensor {
        val result = session.run(input)
        return result.get(0) as OnnxTensor
    }

    override fun close() {
        session.close()
    }
}

class NeuroSMTModelRunner(
    ctx: KContext,
    ordinalsPath: String, embeddingPath: String, convPath: String, decoderPath: String
) {
    val env = OrtEnvironment.getEnvironment()

    val ordinalEncoder = OrdinalEncoder(ordinalsPath)
    val embeddingLayer = ONNXModel(env, embeddingPath)
    val convLayer = ONNXModel(env, convPath)
    val encoder = ExprEncoder(ctx, env, ordinalEncoder, embeddingLayer, convLayer)

    val decoder = ONNXModel(env, decoderPath)

    fun run(expr: KExpr<*>): Float {
        val exprFeatures = encoder.encodeExpr(expr)
        val result = decoder.forward(mapOf("expr_features" to exprFeatures))
        val logit = result.floatBuffer[0]

        return 1f / (1f + exp(-logit))
    }
}

const val UNKNOWN_VALUE = 1999

class OrdinalEncoder(ordinalsPath: String, private val unknownValue: Int = UNKNOWN_VALUE) {
    private val lookup = HashMap<String, Int>()

    init {
        Files.lines(Path.of(ordinalsPath)).asSequence().forEachIndexed { index, s ->
            lookup[s] = index
        }
    }

    fun getOrdinal(s: String): Int {
        return lookup[s] ?: unknownValue
    }
}