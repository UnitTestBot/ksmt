package io.ksmt.solver.neurosmt.runtime

import ai.onnxruntime.OrtEnvironment
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import kotlin.math.exp

class NeuroSMTModelRunner(
    val ctx: KContext,
    ordinalsPath: String, embeddingPath: String, convPath: String, decoderPath: String
) {
    val env = OrtEnvironment.getEnvironment()

    val ordinalEncoder = OrdinalEncoder(ordinalsPath)
    val embeddingLayer = ONNXModel(env, embeddingPath)
    val convLayer = ONNXModel(env, convPath)

    val decoder = ONNXModel(env, decoderPath)

    fun run(expr: KExpr<*>): Float {
        val encoder = ExprEncoder(ctx, env, ordinalEncoder, embeddingLayer, convLayer)
        val exprFeatures = encoder.encodeExpr(expr)
        val result = decoder.forward(mapOf("expr_features" to exprFeatures))
        val logit = result.floatBuffer[0]

        return 1f / (1f + exp(-logit)) // sigmoid calculation
    }
}