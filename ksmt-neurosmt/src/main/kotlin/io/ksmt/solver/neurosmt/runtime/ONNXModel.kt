package io.ksmt.solver.neurosmt.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment

// wrapper for any exported ONNX model
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