package io.ksmt.solver.neurosmt.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.*
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.*

class ExprEncoder(
    override val ctx: KContext,
    val env: OrtEnvironment,
    val ordinalEncoder: OrdinalEncoder,
    val embeddingLayer: ONNXModel,
    val convLayer: ONNXModel
) : KNonRecursiveTransformer(ctx) {

    private val exprToState = IdentityHashMap<KExpr<*>, OnnxTensor>()

    fun encodeExpr(expr: KExpr<*>): OnnxTensor {
        apply(expr)

        return exprToState[expr] ?: error("expression state wasn't calculated yet")
    }

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> {
        when (expr) {
            is KConst<*> -> calcSymbolicVariableState(expr)
            is KInterpretedValue<*> -> calcValueState(expr)
            else -> calcAppState(expr)
        }

        return expr
    }

    private fun getNodeEmbedding(key: String): OnnxTensor {
        val nodeLabel = ordinalEncoder.getOrdinal(key)
        val labelTensor = OnnxTensor.createTensor(
            env, IntBuffer.allocate(1).put(nodeLabel).rewind(), longArrayOf(1, 1)
        )

        return embeddingLayer.forward(mapOf("node_labels" to labelTensor))
    }

    private fun createEdgeTensor(childrenCnt: Int): OnnxTensor {
        val edges = listOf(
            List(childrenCnt) { it + 1L },
            List(childrenCnt) { 0L }
        )

        val buffer = LongBuffer.allocate(childrenCnt * 2)
        edges.forEach { row ->
            row.forEach { node ->
                buffer.put(node)
            }
        }
        buffer.rewind()

        return OnnxTensor.createTensor(env, buffer, longArrayOf(2, childrenCnt.toLong()))
    }

    private fun <T : KSort, A : KSort> calcAppState(expr: KApp<T, A>) {
        val childrenStates = expr.args.map { exprToState[it] ?: error("expression state wasn't calculated yet") }
        val childrenCnt = childrenStates.size

        val nodeEmbedding = getNodeEmbedding(expr.decl.name)
        val embeddingSize = nodeEmbedding.info.shape.reduce { acc, l -> acc * l }

        val buffer = FloatBuffer.allocate((1 + childrenCnt) * embeddingSize.toInt())
        buffer.put(nodeEmbedding.floatBuffer)
        childrenStates.forEach {
            buffer.put(it.floatBuffer)
        }
        buffer.rewind()
        val nodeFeatures = OnnxTensor.createTensor(env, buffer, longArrayOf(1L + childrenCnt, embeddingSize))

        val edges = createEdgeTensor(childrenStates.size)

        val result = convLayer.forward(mapOf("node_features" to nodeFeatures, "edges" to edges))
        val newNodeFeatures = OnnxTensor.createTensor(env, result.floatBuffer.slice(0, embeddingSize.toInt()), longArrayOf(1L, embeddingSize))
        exprToState[expr] = newNodeFeatures
    }

    private fun <T : KSort> calcSymbolicVariableState(symbol: KConst<T>) {
        val key = when (symbol.decl.sort) {
            is KBoolSort -> "SYMBOLIC;Bool"
            is KBvSort -> "SYMBOLIC;BitVec"
            is KFpSort -> "SYMBOLIC;FP"
            is KFpRoundingModeSort -> "SYMBOLIC;FP_RM"
            is KArraySortBase<*> -> "SYMBOLIC;Array"
            is KUninterpretedSort -> "SYMBOLIC;Unint"
            else -> error("unknown symbolic sort: ${symbol.decl.sort::class.simpleName}")
        }

        exprToState[symbol] = getNodeEmbedding(key)
    }

    private fun <T : KSort> calcValueState(value: KInterpretedValue<T>) {
        val key = when (value.decl.sort) {
            is KBoolSort -> "VALUE;Bool"
            is KBvSort -> "VALUE;BitVec"
            is KFpSort -> "VALUE;FP"
            is KFpRoundingModeSort -> "VALUE;FP_RM"
            is KArraySortBase<*> -> "VALUE;Array"
            is KUninterpretedSort -> "VALUE;Unint"
            else -> error("unknown value sort: ${value.decl.sort::class.simpleName}")
        }

        exprToState[value] = getNodeEmbedding(key)
    }
}