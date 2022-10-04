package org.ksmt.runner.serializer

import com.jetbrains.rd.framework.FrameworkMarshallers
import com.jetbrains.rd.framework.RdId
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.getPlatformIndependentHash
import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

class AstSerializationCtx {
    private var context: KContext? = null

    private val serializedAst = hashMapOf<KAst, Int>()
    private val deserializedAst = hashMapOf<Int, KAst>()

    private var nextGeneratedIndex = 1

    val ctx: KContext
        get() = context ?: error("Serialization context is not initialized")

    fun initCtx(ctx: KContext) {
        check(context == null) { "Serialization context is initialized " }
        context = ctx
    }

    fun resetCtx() {
        context = null
    }

    fun mkAstIdx(ast: KAst): Int {
        check(!serializedAst.containsKey(ast)) { "duplicate ast" }
        val idx = nextGeneratedIndex++
        saveAst(idx, ast)
        return idx
    }

    private fun saveAst(idx: Int, ast: KAst) {
        deserializedAst[idx] = ast
        serializedAst[ast] = idx
    }

    fun writeAst(idx: Int, ast: KAst) {
        val current = deserializedAst[idx]
        if (current != null) {
            check(current == ast) { "different ast with same idx" }
            return
        }
        val reversed = serializedAst[ast]
        if (reversed != null) {
            check(deserializedAst[reversed] == ast) { "cache mismatch" }
        }
        saveAst(idx, ast)
        nextGeneratedIndex = maxOf(nextGeneratedIndex, idx + 1)
    }

    fun getAstIndex(ast: KAst): Int? = serializedAst[ast]
    fun getAstByIndexOrError(idx: Int): KAst = deserializedAst[idx] ?: error("not properly deserialized")

    companion object{
        const val SERIALIZED_DATA_END_IDX = -1

        val marshallerId: RdId by lazy {
            RdId(KAst::class.simpleName.getPlatformIndependentHash().toInt().toLong())
        }

        fun register(serializers: Serializers): AstSerializationCtx {
            val ctx = AstSerializationCtx()
            val marshaller = FrameworkMarshallers.create<KAst>(
                writer = { buffer, ast ->
                    val serializer = AstSerializer(ctx, buffer)
                    val serializedAst = with(serializer) {
                        when (ast) {
                            is KDecl<*> -> ast.serializeDecl()
                            is KSort -> ast.serializeSort()
                            is KExpr<*> -> ast.serializeExpr()
                            else -> error("Unexpected ast: ${ast::class}")
                        }
                    }
                    buffer.writeInt(SERIALIZED_DATA_END_IDX)
                    buffer.writeInt(serializedAst)
                },
                reader = { buffer ->
                    val deserializer = AstDeserializer(ctx, buffer)
                    deserializer.deserialize()
                    val serializedAstIdx = buffer.readInt()
                    ctx.getAstByIndexOrError(serializedAstIdx)
                },
                predefinedId = marshallerId.hash.toInt()
            )
            serializers.register(marshaller)
            return ctx
        }
    }

}
