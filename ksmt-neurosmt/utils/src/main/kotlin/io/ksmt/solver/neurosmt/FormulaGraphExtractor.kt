package io.ksmt.solver.neurosmt

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.*
import java.io.OutputStream
import java.util.*

class FormulaGraphExtractor(
    override val ctx: KContext,
    val formula: KExpr<KBoolSort>,
    outputStream: OutputStream
) : KNonRecursiveTransformer(ctx) {

    private val exprToVertexID = IdentityHashMap<KExpr<*>, Long>()
    private var currentID = 0L

    private val writer = outputStream.bufferedWriter()

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> {
        exprToVertexID[expr] = currentID++

        when (expr) {
            is KInterpretedValue<*> -> writeValue(expr)
            is KConst<*> -> writeSymbolicVariable(expr)
            else -> writeApp(expr)
        }

        writer.newLine()

        return expr
    }

    private fun <T : KSort> writeSymbolicVariable(symbol: KConst<T>) {
        writer.write("SYMBOLIC;")

        val sort = symbol.decl.sort
        when (sort) {
            is KBoolSort -> writer.write("Bool")
            is KBvSort -> writer.write("BitVec")
            is KFpSort -> writer.write("FP")
            is KFpRoundingModeSort -> writer.write("FP_RM")
            is KArraySortBase<*> -> writer.write("Array")
            is KUninterpretedSort -> writer.write(sort.name)
            else -> error("unknown symbolic sort: ${sort::class.simpleName}")
        }
    }

    private fun <T : KSort> writeValue(value: KInterpretedValue<T>) {
        writer.write("VALUE;")

        val sort = value.decl.sort
        when (sort) {
            is KBoolSort -> writer.write("Bool")
            is KBvSort -> writer.write("BitVec")
            is KFpSort -> writer.write("FP")
            is KFpRoundingModeSort -> writer.write("FP_RM")
            is KArraySortBase<*> -> writer.write("Array")
            is KUninterpretedSort -> writer.write(sort.name)
            else -> error("unknown value sort: ${sort::class.simpleName}")
        }
    }

    private fun <T : KSort, A : KSort> writeApp(expr: KApp<T, A>) {
        writer.write("${expr.decl.name};")

        for (child in expr.args) {
            writer.write(" ${exprToVertexID[child]}")
        }
    }

    fun extractGraph() {
        apply(formula)
        writer.close()
    }
}