package io.ksmt.expr.printer

import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr

/**
 * Print expressions non-recursively.
 * Use let-bindings for common subexpressions to minimize result string.
 * */
class ExpressionPrinterWithLetBindings {
    fun print(expr: KExpr<*>, out: StringBuilder) {
        val resolvedExpressionPrinters = hashMapOf<KExpr<*>, SingleExpressionPrinter>()
        val printQueue = arrayListOf<SingleExpressionPrinter>()
        val enqueuedToPrint = hashSetOf<KExpr<*>>()
        val hasMultipleOccurrences = hashSetOf<KExpr<*>>()

        val exprStack = arrayListOf(expr)
        while (exprStack.isNotEmpty()) {
            val e = exprStack.removeLast()

            if (e in resolvedExpressionPrinters || e in enqueuedToPrint) {
                hasMultipleOccurrences.add(e)
                continue
            }

            val printer = SingleExpressionPrinter(e).also { e.print(it) }
            printQueue.add(printer)
            enqueuedToPrint.add(e)

            if (!printer.hasDependency) {
                resolvedExpressionPrinters[e] = printer
            } else {
                exprStack.addAll(printer.dependency)
            }
        }

        val letBindings = generateLetBindings(
            resolvedExpressionPrinters = resolvedExpressionPrinters,
            // reverse printer queue to generate lower binding indices for simpler expressions
            unresolvedPrinters = printQueue.asReversed(),
            hasMultipleOccurrences = hasMultipleOccurrences
        )

        while (printQueue.isNotEmpty()) {
            val printer = printQueue.removeLast()
            if (printer.expr in resolvedExpressionPrinters) continue
            printer.resolveDependencies(resolvedExpressionPrinters)
            resolvedExpressionPrinters[printer.expr] = printer
        }

        for ((_, printer) in letBindings) {
            printer.resolveDependencies(resolvedExpressionPrinters)
        }

        val exprPrinter = resolvedExpressionPrinters.getValue(expr)
        printToBuffer(out, letBindings, exprPrinter)
    }

    @Suppress("LoopWithTooManyJumpStatements")
    private fun generateLetBindings(
        resolvedExpressionPrinters: MutableMap<KExpr<*>, SingleExpressionPrinter>,
        unresolvedPrinters: List<SingleExpressionPrinter>,
        hasMultipleOccurrences: Set<KExpr<*>>
    ): List<Pair<String, SingleExpressionPrinter>> {
        val letBindings = arrayListOf<Pair<String, SingleExpressionPrinter>>()
        val immutableResolvedPrinters = resolvedExpressionPrinters.values.toList()
        val letBindingCandidates = immutableResolvedPrinters.asSequence() + unresolvedPrinters.asSequence()
        for (printer in letBindingCandidates) {
            val expr = printer.expr
            when {
                expr !in hasMultipleOccurrences -> continue
                // Don't produce bindings for constants
                expr is KApp<*, *> && expr.args.isEmpty() -> continue
                else -> {
                    val bindingIdx = letBindings.size + 1
                    val bindingName = "e!$bindingIdx"

                    letBindings += bindingName to printer
                    resolvedExpressionPrinters[expr] = SingleExpressionPrinter(expr).apply {
                        append(bindingName)
                    }
                }
            }
        }
        return letBindings
    }

    private fun printToBuffer(
        out: StringBuilder,
        letBindings: List<Pair<String, SingleExpressionPrinter>>,
        exprPrinter: SingleExpressionPrinter
    ) {
        if (letBindings.isNotEmpty()) {
            out.appendLine("(let (")
            for ((name, bindingPrinter) in letBindings) {
                out.append('(')
                out.append(name)
                out.append(' ')
                bindingPrinter.printToBuffer(out)
                out.appendLine(')')
            }
            out.appendLine(')')
        }

        exprPrinter.printToBuffer(out)

        if (letBindings.isNotEmpty()) {
            out.append(')')
        }
    }

    private class SingleExpressionPrinter(val expr: KExpr<*>) : ExpressionPrinter {
        /**
         * Actual type: String | KExpr<*>
         * Type after dependency resolution: String | SingleExpressionPrinter
         * */
        private val parts = arrayListOf<Any>()
        var hasDependency = false
            private set

        val dependency: List<KExpr<*>>
            get() = if (hasDependency) parts.filterIsInstance<KExpr<*>>() else emptyList()

        override fun append(str: String) {
            parts.add(str)
        }

        override fun append(expr: KExpr<*>) {
            parts.add(expr)
            hasDependency = true
        }

        fun resolveDependencies(printedExpressions: Map<KExpr<*>, SingleExpressionPrinter>) {
            if (!hasDependency) return

            val currentParts = parts.toList()
            parts.clear()

            for (part in currentParts) {
                if (part !is KExpr<*>) {
                    parts.add(part)
                    continue
                }

                val printer = printedExpressions[part] ?: error("Printer failed")
                check(!printer.hasDependency) { "Printer has unresolved parts" }

                parts.add(printer)
            }

            hasDependency = false
        }

        fun printToBuffer(buffer: StringBuilder) {
            val printerStack = arrayListOf<Any>(this)
            while (printerStack.isNotEmpty()) {
                when (val element = printerStack.removeLast()) {
                    is String -> buffer.append(element)
                    is SingleExpressionPrinter -> {
                        check(!element.hasDependency) { "Printer has unresolved parts" }
                        printerStack.addAll(element.parts.asReversed())
                    }
                    else -> error("Unexpected printer part")
                }
            }
        }
    }

}
