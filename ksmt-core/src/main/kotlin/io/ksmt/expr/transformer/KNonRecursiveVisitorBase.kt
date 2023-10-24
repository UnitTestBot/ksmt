package io.ksmt.expr.transformer

import io.ksmt.expr.KApp
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.sort.KSort
import java.util.IdentityHashMap

/**
 * Non-recursive expression visitor.
 *
 * Standard use-case: perform bottom-up expression visit.
 * In this scenario, we need to visit expression arguments first,
 * and then perform visit of the expression using
 * the arguments visit result.
 * For this scenario, non-recursive visitor provides
 * a [visitExprAfterVisited] method.
 * */
abstract class KNonRecursiveVisitorBase<V : Any> : KVisitor<KExprVisitResult<V>> {
    private var visitResults = IdentityHashMap<KExpr<*>, V>()
    private val exprStack = ArrayList<KExpr<*>>()

    private var lastVisitedExpr: KExpr<*>? = null
    private var lastVisitResult: KExprVisitResult<V> = KExprVisitResult.EMPTY

    /**
     * Visit [expr] and all it sub-expressions non-recursively.
     * */
    fun <T : KSort> applyVisitor(expr: KExpr<T>): V {
        apply(expr)
        return result(expr)
    }

    override fun <E : KExpr<*>> exprVisitResult(expr: E, result: KExprVisitResult<V>) {
        lastVisitedExpr = expr
        lastVisitResult = result
    }

    fun resetLastVisitResult() {
        lastVisitedExpr = null
        lastVisitResult = KExprVisitResult.EMPTY
    }

    fun lastVisitResult(expr: KExpr<*>): KExprVisitResult<V> {
        check(lastVisitResult.isNotEmpty && expr === lastVisitedExpr) {
            "Expression $expr was not properly visited"
        }
        return lastVisitResult
    }

    fun <E : KExpr<*>> saveVisitResult(expr: E, result: V): KExprVisitResult<V> {
        val wrappedResult = KExprVisitResult<V>(result)
        exprVisitResult(expr, wrappedResult)
        return wrappedResult
    }

    /**
     *  Get [expr] visit result.
     *  Returns null if expression was not visited.
     * */
    fun <T : KSort> visitResult(expr: KExpr<T>): V? =
        visitResults[expr]

    fun <T : KSort> result(expr: KExpr<T>): V =
        visitResult(expr) ?: error("Expr $expr was not properly visited")

    override fun <T : KSort> apply(expr: KExpr<T>): KExpr<T> {
        val initialStackSize = exprStack.size
        try {
            exprStack.add(expr)
            while (exprStack.size > initialStackSize) {
                val e = exprStack.removeLast()

                val cached = visitResult(e)
                if (cached != null) {
                    continue
                }

                resetLastVisitResult()
                e.accept(this)
                val result = lastVisitResult(e)

                if (result.hasResult) {
                    visitResults[e] = result.result
                }
            }
        } finally {
            resetLastVisitResult()
            // cleanup stack after exceptions
            while (exprStack.size > initialStackSize) {
                exprStack.removeLast()
            }
        }

        return expr
    }

    /**
     * Reset visitor expression cache.
     * */
    fun resetCache() {
        check(exprStack.isEmpty()) { "Can not reset cache during expression visit" }
        visitResults = IdentityHashMap()
    }

    /**
     * Disable [KVisitor] visitApp implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <T : KSort, A : KSort> visitApp(expr: KApp<T, A>): KExprVisitResult<V>

    /**
     * Disable [KVisitor] visit KArrayLambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <D : KSort, R : KSort> visit(expr: KArrayLambda<D, R>): KExprVisitResult<V>

    /**
     * Disable [KVisitor] visit KArray2Lambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <D0 : KSort, D1 : KSort, R : KSort> visit(
        expr: KArray2Lambda<D0, D1, R>
    ): KExprVisitResult<V>

    /**
     * Disable [KVisitor] visit KArray3Lambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExprVisitResult<V>

    /**
     * Disable [KVisitor] visit KArrayNLambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <R : KSort> visit(expr: KArrayNLambda<R>): KExprVisitResult<V>

    /**
     * Disable [KVisitor] visit KExistentialQuantifier implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun visit(expr: KExistentialQuantifier): KExprVisitResult<V>

    /**
     * Disable [KVisitor] visit KUniversalQuantifier implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun visit(expr: KUniversalQuantifier): KExprVisitResult<V>

    /**
     * Visit [this] expression after [dependencies] expressions
     * */
    fun KExpr<*>.visitAfter(dependencies: List<KExpr<*>>) {
        exprStack += this
        exprStack += dependencies
    }

    fun KExpr<*>.visitAfter(dependency: KExpr<*>) {
        exprStack += this
        exprStack += dependency
    }

    fun retryExprVisit(expr: KExpr<*>) {
        exprStack += expr
    }

    fun visitExprDependencyIfNeeded(dependency: KExpr<*>, dependencyResult: V?) {
        if (dependencyResult == null) {
            exprStack += dependency
        }
    }

    /**
     * Visit [expr] only if all it sub-expressions
     * [dependencies] were already visited.
     * Otherwise, register [expr] for visit after [dependencies].
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisited(
        expr: E,
        dependencies: List<KExpr<*>>,
        visitor: (List<V>) -> V
    ): KExprVisitResult<V> {
        val dependenciesResults = ArrayList<V>(dependencies.size)
        var hasNonVisitedDependencies = false

        for (dependency in dependencies) {
            val dependencyResult = visitResult(dependency)

            if (dependencyResult != null) {
                dependenciesResults += dependencyResult
                continue
            }

            if (!hasNonVisitedDependencies) {
                hasNonVisitedDependencies = true
                retryExprVisit(expr)
            }
            visitExprDependencyIfNeeded(dependency, dependencyResult)
        }

        if (hasNonVisitedDependencies) {
            return KExprVisitResult.VISIT_DEPENDENCY
        }

        val result = visitor(dependenciesResults)
        return saveVisitResult(expr, result)
    }

    /**
     * Specialized version of [visitExprAfterVisited] for expression with single argument.
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisited(
        expr: E,
        dependency: KExpr<*>,
        visitor: (V) -> V
    ): KExprVisitResult<V> {
        val dependencyResult = visitResult(dependency)

        if (dependencyResult == null) {
            expr.visitAfter(dependency)

            return KExprVisitResult.VISIT_DEPENDENCY
        }

        val result = visitor(dependencyResult)
        return saveVisitResult(expr, result)
    }

    /**
     * Specialized version of [visitExprAfterVisited] for expression with two arguments.
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisited(
        expr: E,
        dependency0: KExpr<*>,
        dependency1: KExpr<*>,
        visitor: (V, V) -> V
    ): KExprVisitResult<V> {
        val dependency0Result = visitResult(dependency0)
        val dependency1Result = visitResult(dependency1)

        if (dependency0Result == null || dependency1Result == null) {
            retryExprVisit(expr)

            visitExprDependencyIfNeeded(dependency0, dependency0Result)
            visitExprDependencyIfNeeded(dependency1, dependency1Result)

            return KExprVisitResult.VISIT_DEPENDENCY
        }

        val result = visitor(dependency0Result, dependency1Result)
        return saveVisitResult(expr, result)
    }

    /**
     * Specialized version of [visitExprAfterVisited] for expression with three arguments.
     * */
    inline fun <E : KExpr<*>> visitExprAfterVisited(
        expr: E,
        dependency0: KExpr<*>,
        dependency1: KExpr<*>,
        dependency2: KExpr<*>,
        visitor: (V, V, V) -> V
    ): KExprVisitResult<V> {
        val dependency0Result = visitResult(dependency0)
        val dependency1Result = visitResult(dependency1)
        val dependency2Result = visitResult(dependency2)

        if (dependency0Result == null || dependency1Result == null || dependency2Result == null) {
            retryExprVisit(expr)

            visitExprDependencyIfNeeded(dependency0, dependency0Result)
            visitExprDependencyIfNeeded(dependency1, dependency1Result)
            visitExprDependencyIfNeeded(dependency2, dependency2Result)

            return KExprVisitResult.VISIT_DEPENDENCY
        }

        val result = visitor(dependency0Result, dependency1Result, dependency2Result)
        return saveVisitResult(expr, result)
    }

    /**
     * Specialized version of [visitExprAfterVisited] for expression with four arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
    inline fun <E : KExpr<*>> visitExprAfterVisited(
        expr: E,
        dependency0: KExpr<*>,
        dependency1: KExpr<*>,
        dependency2: KExpr<*>,
        dependency3: KExpr<*>,
        visitor: (V, V, V, V) -> V
    ): KExprVisitResult<V> {
        val d0r = visitResult(dependency0)
        val d1r = visitResult(dependency1)
        val d2r = visitResult(dependency2)
        val d3r = visitResult(dependency3)

        if (d0r == null || d1r == null || d2r == null || d3r == null) {
            retryExprVisit(expr)

            visitExprDependencyIfNeeded(dependency0, d0r)
            visitExprDependencyIfNeeded(dependency1, d1r)
            visitExprDependencyIfNeeded(dependency2, d2r)
            visitExprDependencyIfNeeded(dependency3, d3r)

            return KExprVisitResult.VISIT_DEPENDENCY
        }

        val result = visitor(d0r, d1r, d2r, d3r)
        return saveVisitResult(expr, result)
    }

    /**
     * Specialized version of [visitExprAfterVisited] for expression with five arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
    inline fun <E : KExpr<*>> visitExprAfterVisited(
        expr: E,
        dependency0: KExpr<*>,
        dependency1: KExpr<*>,
        dependency2: KExpr<*>,
        dependency3: KExpr<*>,
        dependency4: KExpr<*>,
        visitor: (V, V, V, V, V) -> V
    ): KExprVisitResult<V> {
        val d0r = visitResult(dependency0)
        val d1r = visitResult(dependency1)
        val d2r = visitResult(dependency2)
        val d3r = visitResult(dependency3)
        val d4r = visitResult(dependency4)

        if (d0r == null || d1r == null || d2r == null || d3r == null || d4r == null) {
            retryExprVisit(expr)

            visitExprDependencyIfNeeded(dependency0, d0r)
            visitExprDependencyIfNeeded(dependency1, d1r)
            visitExprDependencyIfNeeded(dependency2, d2r)
            visitExprDependencyIfNeeded(dependency3, d3r)
            visitExprDependencyIfNeeded(dependency4, d4r)

            return KExprVisitResult.VISIT_DEPENDENCY
        }

        val result = visitor(d0r, d1r, d2r, d3r, d4r)
        return saveVisitResult(expr, result)
    }
}

@JvmInline
value class KExprVisitResult<out V> internal constructor(private val value: Any?) {
    val isNotEmpty: Boolean
        get() = value !== empty

    val dependencyVisitRequired: Boolean
        get() = value === visitDependency

    val hasResult: Boolean
        get() = value !== empty && value !== visitDependency

    @Suppress("UNCHECKED_CAST")
    val result: V
        get() = value as? V ?: error("expr is not visited")

    companion object {
        private val empty = Any()
        private val visitDependency = Any()

        val EMPTY = KExprVisitResult<Nothing>(empty)
        val VISIT_DEPENDENCY = KExprVisitResult<Nothing>(visitDependency)
    }
}
