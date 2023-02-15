package org.ksmt.expr.transformer

import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort
import java.util.IdentityHashMap

abstract class KNonRecursiveTransformerUtil: KTransformerBase {
    private val transformed = IdentityHashMap<KExpr<*>, KExpr<*>>()
    private val exprStack = ArrayList<KExpr<*>>()
    private var exprWasTransformed = false

    /**
     * Transform [rootExpr] and all it sub-expressions non-recursively.
     * */
    fun <T : KSort> apply(rootExpr: KExpr<T>): KExpr<T> {
        transformedExpr(rootExpr) ?: exprStack.add(rootExpr)

        while (exprStack.isNotEmpty()) {
            val expr = exprStack.removeLast()
            exprWasTransformed = true
            val transformedExpr = expr.accept(this)

            if (exprWasTransformed) {
                transformed[expr] = transformedExpr
            }
        }

        return transformedExpr(rootExpr) ?: error("expr was not properly transformed: $rootExpr")
    }

    /**
     *  Get [expr] transformation result or
     *  null if expression was not transformed yet
     * */
    fun <T : KSort> transformedExpr(expr: KExpr<T>): KExpr<T>? {
        @Suppress("UNCHECKED_CAST")
        return transformed[expr] as? KExpr<T>
    }

    /**
     * Transform [this] expression after [dependencies] expressions
     * */
    fun KExpr<*>.transformAfter(dependencies: List<KExpr<*>>) {
        exprStack += this
        exprStack += dependencies
    }

    fun KExpr<*>.transformAfter(dependency: KExpr<*>) {
        exprStack += this
        exprStack += dependency
    }

    fun retryExprTransformation(expr: KExpr<*>) {
        exprStack += expr
    }

    fun transformExprDependencyIfNeeded(dependency: KExpr<*>, transformedDependency: KExpr<*>?) {
        if (transformedDependency == null) {
            exprStack += dependency
        }
    }

    /**
     *  Inform [KNonRecursiveTransformer] that transformation was not applied to expression
     * */
    fun markExpressionAsNotTransformed() {
        exprWasTransformed = false
    }

    /**
     * [KApp] non-recursive transformation helper.
     * @see [transformExprAfterTransformed]
     * */
    inline fun <T : KSort, A : KSort> transformAppAfterArgsTransformed(
        expr: KApp<T, A>,
        transformer: (List<KExpr<A>>) -> KExpr<T>
    ): KExpr<T> = transformExprAfterTransformed(expr, expr.args, transformer)


    /**
     * Transform [expr] only if all it sub-expressions
     * [dependencies] were already transformed.
     * Otherwise, register [expr] for transformation after [dependencies]
     * and keep [expr] unchanged.
     * */
    inline fun <T : KSort, A : KSort> transformExprAfterTransformed(
        expr: KExpr<T>,
        dependencies: List<KExpr<A>>,
        transformer: (List<KExpr<A>>) -> KExpr<T>
    ): KExpr<T> {
        val transformedDependencies = mutableListOf<KExpr<A>>()
        val notTransformedDependencies = mutableListOf<KExpr<A>>()

        for (dependency in dependencies) {
            val transformedDependency = transformedExpr(dependency)

            if (transformedDependency != null) {
                transformedDependencies += transformedDependency
                continue
            } else {
                notTransformedDependencies += dependency
            }
        }

        if (notTransformedDependencies.isNotEmpty()) {
            expr.transformAfter(notTransformedDependencies)
            markExpressionAsNotTransformed()

            return expr
        }

        return transformer(transformedDependencies)
    }

    inline fun <T : KSort, A : KSort> transformExprAfterTransformed(
        expr: KExpr<T>,
        dependency: KExpr<A>,
        transformer: (KExpr<A>) -> KExpr<T>
    ): KExpr<T> {
        val transformedDependency = transformedExpr(dependency)

        if (transformedDependency == null) {
            expr.transformAfter(dependency)
            markExpressionAsNotTransformed()

            return expr
        }

        return transformer(transformedDependency)
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort> transformExprAfterTransformed(
        expr: KExpr<T>,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        transformer: (KExpr<A0>, KExpr<A1>) -> KExpr<T>
    ): KExpr<T> {
        val transformedDependency0 = transformedExpr(dependency0)
        val transformedDependency1 = transformedExpr(dependency1)

        if (transformedDependency0 == null || transformedDependency1 == null) {
            retryExprTransformation(expr)

            transformExprDependencyIfNeeded(dependency0, transformedDependency0)
            transformExprDependencyIfNeeded(dependency1, transformedDependency1)

            markExpressionAsNotTransformed()

            return expr
        }

        return transformer(transformedDependency0, transformedDependency1)
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> transformExprAfterTransformed(
        expr: KExpr<T>,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        dependency2: KExpr<A2>,
        transformer: (KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ): KExpr<T> {
        val transformedDependency0 = transformedExpr(dependency0)
        val transformedDependency1 = transformedExpr(dependency1)
        val transformedDependency2 = transformedExpr(dependency2)

        if (transformedDependency0 == null || transformedDependency1 == null || transformedDependency2 == null) {
            retryExprTransformation(expr)

            transformExprDependencyIfNeeded(dependency0, transformedDependency0)
            transformExprDependencyIfNeeded(dependency1, transformedDependency1)
            transformExprDependencyIfNeeded(dependency2, transformedDependency2)

            markExpressionAsNotTransformed()

            return expr
        }

        return transformer(transformedDependency0, transformedDependency1, transformedDependency2)
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3: KSort> transformExprAfterTransformed(
        expr: KExpr<T>,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        dependency2: KExpr<A2>,
        dependency3: KExpr<A3>,
        transformer: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>) -> KExpr<T>
    ): KExpr<T> {
        val td0 = transformedExpr(dependency0)
        val td1 = transformedExpr(dependency1)
        val td2 = transformedExpr(dependency2)
        val td3 = transformedExpr(dependency3)

        if (td0 == null || td1 == null || td2 == null || td3 == null) {
            retryExprTransformation(expr)

            transformExprDependencyIfNeeded(dependency0, td0)
            transformExprDependencyIfNeeded(dependency1, td1)
            transformExprDependencyIfNeeded(dependency2, td2)
            transformExprDependencyIfNeeded(dependency3, td3)

            markExpressionAsNotTransformed()

            return expr
        }

        return transformer(td0, td1, td2, td3)
    }
}
