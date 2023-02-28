package org.ksmt.expr.transformer

import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import java.util.IdentityHashMap

/**
 * Non-recursive expression transformer.
 *
 * Standard use-case: perform bottom-up expression transformation.
 * In this scenario, we need to transform expression arguments first,
 * and then perform transformation of the expression using
 * the transformed arguments.
 * For this scenario, non-recursive transformer provides
 * a [transformExprAfterTransformed] method.
 * */
abstract class KNonRecursiveTransformerBase: KTransformer {
    private val transformed = IdentityHashMap<KExpr<*>, KExpr<*>>()
    private val exprStack = ArrayList<KExpr<*>>()
    private var exprWasTransformed = false

    /**
     * Transform [expr] and all it sub-expressions non-recursively.
     * */
    override fun <T : KSort> apply(expr: KExpr<T>): KExpr<T> {
        val cachedExpr = transformedExpr(expr)
        if (cachedExpr != null) {
            return cachedExpr
        }

        exprStack.add(expr)
        while (exprStack.isNotEmpty()) {
            val e = exprStack.removeLast()
            exprWasTransformed = true
            val transformedExpr = e.accept(this)

            if (exprWasTransformed) {
                transformed[e] = transformedExpr
            }
        }

        return transformedExpr(expr) ?: error("expr was not properly transformed: $expr")
    }

    /**
     * Disable [KTransformer] transformApp implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T>

    /**
     * Disable [KTransformer] transform KArrayLambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <D : KSort, R : KSort> transform(expr: KArrayLambda<D, R>): KExpr<KArraySort<D, R>>

    /**
     * Disable [KTransformer] transform KExistentialQuantifier implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort>

    /**
     * Disable [KTransformer] transform KUniversalQuantifier implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort>

    /**
     *  Get [expr] transformation result or
     *  null if expression was not transformed yet
     * */
    fun <T : KSort> transformedExpr(expr: KExpr<T>): KExpr<T>? {
        if (!exprTransformationRequired(expr)) return expr

        @Suppress("UNCHECKED_CAST")
        return transformed[expr] as? KExpr<T>
    }

    /**
     * Allows to skip expression transformation and stop deepening.
     * */
    open fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean = true

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
        val transformedDependencies = ArrayList<KExpr<A>>(dependencies.size)
        val notTransformedDependencies = ArrayList<KExpr<A>>(dependencies.size)

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

    /**
     * Specialized version of [transformExprAfterTransformed] for expression with single argument.
     * */
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

    /**
     * Specialized version of [transformExprAfterTransformed] for expression with two arguments.
     * */
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

    /**
     * Specialized version of [transformExprAfterTransformed] for expression with three arguments.
     * */
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

    /**
     * Specialized version of [transformExprAfterTransformed] for expression with four arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
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
