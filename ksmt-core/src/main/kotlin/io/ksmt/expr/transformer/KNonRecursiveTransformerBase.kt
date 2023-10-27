package io.ksmt.expr.transformer

import io.ksmt.expr.KApp
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
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
    private var transformed = IdentityHashMap<KExpr<*>, KExpr<*>>()
    private val exprStack = ArrayList<KExpr<*>>()
    private var exprWasTransformed = false

    /**
     * Transform [expr] and all it sub-expressions non-recursively.
     * */
    override fun <T : KSort> apply(expr: KExpr<T>): KExpr<T> {
        val initialStackSize = exprStack.size
        try {
            exprStack.add(expr)
            while (exprStack.size > initialStackSize) {
                val e = exprStack.removeLast()

                val cachedExpr = transformedExpr(e)
                if (cachedExpr != null) {
                    continue
                }

                exprWasTransformed = true
                val transformedExpr = e.accept(this)

                if (exprWasTransformed) {
                    transformed[e] = transformedExpr
                }
            }
        } finally {
            // cleanup stack after exceptions
            while (exprStack.size > initialStackSize) {
                exprStack.removeLast()
            }
        }

        return transformedExpr(expr) ?: error("expr was not properly transformed: $expr")
    }

    /**
     * Reset transformer expression cache.
     * */
    fun resetCache() {
        check(exprStack.isEmpty()) { "Can not reset cache during expression transformation" }
        transformed = IdentityHashMap()
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
     * Disable [KTransformer] transform KArray2Lambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Lambda<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>>

    /**
     * Disable [KTransformer] transform KArray3Lambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Lambda<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>>

    /**
     * Disable [KTransformer] transform KArrayNLambda implementation since it is incorrect
     * for non-recursive usage.
     * */
    abstract override fun <R : KSort> transform(expr: KArrayNLambda<R>): KExpr<KArrayNSort<R>>

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
        var hasNonTransformedDependencies = false

        for (dependency in dependencies) {
            val transformedDependency = transformedExpr(dependency)

            if (transformedDependency != null) {
                transformedDependencies += transformedDependency
                continue
            }

            if (!hasNonTransformedDependencies) {
                hasNonTransformedDependencies = true
                retryExprTransformation(expr)
            }
            transformExprDependencyIfNeeded(dependency, transformedDependency)
        }

        if (hasNonTransformedDependencies) {
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

    /**
     * Specialized version of [transformExprAfterTransformed] for expression with five arguments.
     * */
    @Suppress("LongParameterList", "ComplexCondition")
    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort, A4 : KSort> transformExprAfterTransformed(
        expr: KExpr<T>,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        dependency2: KExpr<A2>,
        dependency3: KExpr<A3>,
        dependency4: KExpr<A4>,
        transformer: (KExpr<A0>, KExpr<A1>, KExpr<A2>, KExpr<A3>, KExpr<A4>) -> KExpr<T>
    ): KExpr<T> {
        val td0 = transformedExpr(dependency0)
        val td1 = transformedExpr(dependency1)
        val td2 = transformedExpr(dependency2)
        val td3 = transformedExpr(dependency3)
        val td4 = transformedExpr(dependency4)

        if (td0 == null || td1 == null || td2 == null || td3 == null || td4 == null) {
            retryExprTransformation(expr)

            transformExprDependencyIfNeeded(dependency0, td0)
            transformExprDependencyIfNeeded(dependency1, td1)
            transformExprDependencyIfNeeded(dependency2, td2)
            transformExprDependencyIfNeeded(dependency3, td3)
            transformExprDependencyIfNeeded(dependency4, td4)

            markExpressionAsNotTransformed()

            return expr
        }

        return transformer(td0, td1, td2, td3, td4)
    }
}
