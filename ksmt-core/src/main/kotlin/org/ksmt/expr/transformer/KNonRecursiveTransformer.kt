package org.ksmt.expr.transformer

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import java.util.IdentityHashMap

abstract class KNonRecursiveTransformer(override val ctx: KContext) : KTransformer {
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

    fun addExprDependency(dependency: KExpr<*>) {
        exprStack += dependency
    }

    /**
     *  Inform [KNonRecursiveTransformer] that transformation was not applied to expression
     * */
    fun markExpressionAsNotTransformed() {
        exprWasTransformed = false
    }

    override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> =
        transformExprAfterTransformedDefault(expr, expr.args) { transformedArgs ->
            expr.decl.apply(transformedArgs)
        }

    override fun <D : KSort, R : KSort> transform(
        expr: KArrayLambda<D, R>
    ): KExpr<KArraySort<D, R>> = transformExprAfterTransformedDefault(expr, expr.body) { body ->
        mkArrayLambda(expr.indexVarDecl, body)
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.body) { body ->
            mkExistentialQuantifier(body, expr.bounds)
        }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        transformExprAfterTransformedDefault(expr, expr.body) { body ->
            mkUniversalQuantifier(body, expr.bounds)
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

            if (transformedDependency0 == null) {
                addExprDependency(dependency0)
            }

            if (transformedDependency1 == null) {
                addExprDependency(dependency1)
            }

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

            if (transformedDependency0 == null) {
                addExprDependency(dependency0)
            }

            if (transformedDependency1 == null) {
                addExprDependency(dependency1)
            }

            if (transformedDependency2 == null) {
                addExprDependency(dependency2)
            }

            markExpressionAsNotTransformed()

            return expr
        }

        return transformer(transformedDependency0, transformedDependency1, transformedDependency2)
    }

    inline fun <T : KSort, A : KSort> transformExprAfterTransformedDefault(
        expr: KExpr<T>,
        dependencies: List<KExpr<A>>,
        transformer: KContext.(List<KExpr<A>>) -> KExpr<T>
    ): KExpr<T> = transformExprAfterTransformed(expr, dependencies) { transformedDependencies ->
        if (transformedDependencies == dependencies) return transformExpr(expr)

        val transformedExpr = ctx.transformer(transformedDependencies)

        return transformExpr(transformedExpr)
    }

    inline fun <T : KSort, A : KSort> transformExprAfterTransformedDefault(
        expr: KExpr<T>,
        dependency: KExpr<A>,
        transformer: KContext.(KExpr<A>) -> KExpr<T>
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency) { td ->
        if (td == dependency) return transformExpr(expr)

        val transformedExpr = ctx.transformer(td)

        return transformExpr(transformedExpr)
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort> transformExprAfterTransformedDefault(
        expr: KExpr<T>,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>) -> KExpr<T>
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency0, dependency1) { td0, td1 ->
        if (td0 == dependency0 && td1 == dependency1) return transformExpr(expr)

        val transformedExpr = ctx.transformer(td0, td1)

        return transformExpr(transformedExpr)
    }

    inline fun <T : KSort, A0 : KSort, A1 : KSort, A2 : KSort> transformExprAfterTransformedDefault(
        expr: KExpr<T>,
        dependency0: KExpr<A0>,
        dependency1: KExpr<A1>,
        dependency2: KExpr<A2>,
        transformer: KContext.(KExpr<A0>, KExpr<A1>, KExpr<A2>) -> KExpr<T>
    ): KExpr<T> = transformExprAfterTransformed(expr, dependency0, dependency1, dependency2) { td0, td1, td2 ->
        if (td0 == dependency0 && td1 == dependency1 && td2 == dependency2) return transformExpr(expr)

        val transformedExpr = ctx.transformer(td0, td1, td2)

        return transformExpr(transformedExpr)
    }
}
