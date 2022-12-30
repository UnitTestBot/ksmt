package org.ksmt.test

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast
import java.lang.reflect.InvocationTargetException
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.random.nextUInt
import kotlin.reflect.KClass
import kotlin.reflect.KFunction
import kotlin.reflect.KParameter
import kotlin.reflect.KType
import kotlin.reflect.KTypeParameter
import kotlin.reflect.KTypeProjection
import kotlin.reflect.KVisibility
import kotlin.reflect.full.allSupertypes
import kotlin.reflect.full.createType
import kotlin.reflect.full.isSubclassOf

class RandomExpressionGenerator(private val ctx: KContext, private val random: Random = Random(42)) {
    private val trace = arrayListOf<Any>()
    private val expressions = hashMapOf<KSort, MutableList<KExpr<*>>>()
    private val generators = arrayListOf<ExpressionGenerator>()
    private val sortGenerators = arrayListOf<SortGenerator>()
    private lateinit var mkConstFunction: KFunction<*>
    private lateinit var mkArraySortFunction: KFunction<*>

    fun generate(limit: Int): KExpr<*> {
        trace.clear()
        expressions.clear()
        mkGenerators()
        generateInitialSeed(sortVariants = 10)

        var expr: KExpr<*>? = null
        var i = 0
        do {
            val generator = generators.random()
            expr = generator.generate() ?: continue
            i++
        } while (i <= limit)

        return expr!!
    }

    private fun registerExpr(expr: KExpr<*>) {
        expressions.getOrPut(expr.sort) { arrayListOf() }.add(expr)
    }

    private fun mkGenerators() {
        if (generators.isNotEmpty()) return
        val ctxFunctions = ctx::class.members
            .filter { it.visibility == KVisibility.PUBLIC }
            .filterIsInstance<KFunction<*>>()

        val expressionGenerators = ctxFunctions
            .filter { it.returnType.isKExpr() }
            .mapNotNull { it.mkGenerator { f, refSorts, args -> ExpressionGenerator(f, refSorts, args) } }

        val sortGenerators = ctxFunctions
            .filter { it.returnType.isKSort() }
            .mapNotNull { it.mkGenerator { f, refSorts, args -> SortGenerator(f, refSorts, args) } }

        this.generators.addAll(expressionGenerators)
        this.sortGenerators.addAll(sortGenerators)
        mkConstFunction = ctxFunctions.single { it.name == "mkConst" }
        mkArraySortFunction = ctxFunctions.single { it.name == "mkArraySort" }
    }

    private fun generateInitialSeed(sortVariants: Int) {
        for (sortGenerator in sortGenerators.filter { it.refSortProviders.isEmpty() }) {
            repeat(sortVariants) {
                sortGenerator.mkSeed()
            }
        }
        for (sortGenerator in sortGenerators.filter { it.refSortProviders.isNotEmpty() }) {
            repeat(sortVariants) {
                sortGenerator.mkSeed()
            }
        }
    }

    private fun SortGenerator.mkSeed(context: Map<String, KSort> = emptyMap()): KSort {
        val exprGenerator = ExpressionGenerator(
            mkConstFunction,
            emptyMap(),
            listOf(SimpleProvider(String::class), this)
        )
        val expr = exprGenerator.generate(context) ?: error("Error in constant generation")
        val sortExprs = expressions[expr.sort] ?: mutableListOf()
        // Don't store non unique seeds
        if (sortExprs.size > 1) {
            sortExprs.removeLast()
        }
        return expr.sort
    }

    private inline fun <reified G> KFunction<*>.mkGenerator(
        generator: (KFunction<*>, Map<String, SortProvider>, List<ArgumentProvider>) -> G
    ): G? {
        if (parameters.isEmpty()) return null
        if (parameters.any { it.kind == KParameter.Kind.EXTENSION_RECEIVER }) return null
        if (parameters[0].kind != KParameter.Kind.INSTANCE || !parameters[0].type.isKContext()) return null
        val valueParams = parameters.drop(1)
        if (valueParams.any { it.kind != KParameter.Kind.VALUE }) return null

        val typeParametersProviders = hashMapOf<String, SortProvider>()
        typeParameters.forEach {
            it.mkReferenceSortProvider(typeParametersProviders)
        }

        val argumentProviders = valueParams.map { it.mkArgProvider(typeParametersProviders) ?: return null }
        return generator(this, typeParametersProviders, argumentProviders)
    }

    private fun KParameter.mkArgProvider(typeParametersProviders: MutableMap<String, SortProvider>): ArgumentProvider? {
        if (type.isKExpr()) {
            val sortProvider = type.mkKExprSortProvider(typeParametersProviders)
            return ExprProvider(sortProvider)
        }
        if (type.isKSort()) {
            return type.mkSortProvider(typeParametersProviders)
        }
        if (type.isSubclassOf(List::class)) {
            val elementType = type.arguments.single().type ?: return null
            if (!elementType.isKExpr()) return null
            val sortProvider = elementType.mkKExprSortProvider(typeParametersProviders)
            return ListProvider(sortProvider)
        }
        if (type.isSimple()) {
            val cls = type.classifier as? KClass<*> ?: return null
            return SimpleProvider(cls)
        }
        val sortClass = type.classifier
        if (sortClass is KTypeParameter && sortClass.name in typeParametersProviders) {
            return type.mkSortProvider(typeParametersProviders)
        }
        return null
    }

    private fun KTypeParameter.mkReferenceSortProvider(references: MutableMap<String, SortProvider>) {
        val sortProvider = if (upperBounds.size == 1 && upperBounds.single().isKSort()) {
            upperBounds.single().mkSortProvider(references)
        } else {
            TODO()
        }
        references[name] = sortProvider
    }

    private fun KType.mkKExprSortProvider(references: MutableMap<String, SortProvider>): SortProvider {
        val expr = if (classifier == KExpr::class) {
            this
        } else {
            (classifier as? KClass<*>)?.allSupertypes?.find { it.classifier == KExpr::class }
                ?: TODO()
        }
        val sort = expr.arguments.single()
        if (sort == KTypeProjection.STAR) return SingleSortProvider(KSort::class)
        val sortType = sort.type ?: TODO()
        return sortType.mkSortProvider(references)
    }

    private fun KType.mkSortProvider(references: MutableMap<String, SortProvider>): SortProvider {
        val sortClass = classifier
        if (sortClass is KTypeParameter) {
            if (sortClass.name !in references) {
                sortClass.mkReferenceSortProvider(references)
            }
            return ReferenceSortProvider(sortClass.name)
        }
        if (this.isSubclassOf(KArraySort::class)) {
            val (domain, range) = arguments.map { it.type?.mkSortProvider(references) ?: TODO() }
            return ArraySortProvider(domain, range)
        }
        if (this.isKSort()) {
            return SingleSortProvider(sortClass.uncheckedCast())
        }
        TODO()
    }


    private fun KType.isSubclassOf(other: KClass<*>): Boolean =
        (classifier as? KClass<*>)?.isSubclassOf(other) ?: false

    private fun KType.isSimple(): Boolean = listOf(
        Boolean::class,
        Byte::class,
        Short::class,
        Int::class,
        UInt::class,
        Long::class,
        Float::class,
        Double::class,
        String::class,
        Enum::class
    ).any { isSubclassOf(it) }

    private fun KType.isKExpr(): Boolean = isSubclassOf(KExpr::class)

    private fun KType.isKSort(): Boolean = isSubclassOf(KSort::class)

    private fun KType.isKContext(): Boolean = this == KContext::class.createType()

    sealed interface ArgumentProvider {
        fun provide(references: Map<String, KSort>): Any
    }

    sealed interface SortProvider : ArgumentProvider {
        override fun provide(references: Map<String, KSort>): Any = resolve(references)
        fun resolve(references: Map<String, KSort>): KSort
    }

    inner class SingleSortProvider(val sort: KClass<KSort>) : SortProvider {
        override fun resolve(references: Map<String, KSort>): KSort {
            val candidates = expressions.keys.filter { it::class.isSubclassOf(sort) }
            return candidates.random(random)
        }
    }

    class ReferenceSortProvider(val reference: String) : SortProvider {
        override fun resolve(references: Map<String, KSort>): KSort {
            return references[reference]
                ?: TODO()
        }
    }

    inner class ArraySortProvider(val domain: SortProvider, val range: SortProvider) : SortProvider {
        override fun resolve(references: Map<String, KSort>): KSort {
            val generationParams = mapOf(
                "D" to domain,
                "R" to range
            )
            val generator = SortGenerator(
                mkArraySortFunction,
                generationParams,
                generationParams.keys.map { ReferenceSortProvider(it) }
            )
            return generator.mkSeed(references)
        }
    }

    inner class SimpleProvider(val type: KClass<*>) : ArgumentProvider {
        override fun provide(references: Map<String, KSort>): Any {
            return type.generateSimpleValue()
        }
    }

    inner class ListProvider(val element: SortProvider) : ArgumentProvider {
        override fun provide(references: Map<String, KSort>): Any {
            val concreteSort = element.resolve(references)
            val size = random.nextInt(2..10)
            val candidateExpressions = expressions[concreteSort] ?: TODO()
            return List(size) { candidateExpressions.random(random) }
        }
    }

    inner class ExprProvider(val sort: SortProvider) : ArgumentProvider {
        override fun provide(references: Map<String, KSort>): Any {
            val concreteSort = sort.resolve(references)
            val candidateExpressions = expressions[concreteSort] ?: TODO()
            return candidateExpressions.random(random)
        }
    }

    inner class ExpressionGenerator(
        val function: KFunction<*>,
        val refSortProviders: Map<String, SortProvider>,
        val argProviders: List<ArgumentProvider>
    ) {
        fun generate(context: Map<String, KSort> = emptyMap()): KExpr<*>? {
            val resolvedRefProviders = refSortProviders.mapValues {
                it.value.resolve(context)
            }
            val arguments = argProviders.map { it.provide(context + resolvedRefProviders) }

            val expr = try {
                function.call(ctx, *arguments.toTypedArray()) as KExpr<*>
            } catch (ex: InvocationTargetException) {
                System.err.println(ex.targetException.message)
                return null
            } catch (ex: IllegalArgumentException) {
                System.err.println(ex.message)
                return null
            }

            registerExpr(expr)
            return expr
        }
    }

    inner class SortGenerator(
        val function: KFunction<*>,
        val refSortProviders: Map<String, SortProvider>,
        val argProviders: List<ArgumentProvider>
    ) : ArgumentProvider {
        override fun provide(references: Map<String, KSort>): Any {
            val resolvedRefProviders = refSortProviders.mapValues {
                it.value.resolve(references)
            }
            val arguments = argProviders.map { it.provide(resolvedRefProviders) }
            return function.call(ctx, *arguments.toTypedArray()) as KSort
        }
    }

    private fun KClass<*>.generateSimpleValue(): Any = when (this) {
        Boolean::class -> random.nextBoolean()
        Byte::class -> random.nextInt().toByte()
        Short::class -> random.nextInt().toShort()
        Int::class -> random.nextInt()
        UInt::class -> random.nextUInt(3u..120u)
        Long::class -> random.nextLong()
        Float::class -> random.nextFloat()
        Double::class -> random.nextDouble()
        String::class -> randomString(10)
        else -> if (this.isSubclassOf(Enum::class)) {
            java.enumConstants.random(random)
        } else {
            error("Unexpected simple type: $this")
        }
    }

    private fun randomString(length: Int): String {
        val chars = CharArray(length) { ('a'..'z').random() }
        return String(chars)
    }
}


fun main() {
    val generator = RandomExpressionGenerator(KContext())
    val expr = generator.generate(1000)
    println(expr)
}
