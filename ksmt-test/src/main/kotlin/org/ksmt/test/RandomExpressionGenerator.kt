package org.ksmt.test

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast
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

    private val sortVariants = 10

    private val trace = arrayListOf<Any>()
    private val expressions = hashMapOf<KSort, MutableList<KExpr<*>>>()
    private val generators = arrayListOf<ExpressionGenerator>()

    fun generate(limit: Int): KExpr<*> {
        trace.clear()
        expressions.clear()
        mkGenerators()
        generateInitialSeed()

        generators.forEach { it.generate() }

        var expr: KExpr<*>
        var i = 0
        do {
            val generator = generators.random()
            expr = TODO()
            i++
        } while (i <= limit)
        return expr
    }

    private fun registerExpr(expr: KExpr<*>) {
        expressions.getOrPut(expr.sort) { arrayListOf() }.add(expr)
    }

    private fun mkGenerators() {
        if (generators.isNotEmpty()) return
        val expressionGenerators = ctx::class.members
            .filter { it.visibility == KVisibility.PUBLIC }
            .filterIsInstance<KFunction<*>>()
            .filter { it.returnType.isKExpr() }
            .mapNotNull { it.resolveArgumentSorts() }
        generators.addAll(expressionGenerators)
    }

    private fun generateInitialSeed() {
        val simpleSorts = ctx::class.members
            .filter { it.visibility == KVisibility.PUBLIC }
            .filterIsInstance<KFunction<*>>()
            .filter { it.returnType.isKSort() }
            .flatMap { it.generateSort() }
            .toSet()
        simpleSorts.forEach { it.mkSeed() }
    }

    private fun KSort.mkSeed() {
        if (this in expressions) return
        val seed = ctx.mkConst(randomString(10), this)
        registerExpr(seed)
    }

    private fun KFunction<*>.resolveArgumentSorts(): ExpressionGenerator? {
        if (parameters.isEmpty()) return null
        if (parameters.any { it.kind == KParameter.Kind.EXTENSION_RECEIVER }) return null
        if (parameters[0].kind != KParameter.Kind.INSTANCE || !parameters[0].type.isKContext()) return null
        if (parameters.drop(1).any { it.kind != KParameter.Kind.VALUE }) return null
        val valueParams = parameters.drop(1)


        val typeParametersProviders = hashMapOf<String, SortProvider>()

        typeParameters.forEach {
            it.mkReferenceSortProvider(typeParametersProviders)
        }

        val argumentProviders = arrayListOf<ArgumentProvider>()
        for (param in valueParams) {
            if (param.type.isKExpr()) {
                val sortProvider = param.type.mkKExprSortProvider(typeParametersProviders)
                argumentProviders += ExprProvider(sortProvider)
                continue
            }
            if (param.type.isKSort()) {
                val provider = param.type.mkSortProvider(typeParametersProviders)
                argumentProviders += provider
                continue
            }
            if (param.type.isSubclassOf(List::class)) {
                val elementType = param.type.arguments.single().type ?: return null
                if (!elementType.isKExpr()) return null
                val sortProvider = elementType.mkKExprSortProvider(typeParametersProviders)
                argumentProviders += ListProvider(sortProvider)
                continue
            }
            if (param.type.isSimple()) {
                val cls = param.type.classifier as? KClass<*> ?: return null
                argumentProviders += SimpleProvider(cls)
                continue
            }
            val sortClass = param.type.classifier
            if (sortClass is KTypeParameter && sortClass.name in typeParametersProviders) {
                val provider = param.type.mkSortProvider(typeParametersProviders)
                argumentProviders += provider
                continue
            }
            return null
        }
        return ExpressionGenerator(this, typeParametersProviders, argumentProviders)
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


    private fun KFunction<*>.generateSort(): List<KSort> {
        if (parameters.isEmpty()) return emptyList()
        if (!parameters[0].type.isKContext()) return emptyList()
        val params = parameters.drop(1)
        if (params.isEmpty()) return listOf(call(ctx) as KSort)
        val paramsVariants = (sortVariants / params.size) + 1
        val paramValues = params.map { it.type.generateSortParameter(paramsVariants) ?: return emptyList() }
        val allVariants = paramValues.crossProduct()
        return allVariants.map { call(ctx, *it.toTypedArray()) as KSort }
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
            val resolvedDomain = domain.resolve(references)
            val resolvedRange = range.resolve(references)
            return ctx.mkArraySort(resolvedDomain, resolvedRange).also {
                it.mkSeed()
            }
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
        fun generate() {
            val resolvedRefProviders = refSortProviders.mapValues {
                it.value.resolve(emptyMap())
            }
            val arguments = argProviders.map { it.provide(resolvedRefProviders) }
            println(arguments)
        }
    }

    private fun KType.generateSortParameter(variants: Int): List<Any>? = when (this) {
        UInt::class.createType() -> List(variants) { random.nextUInt(3u..120u) }
        Int::class.createType() -> List(variants) { random.nextInt(3..120) }
        String::class.createType() -> List(variants) { randomString(7) }
        else -> null
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

    private fun <T> List<List<T>>.crossProduct(): List<List<T>> {
        val result = Array<Any?>(size) { null }
        return sequence {
            crossProductUtil(result, 0, this@crossProduct)
        }.toList()
    }

    private suspend fun <T> SequenceScope<List<T>>.crossProductUtil(
        result: Array<Any?>,
        idx: Int,
        data: List<List<T>>
    ) {
        if (idx == data.size) {
            yield(result.toList().uncheckedCast())
            return
        }
        val values = data[idx]
        for (v in values) {
            result[idx] = v
            crossProductUtil(result, idx + 1, data)
        }
    }
}


fun main() {
    val generator = RandomExpressionGenerator(KContext())
    val expr = generator.generate(10)
    println(expr)
}
