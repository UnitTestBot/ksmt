package org.ksmt.test

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KInterpretedConstant
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.uncheckedCast
import java.util.SortedMap
import kotlin.random.Random
import kotlin.random.nextInt
import kotlin.random.nextUInt
import kotlin.reflect.KClass
import kotlin.reflect.KClassifier
import kotlin.reflect.KFunction
import kotlin.reflect.KParameter
import kotlin.reflect.KType
import kotlin.reflect.KTypeParameter
import kotlin.reflect.KTypeProjection
import kotlin.reflect.KVisibility
import kotlin.reflect.full.allSupertypes
import kotlin.reflect.full.createType
import kotlin.reflect.full.isSubclassOf

class RandomExpressionGenerator {
    private lateinit var generationContext: GenerationContext

    fun generate(limit: Int, context: KContext, random: Random = Random(42)): List<KExpr<*>> {
        generationContext = GenerationContext(
            random = random,
            context = context,
            expressionsAmountEstimation = limit,
            sortsAmountEstimation = 1000
        )

        generateInitialSeed(samplesPerSort = seedExpressionsPerSort)

        var i = 0
        do {
            val generator = generators.random(random)

            nullIfGenerationFailed {
                generator.generate(generationContext)
            } ?: continue

            i++
        } while (i <= limit)

        return generationContext.expressions
    }

    fun replay(context: KContext): List<KExpr<*>> {
        val replayContext = GenerationContext(
            random = Random(0),
            context = context,
            expressionsAmountEstimation = generationContext.expressions.size,
            sortsAmountEstimation = generationContext.sorts.size
        )
        for (entry in generationContext.trace) {
            val resolvedEntry = resolveTraceEntry(entry, replayContext)
            val result = resolvedEntry.call(context)
            when (result) {
                // We don't care about expression depth since it is unused during replay
                is KExpr<*> -> replayContext.registerExpr(result, depth = 0, inReplayMode = true)
                is KSort -> replayContext.registerSort(result, inReplayMode = true)
            }
        }
        return replayContext.expressions
    }

    private fun resolveTraceEntry(
        entry: FunctionInvocation,
        replayContext: GenerationContext
    ): FunctionInvocation {
        val args = entry.args.map { resolveArgument(it, replayContext) }
        return FunctionInvocation(entry.function, args)
    }

    private fun resolveArgument(argument: Argument, replayContext: GenerationContext): Argument =
        when (argument) {
            is SimpleArgument -> argument
            is ListArgument -> ListArgument(argument.nested.map { resolveArgument(it, replayContext) })
            is ExprArgument -> ExprArgument(replayContext.expressions[argument.idx], argument.idx, argument.depth)
            is SortArgument -> SortArgument(replayContext.sorts[argument.idx], argument.idx)
        }

    private fun generateInitialSeed(samplesPerSort: Int) {
        for (sortGenerator in sortGenerators.filter { it.refSortProviders.isEmpty() }) {
            repeat(samplesPerSort) {
                nullIfGenerationFailed { sortGenerator.mkSeed(generationContext) }
            }
        }
        for (sortGenerator in sortGenerators.filter { it.refSortProviders.isNotEmpty() }) {
            repeat(samplesPerSort) {
                nullIfGenerationFailed { sortGenerator.mkSeed(generationContext) }
            }
        }
    }

    companion object {
        private const val seedExpressionsPerSort = 10
        private const val freshConstantExpressionProbability = 0.7
        private const val deepExpressionProbability = 0.4

        private val ctxFunctions by lazy {
            KContext::class.members
                .filter { it.visibility == KVisibility.PUBLIC }
                .filterIsInstance<KFunction<*>>()
        }

        private val generators by lazy {
            ctxFunctions
                .asSequence()
                .filter { it.returnType.isKExpr() }
                .map { it.uncheckedCast<KFunction<*>, KFunction<KExpr<*>>>() }
                .filterNot { (it.name == "mkBv" && it.parameters.any { p -> p.type.isSubclassOf(String::class) }) }
                .filterNot { (it.name == "mkIntNum" && it.parameters.any { p -> p.type.isSubclassOf(String::class) }) }
                .filterNot { (it.name == "mkRealNum" && it.parameters.any { p -> p.type.isSubclassOf(String::class) }) }
                .mapNotNull {
                    nullIfGenerationFailed {
                        it.mkGenerator()?.uncheckedCast<AstGenerator<*>, AstGenerator<ExprArgument>>()
                    }
                }
                .toList()
        }

        private val sortGenerators by lazy {
            ctxFunctions
                .filter { it.returnType.isKSort() }
                .map { it.uncheckedCast<KFunction<*>, KFunction<KSort>>() }
                .mapNotNull {
                    nullIfGenerationFailed {
                        it.mkGenerator()?.uncheckedCast<AstGenerator<*>, AstGenerator<SortArgument>>()
                    }
                }
        }

        private val boolGen by lazy { generators.single { it.function.match("mkBool", Boolean::class) } }
        private val intGen by lazy { generators.single { it.function.match("mkIntNum", Int::class) } }
        private val realGen by lazy { generators.single { it.function.match("mkRealNum", Int::class) } }
        private val bvGen by lazy { generators.single { it.function.match("mkBv", Int::class, UInt::class) } }
        private val fpGen by lazy { generators.single { it.function.match("mkFp", Double::class, KSort::class) } }
        private val arrayGen by lazy { generators.single { it.function.match("mkArrayConst", KSort::class, KExpr::class) } }
        private val fpRmGen by lazy { generators.single { it.function.match("mkFpRoundingModeExpr", KFpRoundingMode::class) } }
        private val constGen by lazy { generators.single { it.function.match("mkConst", String::class, KSort::class) } }
        private val arraySortGen by lazy { sortGenerators.single { it.function.match("mkArraySort", KSort::class, KSort::class) } }

        private fun KType.isSubclassOf(other: KClass<*>): Boolean =
            when (val cls = classifier) {
                is KClass<*> -> cls.isSubclassOf(other)
                is KTypeParameter -> cls.upperBounds.all { it.isSubclassOf(other) }
                else -> false
            }

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
        private fun KType.isConst(): Boolean = isSubclassOf(KInterpretedConstant::class)

        private fun KType.isKSort(): Boolean = isSubclassOf(KSort::class)

        private fun KType.isKContext(): Boolean = this == KContext::class.createType()

        private fun KFunction<*>.match(name: String, vararg valueParams: KClass<*>): Boolean {
            if (this.name != name) return false
            val actualValueParams = parameters.drop(1)
            if (actualValueParams.size != valueParams.size) return false
            return valueParams.zip(actualValueParams).all { (expect, actual) -> actual.type.isSubclassOf(expect) }
        }

        private fun <T : KAst> KFunction<T>.mkGenerator(): AstGenerator<*>? {
            if (!parametersAreCorrect()) return null
            val valueParams = parameters.drop(1)

            val typeParametersProviders = hashMapOf<String, SortProvider>()
            typeParameters.forEach {
                it.mkReferenceSortProvider(typeParametersProviders)
            }

            val argumentProviders = valueParams.map { it.mkArgProvider(typeParametersProviders) ?: return null }
            return AstGenerator<Argument>(this, typeParametersProviders, argumentProviders) { args ->
                when (name) {
                    "mkBvRepeatExpr" -> listOf(
                        SimpleArgument((args[0] as SimpleArgument).value as Int % 3),
                        args[1]
                    )

                    else -> args
                }
            }
        }

        private fun KFunction<*>.parametersAreCorrect(): Boolean {
            if (parameters.isEmpty()) return false
            if (parameters.any { it.kind == KParameter.Kind.EXTENSION_RECEIVER }) return false
            if (parameters[0].kind != KParameter.Kind.INSTANCE || !parameters[0].type.isKContext()) return false
            if (parameters.drop(1).any { it.kind != KParameter.Kind.VALUE }) return false
            return true
        }

        private fun KParameter.mkArgProvider(typeParametersProviders: MutableMap<String, SortProvider>): ArgumentProvider? {
            if (type.isKExpr()) {
                val sortProvider = type.mkKExprSortProvider(typeParametersProviders)
                return if (type.isConst()) {
                    ConstExprProvider(sortProvider)
                } else {
                    ExprProvider(sortProvider)
                }
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
                generationFailed("Not a KSort type argument")
            }
            references[name] = sortProvider
        }

        private fun KType.mkKExprSortProvider(references: MutableMap<String, SortProvider>): SortProvider {
            val expr = if (classifier == KExpr::class) {
                this
            } else {
                (classifier as? KClass<*>)?.allSupertypes?.find { it.classifier == KExpr::class }
                    ?: generationFailed("No KExpr superclass found")
            }
            val sort = expr.arguments.single()
            if (sort == KTypeProjection.STAR) return SingleSortProvider(KSort::class.java)
            val sortType = sort.type ?: generationFailed("No type available")
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
                val (domain, range) = arguments.map {
                    it.type?.mkSortProvider(references)
                        ?: generationFailed("Array sort type is not available")
                }
                return ArraySortProvider(domain, range)
            }
            if (this.isKSort() && sortClass != null) {
                return SingleSortProvider((sortClass.uncheckedCast<KClassifier, KClass<KSort>>()).java)
            }
            generationFailed("Unexpected type $this")
        }


        private class ConstExprGenerator(
            val sortArgument: SortArgument,
            val generationContext: GenerationContext
        ) : KSortVisitor<AstGenerator<ExprArgument>> {
            override fun visit(sort: KBoolSort): AstGenerator<ExprArgument> = boolGen
            override fun visit(sort: KIntSort): AstGenerator<ExprArgument> = intGen
            override fun visit(sort: KRealSort): AstGenerator<ExprArgument> = realGen
            override fun visit(sort: KFpRoundingModeSort): AstGenerator<ExprArgument> = fpRmGen

            override fun <S : KBvSort> visit(sort: S): AstGenerator<ExprArgument> =
                AstGenerator(bvGen.function, bvGen.refSortProviders, listOf(SimpleProvider(Int::class))) { args ->
                    args + SimpleArgument(sort.sizeBits)
                }

            override fun <S : KFpSort> visit(sort: S): AstGenerator<ExprArgument> =
                AstGenerator(fpGen.function, fpGen.refSortProviders, listOf(SimpleProvider(Double::class))) { args ->
                    args + sortArgument
                }

            override fun visit(sort: KUninterpretedSort): AstGenerator<ExprArgument> =
                AstGenerator(constGen.function, constGen.refSortProviders, listOf(SimpleProvider(String::class))) { args ->
                    args + sortArgument
                }

            override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): AstGenerator<ExprArgument> {
                val rangeSortArgument = SortArgument(sort.range, generationContext.findSortIdx(sort.range))
                val rangeExprGenerator = sort.range.accept(ConstExprGenerator(rangeSortArgument, generationContext))
                val rangeExpr = rangeExprGenerator.generate(generationContext)
                return AstGenerator(arrayGen.function, arrayGen.refSortProviders, emptyList()) {
                    listOf(sortArgument, rangeExpr)
                }
            }
        }

        sealed interface Argument {
            val value: Any
            val depth: Int
        }

        class ListArgument(val nested: List<Argument>) : Argument {
            override val value: Any
                get() = nested.map { it.value }

            override val depth: Int
                get() = nested.maxOf { it.depth }
        }

        class SimpleArgument(override val value: Any) : Argument {
            override val depth: Int = 0
        }

        class SortArgument(override val value: KSort, val idx: Int) : Argument {
            override val depth: Int = 0
        }

        class ExprArgument(override val value: KExpr<*>, val idx: Int, override val depth: Int) : Argument

        private sealed interface ArgumentProvider {
            fun provide(generationContext: GenerationContext, references: Map<String, SortArgument>): Argument
        }

        private sealed interface SortProvider : ArgumentProvider {
            override fun provide(generationContext: GenerationContext, references: Map<String, SortArgument>): Argument =
                resolve(generationContext, references)

            fun resolve(generationContext: GenerationContext, references: Map<String, SortArgument>): SortArgument
        }

        private class SingleSortProvider(val sort: Class<KSort>) : SortProvider {
            override fun resolve(generationContext: GenerationContext, references: Map<String, SortArgument>): SortArgument {
                val candidates = generationContext.sortIndex[sort] ?: generationFailed("No sort matching $sort")
                val idx = candidates.random(generationContext.random)
                return SortArgument(generationContext.sorts[idx], idx)
            }
        }

        private class ReferenceSortProvider(val reference: String) : SortProvider {
            override fun resolve(generationContext: GenerationContext, references: Map<String, SortArgument>): SortArgument {
                return references[reference] ?: generationFailed("Unresolved sort reference $references")
            }
        }

        private class ArraySortProvider(val domain: SortProvider, val range: SortProvider) : SortProvider {
            override fun resolve(generationContext: GenerationContext, references: Map<String, SortArgument>): SortArgument {
                val generationParams = mapOf(
                    "D" to domain,
                    "R" to range
                )
                val generator = AstGenerator<SortArgument>(
                    arraySortGen.function,
                    generationParams,
                    generationParams.keys.map { ReferenceSortProvider(it) }
                )
                val expr = generator.mkSeed(generationContext, references)
                val sort = expr.value.sort
                val sortIdx = generationContext.findSortIdx(sort)
                return SortArgument(generationContext.sorts[sortIdx], sortIdx)
            }
        }

        private class SimpleProvider(val type: KClass<*>) : ArgumentProvider {
            override fun provide(generationContext: GenerationContext, references: Map<String, SortArgument>): Argument {
                val value = type.generateSimpleValue(generationContext.random)
                return SimpleArgument(value)
            }
        }

        private class ListProvider(val element: SortProvider) : ArgumentProvider {
            override fun provide(generationContext: GenerationContext, references: Map<String, SortArgument>): Argument {
                val concreteSort = element.resolve(generationContext, references).value
                val size = generationContext.random.nextInt(2..10)
                val candidateExpressions = generationContext.expressionIndex[concreteSort]
                    ?: generationFailed("No expressions for sort $concreteSort")
                val nested = List(size) {
                    val (exprId, exprDepth) = selectRandomExpressionId(generationContext.random, candidateExpressions)
                    val expr = generationContext.expressions[exprId]
                    ExprArgument(expr, exprId, exprDepth)
                }
                return ListArgument(nested)
            }
        }

        private class ExprProvider(val sort: SortProvider) : ArgumentProvider {
            override fun provide(generationContext: GenerationContext, references: Map<String, SortArgument>): Argument {
                val concreteSort = sort.resolve(generationContext, references).value
                val candidateExpressions = generationContext.expressionIndex[concreteSort]
                    ?: generationFailed("No expressions for sort $concreteSort")
                val (exprId, exprDepth) = selectRandomExpressionId(generationContext.random, candidateExpressions)
                return ExprArgument(generationContext.expressions[exprId], exprId, exprDepth)
            }
        }

        private class ConstExprProvider(val sort: SortProvider) : ArgumentProvider {
            override fun provide(generationContext: GenerationContext, references: Map<String, SortArgument>): Argument {
                val concreteSort = sort.resolve(generationContext, references)
                val constants = generationContext.constantIndex[concreteSort.value] ?: emptyList()
                return if (
                    constants.isNotEmpty()
                    && generationContext.random.nextDouble() > freshConstantExpressionProbability
                ) {
                    val idx = constants.random(generationContext.random)
                    ExprArgument(generationContext.expressions[idx], idx, depth = 1)
                } else {
                    val generator = concreteSort.value.accept(ConstExprGenerator(concreteSort, generationContext))
                    generator.generate(generationContext, references)
                }
            }
        }

        private class AstGenerator<T : Argument>(
            val function: KFunction<*>,
            val refSortProviders: Map<String, SortProvider>,
            val argProviders: List<ArgumentProvider>,
            val provideArguments: (List<Argument>) -> List<Argument> = { it }
        ) : ArgumentProvider {
            override fun provide(generationContext: GenerationContext, references: Map<String, SortArgument>): Argument =
                generate(generationContext, references)

            fun generate(generationContext: GenerationContext, context: Map<String, SortArgument> = emptyMap()): T {
                val resolvedRefProviders = refSortProviders.mapValues {
                    it.value.resolve(generationContext, context)
                }
                val baseArguments = argProviders.map { it.provide(generationContext, context + resolvedRefProviders) }
                val arguments = provideArguments(baseArguments)

                val invocation = FunctionInvocation(function, arguments)

                val ast = try {
                    invocation.call(generationContext.context)
                } catch (ex: Throwable) {
                    throw GenerationFailedException("Generator failed", ex)
                }

                if (ast is KExpr<*>) {
                    if (!ast.isCorrect()) {
                        generationFailed("Incorrect ast generated")
                    }
                    val depth = (arguments.maxOfOrNull { it.depth } ?: 0) + 1
                    val idx = generationContext.registerExpr(ast, depth)
                    generationContext.trace += invocation
                    return ExprArgument(ast, idx, depth).uncheckedCast()
                }

                if (ast is KSort) {
                    var idx = generationContext.registerSort(ast)
                    if (idx == -1) {
                        idx = generationContext.findSortIdx(ast)
                    }
                    generationContext.trace += invocation
                    return SortArgument(ast, idx).uncheckedCast()
                }

                generationFailed("Unexpected generation result: $ast")
            }

            private fun KExpr<*>.isCorrect(): Boolean {
                val sort = sort
                if (sort is KBvSort && sort.sizeBits == 0u) return false
                return true
            }
        }

        private fun AstGenerator<SortArgument>.mkSeed(
            generationContext: GenerationContext,
            context: Map<String, SortArgument> = emptyMap()
        ): ExprArgument {
            val exprGenerator = AstGenerator<ExprArgument>(
                constGen.function,
                emptyMap(),
                listOf(SimpleProvider(String::class), this)
            )
            return exprGenerator.generate(generationContext, context)
        }

        private fun selectRandomExpressionId(
            random: Random,
            expressionIds: SortedMap<Int, MutableList<Int>>
        ): Pair<Int, Int> {
            val expressionDepth = when (random.nextDouble()) {
                in 0.0..deepExpressionProbability -> expressionIds.lastKey()
                else -> expressionIds.keys.random(random)
            }
            val candidateExpressions = expressionIds.getValue(expressionDepth)
            val exprId = candidateExpressions.random(random)
            return exprId to expressionDepth
        }

        private fun KClass<*>.generateSimpleValue(random: Random): Any = when (this) {
            Boolean::class -> random.nextBoolean()
            Byte::class -> random.nextInt().toByte()
            Short::class -> random.nextInt().toShort()
            Int::class -> random.nextInt(3..100)
            UInt::class -> random.nextUInt(3u..100u)
            Long::class -> random.nextLong()
            Float::class -> random.nextFloat()
            Double::class -> random.nextDouble()
            String::class -> randomString(random, length = 10)
            else -> if (this.isSubclassOf(Enum::class)) {
                java.enumConstants.random(random)
            } else {
                generationFailed("Unexpected simple type: $this")
            }
        }

        private fun randomString(random: Random, length: Int): String {
            val chars = CharArray(length) { ('a'..'z').random(random) }
            return String(chars)
        }

        private class FunctionInvocation(
            val function: KFunction<*>,
            val args: List<Argument>
        ) {
            fun call(ctx: KContext): Any? {
                val argumentValues = args.map { it.value }
                return function.call(ctx, *argumentValues.toTypedArray())
            }
        }

        private class GenerationContext(
            val random: Random,
            val context: KContext,
            expressionsAmountEstimation: Int,
            sortsAmountEstimation: Int
        ) {
            val expressions = ArrayList<KExpr<*>>(expressionsAmountEstimation)
            val sorts = ArrayList<KSort>(sortsAmountEstimation)
            val registeredSorts = HashMap<KSort, Int>(sortsAmountEstimation)
            val expressionIndex = hashMapOf<KSort, SortedMap<Int, MutableList<Int>>>()
            val constantIndex = hashMapOf<KSort, MutableList<Int>>()
            val sortIndex = hashMapOf<Class<*>, MutableList<Int>>()
            val trace = ArrayList<FunctionInvocation>(expressionsAmountEstimation + sortsAmountEstimation)

            fun registerSort(sort: KSort, inReplayMode: Boolean = false): Int {
                val knownSortId = registeredSorts[sort]

                val sortId = if (knownSortId != null) {
                    knownSortId
                } else {
                    val idx = sorts.size
                    sorts.add(sort)
                    registeredSorts[sort] = idx
                    idx
                }

                if (knownSortId != null || inReplayMode) {
                    return sortId
                }

                var sortCls: Class<*> = sort::class.java
                while (sortCls != KSort::class.java) {
                    sortIndex.getOrPut(sortCls) { arrayListOf() }.add(sortId)
                    sortCls = sortCls.superclass
                }
                sortIndex.getOrPut(sortCls) { arrayListOf() }.add(sortId)

                return sortId
            }

            fun registerExpr(expr: KExpr<*>, depth: Int, inReplayMode: Boolean = false): Int {
                registerSort(expr.sort, inReplayMode)

                val exprId = expressions.size
                expressions.add(expr)

                if (inReplayMode) {
                    return exprId
                }

                val index = expressionIndex.getOrPut(expr.sort) { sortedMapOf() }
                val expressionIds = index.getOrPut(depth) { arrayListOf() }
                expressionIds.add(exprId)

                if (expr is KInterpretedConstant) {
                    constantIndex.getOrPut(expr.sort) { arrayListOf() }.add(exprId)
                }

                return exprId
            }

            fun findSortIdx(sort: KSort): Int =
                registeredSorts[sort] ?: generationFailed("No idx for sort $sort")
        }
    }
}

@Suppress("SwallowedException")
private inline fun <T> nullIfGenerationFailed(body: () -> T): T? = try {
    body()
} catch (ex: GenerationFailedException) {
    null
}

private class GenerationFailedException : Exception {
    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)
}

private fun generationFailed(message: String): Nothing =
    throw GenerationFailedException(message)
