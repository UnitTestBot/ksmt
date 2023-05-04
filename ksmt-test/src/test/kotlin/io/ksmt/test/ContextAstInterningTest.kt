package io.ksmt.test

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import io.ksmt.KContext
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

class ContextAstInterningTest {

    @Test
    fun testSingleThreadNoGcContextInterning() = KContext(
        operationMode = KContext.OperationMode.SINGLE_THREAD,
        astManagementMode = KContext.AstManagementMode.NO_GC
    ).runSingleThread(::testExpressionsEquality)

    @Test
    fun testSingleThreadWithGcContextInterning() = KContext(
        operationMode = KContext.OperationMode.SINGLE_THREAD,
        astManagementMode = KContext.AstManagementMode.GC
    ).runSingleThread(::testExpressionsEquality)

    @Test
    fun testConcurrentNoGcContextInterning() = KContext(
        operationMode = KContext.OperationMode.CONCURRENT,
        astManagementMode = KContext.AstManagementMode.NO_GC
    ).runParallel(::testExpressionsEquality)

    @Test
    fun testConcurrentWithGcContextInterning() = KContext(
        operationMode = KContext.OperationMode.CONCURRENT,
        astManagementMode = KContext.AstManagementMode.GC
    ).runParallel(::testExpressionsEquality)

    private fun KContext.runSingleThread(test: (KContext, RandomExpressionGenerator) -> Unit) =
        generators.forEach { test(this, it) }

    private fun KContext.runParallel(test: (KContext, RandomExpressionGenerator) -> Unit) = runBlocking {
        generators.map {
            async(Dispatchers.Default) { test(this@runParallel, it) }
        }.joinAll()
    }

    private fun testExpressionsEquality(ctx: KContext, generator: RandomExpressionGenerator) {
        val e1 = generator.replay(ctx)
        val e2 = generator.replay(ctx)

        assertEquals(e1, e2)
        e1.zip(e2).forEach {
            assertSame(it.first, it.second)
        }
    }

    companion object {
        private val generators = mutableListOf<RandomExpressionGenerator>()

        @BeforeAll
        @JvmStatic
        fun setupGenerators() {
            val generationCtx = KContext(KContext.OperationMode.SINGLE_THREAD, KContext.AstManagementMode.NO_GC)
            val random = Random(42)

            repeat(20) {
                generators += RandomExpressionGenerator().apply {
                    generate(
                        10000, generationCtx, random,
                        generatorFilter = {
                            RandomExpressionGenerator.noFreshConstants(it)
                                // Disabled because can be slow on fp with big exponent
                                && RandomExpressionGenerator.noFpRem(it)
                        }
                    )
                }
            }
        }

        @AfterAll
        @JvmStatic
        fun cleanupGenerators() {
            generators.clear()
        }
    }
}
