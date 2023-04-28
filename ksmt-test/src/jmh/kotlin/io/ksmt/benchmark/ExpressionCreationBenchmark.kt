package io.ksmt.benchmark

import io.ksmt.KContext
import io.ksmt.test.RandomExpressionGenerator
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Fork
import org.openjdk.jmh.annotations.Measurement
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.Warmup
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.random.Random

open class ExpressionCreationBenchmark {

    @Benchmark
    @Fork(5)
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 5)
    @Measurement(iterations = 10, time = 60, timeUnit = TimeUnit.SECONDS)
    fun expressionCreationBenchmark(state: ExprGenerationState, bh: Blackhole) {
        val ctx = KContext(
            operationMode = state.contextOperationMode,
            astManagementMode = state.contextAstManagementMode,
            simplificationMode = state.contextSimplificationMode
        )

        bh.consume(state.generator.replay(ctx))
    }

    @Benchmark
    @Fork(5)
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 5)
    @Measurement(iterations = 10, time = 60, timeUnit = TimeUnit.SECONDS)
    fun expressionCreationWithCleanupsBenchmark(state: ExprGenerationWithCleanUpState, bh: Blackhole) {
        val ctx = KContext(
            operationMode = state.contextOperationMode,
            astManagementMode = state.contextAstManagementMode
        )

        for (generator in state.generators) {
            bh.consume(generator.replay(ctx))
        }
    }

    @Benchmark
    @Fork(5)
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 5)
    @Measurement(iterations = 10, time = 60, timeUnit = TimeUnit.SECONDS)
    fun expressionCreationConcurrentWithCleanupsBenchmark(
        state: ExprConcurrentGenerationWithCleanUpState,
        bh: Blackhole
    ) {
        val ctx = KContext(
            operationMode = KContext.OperationMode.CONCURRENT,
            astManagementMode = state.contextAstManagementMode
        )

        val tasks = state.generators.map { generator ->
            state.executor.submit {
                bh.consume(generator.replay(ctx))
            }
        }

        tasks.forEach { bh.consume(it.get()) }
    }

    @State(Scope.Benchmark)
    open class ExprGenerationState {
        @Param("SINGLE_THREAD", "CONCURRENT")
        lateinit var contextOperationMode: KContext.OperationMode

        @Param("NO_GC", "GC")
        lateinit var contextAstManagementMode: KContext.AstManagementMode

        @Param("NO_SIMPLIFY", "SIMPLIFY")
        lateinit var contextSimplificationMode: KContext.SimplificationMode

        val generator = RandomExpressionGenerator().apply {
            generate(
                limit = SINGLE_BENCHMARK_EXPRESSIONS,
                context = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY),
                random = Random(42),
                generatorFilter = RandomExpressionGenerator.noFreshConstants
            )
        }
    }

    @State(Scope.Benchmark)
    open class ExprGenerationWithCleanUpState {
        @Param("SINGLE_THREAD", "CONCURRENT")
        lateinit var contextOperationMode: KContext.OperationMode

        @Param("NO_GC", "GC")
        lateinit var contextAstManagementMode: KContext.AstManagementMode

        val generators = (1..GENERATORS_COUNT).map { seed ->
            RandomExpressionGenerator().apply {
                generate(
                    limit = SINGLE_BENCHMARK_EXPRESSIONS / GENERATORS_COUNT,
                    context = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY),
                    random = Random(seed),
                    generatorFilter = RandomExpressionGenerator.noFreshConstants
                )
            }
        }
    }

    @State(Scope.Benchmark)
    open class ExprConcurrentGenerationWithCleanUpState {
        @Param("NO_GC", "GC")
        lateinit var contextAstManagementMode: KContext.AstManagementMode

        val generators = (1..GENERATORS_COUNT).map { seed ->
            RandomExpressionGenerator().apply {
                generate(
                    limit = SINGLE_BENCHMARK_EXPRESSIONS / GENERATORS_COUNT,
                    context = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY),
                    random = Random(seed),
                    generatorFilter = RandomExpressionGenerator.noFreshConstants
                )
            }
        }

        val executor = Executors.newFixedThreadPool(THREADS_COUNT)
    }

    companion object {
        const val SINGLE_BENCHMARK_EXPRESSIONS = 200000
        const val GENERATORS_COUNT = 10
        const val THREADS_COUNT = 5
    }
}
