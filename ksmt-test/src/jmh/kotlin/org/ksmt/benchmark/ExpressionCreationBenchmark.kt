package org.ksmt.benchmark

import org.ksmt.KContext
import org.ksmt.test.RandomExpressionGenerator
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Fork
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.annotations.Measurement
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.Warmup
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit
import kotlin.random.Random

open class ExpressionCreationBenchmark {

    @Benchmark
    @Fork(5)
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 5)
    @Measurement(iterations = 10, time = 20, timeUnit = TimeUnit.SECONDS)
    fun expressionCreationBenchmark(state: ExprGenerationState, bh: Blackhole) {
        val ctx = KContext(
            operationMode = state.contextOperationMode,
            astManagementMode = state.contextAstManagementMode
        )

        bh.consume(state.generator.replay(ctx))
    }

    @State(Scope.Thread)
    open class ExprGenerationState {
        @Param(
            "SINGLE_THREAD",
            "CONCURRENT"
        )
        lateinit var contextOperationMode: KContext.OperationMode

        @Param(
            "GC",
            "NO_GC"
        )
        lateinit var contextAstManagementMode: KContext.AstManagementMode

        val generator = RandomExpressionGenerator()

        @Setup(Level.Iteration)
        fun regenerateExpressions() {
            val random = Random(42)
            generator.generate(limit = 100000, KContext(), random)
        }
    }
}
