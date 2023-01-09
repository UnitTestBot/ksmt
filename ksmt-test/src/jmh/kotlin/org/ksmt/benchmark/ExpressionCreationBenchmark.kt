package org.ksmt.benchmark

import org.ksmt.KContext
import org.ksmt.test.RandomExpressionGenerator
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.annotations.Measurement
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.Warmup
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit

open class ExpressionCreationBenchmark {

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 5)
    @Measurement(iterations = 10, time = 20, timeUnit = TimeUnit.SECONDS)
    fun expressionCreationBenchmark(state: ExprGenerationState, bh: Blackhole) {
        val ctx = KContext()
        bh.consume(state.generator.replay(ctx))
    }

    @State(Scope.Thread)
    open class ExprGenerationState {
        val generator = RandomExpressionGenerator()

        @Setup(Level.Iteration)
        fun regenerateExpressions() {
            generator.generate(limit = 100000, KContext())
        }
    }
}
