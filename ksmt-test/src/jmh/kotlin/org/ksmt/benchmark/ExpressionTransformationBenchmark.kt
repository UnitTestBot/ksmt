package org.ksmt.benchmark

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.test.GenerationParameters
import org.ksmt.test.RandomExpressionGenerator
import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Fork
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.annotations.Measurement
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.Warmup
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit
import kotlin.random.Random

open class ExpressionTransformationBenchmark {

    @Benchmark
    @Fork(1)
    @BenchmarkMode(Mode.AverageTime)
    @Warmup(iterations = 5)
    @Measurement(iterations = 10, time = 60, timeUnit = TimeUnit.SECONDS)
    fun expressionCreationBenchmark(state: ExprTransformationState, bh: Blackhole) {
        val transformer = VisitAllExpressionsTransformer(state.ctx)
        val transformed = state.expressions.map { transformer.apply(it) }
        bh.consume(transformed)
    }

    class VisitAllExpressionsTransformer(ctx: KContext) : KNonRecursiveTransformer(ctx)

    @State(Scope.Benchmark)
    open class ExprTransformationState {
        private val generator = RandomExpressionGenerator().apply {
            generate(
                limit = 300_000,
                context = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY),
                random = Random(42),
                params = GenerationParameters().copy(deepExpressionProbability = 0.7),
                generatorFilter = RandomExpressionGenerator.noFreshConstants
            )
        }

        lateinit var ctx: KContext
        lateinit var expressions: List<KExpr<*>>

        @Setup(Level.Iteration)
        fun prepareExpressions() {
            ctx = KContext(
                operationMode = KContext.OperationMode.SINGLE_THREAD,
                astManagementMode = KContext.AstManagementMode.NO_GC,
                simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY
            )

            val allExpressions = generator.replay(ctx)
            expressions = allExpressions.takeLast(1000)
        }
    }
}
