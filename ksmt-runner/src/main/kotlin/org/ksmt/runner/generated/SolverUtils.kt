package org.ksmt.runner.generated

import org.ksmt.KContext
import org.ksmt.runner.generated.models.SolverType
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverUniversalConfigurationBuilder
import kotlin.reflect.KClass

typealias ConfigurationBuilder<C> = (KSolverUniversalConfigurationBuilder) -> C

private val solverConstructorZ3: (KContext) -> KSolver<*> by lazy {
    val cls = Class.forName("org.ksmt.solver.z3.KZ3Solver")
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver<*> })
}

private val configConstructorZ3: (KSolverUniversalConfigurationBuilder) -> KSolverConfiguration by lazy {
    val cls = Class.forName("org.ksmt.solver.z3.KZ3SolverUniversalConfiguration")
    val ctor = cls.getConstructor(KSolverUniversalConfigurationBuilder::class.java)
    ({ builder: KSolverUniversalConfigurationBuilder -> ctor.newInstance(builder) as KSolverConfiguration })
}

private val solverConstructorBitwuzla: (KContext) -> KSolver<*> by lazy {
    val cls = Class.forName("org.ksmt.solver.bitwuzla.KBitwuzlaSolver")
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver<*> })
}

private val configConstructorBitwuzla: (KSolverUniversalConfigurationBuilder) -> KSolverConfiguration by lazy {
    val cls = Class.forName("org.ksmt.solver.bitwuzla.KBitwuzlaSolverUniversalConfiguration")
    val ctor = cls.getConstructor(KSolverUniversalConfigurationBuilder::class.java)
    ({ builder: KSolverUniversalConfigurationBuilder -> ctor.newInstance(builder) as KSolverConfiguration })
}

private val solverConstructorYices: (KContext) -> KSolver<*> by lazy {
    val cls = Class.forName("org.ksmt.solver.yices.KYicesSolver")
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver<*> })
}

private val configConstructorYices: (KSolverUniversalConfigurationBuilder) -> KSolverConfiguration by lazy {
    val cls = Class.forName("org.ksmt.solver.yices.KYicesSolverUniversalConfiguration")
    val ctor = cls.getConstructor(KSolverUniversalConfigurationBuilder::class.java)
    ({ builder: KSolverUniversalConfigurationBuilder -> ctor.newInstance(builder) as KSolverConfiguration })
}

private val solverTypes = mapOf(
    "org.ksmt.solver.z3.KZ3Solver" to SolverType.Z3,
    "org.ksmt.solver.bitwuzla.KBitwuzlaSolver" to SolverType.Bitwuzla,
    "org.ksmt.solver.yices.KYicesSolver" to SolverType.Yices,
)

val KClass<out KSolver<*>>.solverType: SolverType
    get() = solverTypes[qualifiedName] ?: error("Unsupported solver: ${qualifiedName}")

fun SolverType.createInstance(ctx: KContext): KSolver<*> = when (this) {
    SolverType.Z3 -> solverConstructorZ3(ctx)
    SolverType.Bitwuzla -> solverConstructorBitwuzla(ctx)
    SolverType.Yices -> solverConstructorYices(ctx)
}

@Suppress("UNCHECKED_CAST")
fun <C : KSolverConfiguration> SolverType.createConfigurationBuilder(): ConfigurationBuilder<C> = when (this) {
    SolverType.Z3 -> { builder -> configConstructorZ3(builder) as C }
    SolverType.Bitwuzla -> { builder -> configConstructorBitwuzla(builder) as C }
    SolverType.Yices -> { builder -> configConstructorYices(builder) as C }
}

