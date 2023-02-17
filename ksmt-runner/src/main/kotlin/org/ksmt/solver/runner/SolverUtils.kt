package org.ksmt.solver.runner

import org.ksmt.KContext
import org.ksmt.runner.generated.models.SolverType
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverUniversalConfigurationBuilder
import kotlin.reflect.KClass

typealias ConfigurationBuilder<C> = (KSolverUniversalConfigurationBuilder) -> C

private const val KSMT_SOLVER_PACKAGE = "org.ksmt.solver"

private const val Z3_SOLVER_CLASS_NAME = "$KSMT_SOLVER_PACKAGE.z3.KZ3Solver"
private const val Z3_UNIVERSAL_CONFIG_CLASS_NAME = "$KSMT_SOLVER_PACKAGE.z3.KZ3SolverUniversalConfiguration"

private const val BITWUZLA_SOLVER_CLASS_NAME = "$KSMT_SOLVER_PACKAGE.bitwuzla.KBitwuzlaSolver"
private const val BITWUZLA_UNIVERSAL_CONFIG_CLASS_NAME =
    "$KSMT_SOLVER_PACKAGE.bitwuzla.KBitwuzlaSolverUniversalConfiguration"

private const val YICES_SOLVER_CLASS_NAME = "$KSMT_SOLVER_PACKAGE.yices.KYicesSolver"
private const val YICES_UNIVERSAL_CONFIG_CLASS_NAME =
    "$KSMT_SOLVER_PACKAGE.yices.KYicesSolverUniversalConfiguration"

val KClass<out KSolver<*>>.solverType: SolverType
    get() = when (qualifiedName) {
        Z3_SOLVER_CLASS_NAME -> SolverType.Z3
        BITWUZLA_SOLVER_CLASS_NAME -> SolverType.Bitwuzla
        YICES_SOLVER_CLASS_NAME -> SolverType.Yices
        else -> error("Unsupported solver: ${this.qualifiedName}")
    }

private val z3SolverConstructor: (KContext) -> KSolver<*> by lazy {
    val cls = Class.forName(Z3_SOLVER_CLASS_NAME)
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver<*> })
}

private val z3ConfigurationConstructor: (KSolverUniversalConfigurationBuilder) -> KSolverConfiguration by lazy {
    val cls = Class.forName(Z3_UNIVERSAL_CONFIG_CLASS_NAME)
    val ctor = cls.getConstructor(KSolverUniversalConfigurationBuilder::class.java)
    ({ builder: KSolverUniversalConfigurationBuilder -> ctor.newInstance(builder) as KSolverConfiguration })
}

private val bitwuzlaSolverConstructor: (KContext) -> KSolver<*> by lazy {
    val cls = Class.forName(BITWUZLA_SOLVER_CLASS_NAME)
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver<*> })
}

private val bitwuzlaConfigurationConstructor: (KSolverUniversalConfigurationBuilder) -> KSolverConfiguration by lazy {
    val cls = Class.forName(BITWUZLA_UNIVERSAL_CONFIG_CLASS_NAME)
    val ctor = cls.getConstructor(KSolverUniversalConfigurationBuilder::class.java)
    ({ builder: KSolverUniversalConfigurationBuilder -> ctor.newInstance(builder) as KSolverConfiguration })
}

private val yicesSolverConstructor: (KContext) -> KSolver<*> by lazy {
    val cls = Class.forName(YICES_SOLVER_CLASS_NAME)
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver<*> })
}

private val yicesConfigurationConstructor: (KSolverUniversalConfigurationBuilder) -> KSolverConfiguration by lazy {
    val cls = Class.forName(YICES_UNIVERSAL_CONFIG_CLASS_NAME)
    val ctor = cls.getConstructor(KSolverUniversalConfigurationBuilder::class.java)
    ({ builder: KSolverUniversalConfigurationBuilder -> ctor.newInstance(builder) as KSolverConfiguration })
}

fun SolverType.createInstance(ctx: KContext): KSolver<*> = when (this) {
    SolverType.Z3 -> z3SolverConstructor(ctx)
    SolverType.Bitwuzla -> bitwuzlaSolverConstructor(ctx)
    SolverType.Yices -> yicesSolverConstructor(ctx)
}

@Suppress("UNCHECKED_CAST")
fun <C : KSolverConfiguration> KClass<out KSolver<C>>.createConfigurationBuilder(): ConfigurationBuilder<C> =
    when (solverType) {
        SolverType.Z3 -> { builder -> z3ConfigurationConstructor(builder) as C }
        SolverType.Bitwuzla -> { builder -> bitwuzlaConfigurationConstructor(builder) as C }
        SolverType.Yices -> { builder -> yicesConfigurationConstructor(builder) as C }
    }
