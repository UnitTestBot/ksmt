import org.ksmt.KContext
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverUniversalConfigurationBuilder
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.bitwuzla.KBitwuzlaSolverUniversalConfiguration
import org.ksmt.solver.yices.KYicesSolver
import org.ksmt.solver.yices.KYicesSolverUniversalConfiguration
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.solver.z3.KZ3SolverUniversalConfiguration
import kotlin.io.path.Path
import kotlin.io.path.bufferedWriter
import kotlin.reflect.KClass
import kotlin.reflect.full.createType

val solvers = listOf(
    SolverDescription("Z3", KZ3Solver::class, KZ3SolverUniversalConfiguration::class),
    SolverDescription("Bitwuzla", KBitwuzlaSolver::class, KBitwuzlaSolverUniversalConfiguration::class),
    SolverDescription("Yices", KYicesSolver::class, KYicesSolverUniversalConfiguration::class),
)

data class SolverDescription(
    val solverType: String,
    val solverCls: KClass<out KSolver<*>>,
    val solverUniversalConfig: KClass<out KSolverConfiguration>
)

private const val SOLVER_TYPE = "SolverType"
private const val SOLVER_TYPE_QUALIFIED_NAME = "org.ksmt.runner.generated.models.SolverType"

private val kSolverTypeName = "${KSolver::class.simpleName}<*>"
private val kContextTypeName = "${KContext::class.simpleName}"
private val kSolverConfigTypeName = "${KSolverConfiguration::class.simpleName}"
private val kSolverConfigBuilderTypeName = "${KSolverUniversalConfigurationBuilder::class.simpleName}"
private const val DOLLAR = "$"

private const val CONFIG_CONSTRUCTOR = "configConstructor"
private const val SOLVER_CONSTRUCTOR = "solverConstructor"

private fun generateHeader(packageName: String) = """
    package $packageName

    import ${KContext::class.qualifiedName}
    import $SOLVER_TYPE_QUALIFIED_NAME
    import ${KSolver::class.qualifiedName}
    import ${KSolverConfiguration::class.qualifiedName}
    import ${KSolverUniversalConfigurationBuilder::class.qualifiedName}
    import ${KClass::class.qualifiedName}

    typealias ConfigurationBuilder<C> = ($kSolverConfigBuilderTypeName) -> C
""".trimIndent()

private fun checkHasSuitableSolverConstructor(solver: SolverDescription) {
    val constructor = solver.solverCls.constructors
        .filter { it.parameters.size == 1 }
        .find { it.parameters.single().type == KContext::class.createType() }

    check(constructor != null) { "No constructor for solver $solver" }
}

private fun checkHasSuitableConfigConstructor(solver: SolverDescription) {
    val constructor = solver.solverUniversalConfig.constructors
        .filter { it.parameters.size == 1 }
        .find { it.parameters.single().type == KSolverUniversalConfigurationBuilder::class.createType() }

    check(constructor != null) { "No constructor for solver $solver" }
}

private fun generateSolverConstructor(solver: SolverDescription): String {
    checkHasSuitableSolverConstructor(solver)

    return """
        private val $SOLVER_CONSTRUCTOR${solver.solverType}: ($kContextTypeName) -> $kSolverTypeName by lazy {
            val cls = Class.forName("${solver.solverCls.qualifiedName}")
            val ctor = cls.getConstructor($kContextTypeName::class.java)
            ({ ctx: $kContextTypeName -> ctor.newInstance(ctx) as $kSolverTypeName })
        }
    """.trimIndent()
}

@Suppress("MaxLineLength")
private fun generateConfigConstructor(solver: SolverDescription): String {
    checkHasSuitableConfigConstructor(solver)

    return """
        private val $CONFIG_CONSTRUCTOR${solver.solverType}: ($kSolverConfigBuilderTypeName) -> $kSolverConfigTypeName by lazy {
            val cls = Class.forName("${solver.solverUniversalConfig.qualifiedName}")
            val ctor = cls.getConstructor($kSolverConfigBuilderTypeName::class.java)
            ({ builder: $kSolverConfigBuilderTypeName -> ctor.newInstance(builder) as $kSolverConfigTypeName })
        }
    """.trimIndent()
}

private fun generateSolverTypeMapping(prefix: String) = solvers.joinToString("\n") {
    """$prefix"${it.solverCls.qualifiedName}" to $SOLVER_TYPE.${it.solverType},"""
}

private fun generateSolverTypeGetter(): String = """
    |private val solverTypes = mapOf(
    ${generateSolverTypeMapping(prefix = "|    ")}
    |)
    |
    |val KClass<out ${kSolverTypeName}>.solverType: $SOLVER_TYPE
    |    get() = solverTypes[qualifiedName] ?: error("Unsupported solver: $DOLLAR{qualifiedName}")
""".trimMargin()

private fun generateSolverInstanceCreation(prefix: String) = solvers.joinToString("\n") {
    """$prefix${SOLVER_TYPE}.${it.solverType} -> $SOLVER_CONSTRUCTOR${it.solverType}(ctx)"""
}

private fun generateSolverCreateInstance(): String = """
    |fun $SOLVER_TYPE.createInstance(ctx: $kContextTypeName): $kSolverTypeName = when (this) {
    ${generateSolverInstanceCreation(prefix = "|    ")}
    |}
""".trimMargin()

private fun generateConfigInstanceCreation(prefix: String) = solvers.joinToString("\n") {
    """$prefix${SOLVER_TYPE}.${it.solverType} -> { builder -> $CONFIG_CONSTRUCTOR${it.solverType}(builder) as C }"""
}

private fun generateConfigCreateInstance(): String = """
    |@Suppress("UNCHECKED_CAST")
    |fun <C : $kSolverConfigTypeName> $SOLVER_TYPE.createConfigurationBuilder(): ConfigurationBuilder<C> = when (this) {
    ${generateConfigInstanceCreation(prefix = "|    ")}
    |}
""".trimMargin()

fun main(args: Array<String>) {
    val (generatedFilePath, generatedFilePackage) = args

    Path(generatedFilePath, "SolverUtils.kt").bufferedWriter().use {
        it.appendLine(generateHeader(generatedFilePackage))
        it.newLine()

        solvers.forEach { solver ->
            it.appendLine(generateSolverConstructor(solver))
            it.newLine()
            it.appendLine(generateConfigConstructor(solver))
            it.newLine()
        }

        it.appendLine(generateSolverTypeGetter())
        it.newLine()

        it.appendLine(generateSolverCreateInstance())
        it.newLine()

        it.appendLine(generateConfigCreateInstance())
        it.newLine()
    }
}
