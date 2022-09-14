package org.ksmt.solver.fixtures

import org.junit.jupiter.api.Assumptions
import org.ksmt.solver.KSolverUnsupportedFeatureException

inline fun skipUnsupportedSolverFeatures(body: () -> Unit) = try {
    body()
} catch (ex: NotImplementedError) {
    val reducedStackTrace = ex.stackTrace.take(5).joinToString("\n") { it.toString() }
    val report = "${ex.message}\n$reducedStackTrace"

    System.err.println(report)
    // skip test with not implemented feature

    Assumptions.assumeTrue(false, ex.message)
} catch (ex: KSolverUnsupportedFeatureException) {
    Assumptions.assumeTrue(false, ex.message)
}

inline fun <reified T> parseAndSkipTestIfError(parse: () -> T) = try {
    parse()
} catch (ex: SmtLibParser.ParseError) {
    val testIgnoreReason = "parse failed -- ${ex.message}"

    System.err.println(testIgnoreReason)

    Assumptions.assumeTrue(false, testIgnoreReason)
    /**
     * assumeTrue throws an exception,
     * but we need something with [Nothing] return type
     * */
    error("ignored")
}
