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
