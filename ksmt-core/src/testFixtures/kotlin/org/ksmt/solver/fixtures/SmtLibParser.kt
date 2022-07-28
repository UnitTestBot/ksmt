package org.ksmt.solver.fixtures

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import java.nio.file.Path

interface SmtLibParser {
    /**
     * Parse SmtLib file and return assertions.
     * @throws [ParseError] if parsing failed with exception.
     * */
    fun parse(ctx: KContext, path: Path): List<KExpr<KBoolSort>>

    class ParseError(cause: Throwable): Exception(cause)
}
