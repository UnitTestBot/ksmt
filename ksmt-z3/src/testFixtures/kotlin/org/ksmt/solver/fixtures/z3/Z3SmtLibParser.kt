package org.ksmt.solver.fixtures.z3

import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.fixtures.SmtLibParser
import org.ksmt.solver.z3.KZ3ExprConverter
import org.ksmt.solver.z3.KZ3InternalizationContext
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import java.nio.file.Path

class Z3SmtLibParser : SmtLibParser {
    override fun parse(ctx: KContext, path: Path): List<KExpr<KBoolSort>> =
        Context().use { parseCtx ->
            val assertions = parseFile(parseCtx, path)
            convert(ctx, assertions)
        }

    fun parseFile(ctx: Context, path: Path): List<BoolExpr> = try {
        ctx.parseSMTLIB2File(
            path.toAbsolutePath().toString(),
            emptyArray(),
            emptyArray(),
            emptyArray(),
            emptyArray()
        ).toList()
    } catch (ex: Exception) {
        throw SmtLibParser.ParseError(ex)
    }

    fun convert(ctx: KContext, assertions: List<BoolExpr>): List<KExpr<KBoolSort>> {
        val internCtx = KZ3InternalizationContext()
        val converter = KZ3ExprConverter(ctx, internCtx)

        return with(converter) { assertions.map { it.convert() } }
    }

    companion object {
        init {
            // ensure loading of native library provided by ksmt
            KZ3Solver(KContext()).close()
        }
    }
}
