package org.ksmt.solver.z3

import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Z3Exception
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.parser.KSMTLibParseException
import org.ksmt.parser.KSMTLibParser
import org.ksmt.sort.KBoolSort
import java.nio.file.Path
import kotlin.io.path.absolutePathString

class KZ3SMTLibParser(private val ctx: KContext) : KSMTLibParser {
    override fun parse(smtLibFile: Path): List<KExpr<KBoolSort>> = parse {
        parseSMTLIB2File(
            smtLibFile.absolutePathString(),
            emptyArray(),
            emptyArray(),
            emptyArray(),
            emptyArray()
        )
    }

    override fun parse(smtLibString: String): List<KExpr<KBoolSort>> = parse {
        parseSMTLIB2String(
            smtLibString,
            emptyArray(),
            emptyArray(),
            emptyArray(),
            emptyArray()
        )
    }

    private fun parse(parser: Context.() -> Array<BoolExpr>) = try {
        Context().use {
            convertAssertions(it.parser().toList())
        }
    } catch (ex: Z3Exception) {
        throw KSMTLibParseException(ex)
    }

    private fun convertAssertions(assertions: List<BoolExpr>): List<KExpr<KBoolSort>> {
        val internCtx = KZ3InternalizationContext()
        val converter = KZ3ExprConverter(ctx, internCtx)
        return with(converter) { assertions.map { it.convert() } }
    }

    companion object {
        init {
            // ensure z3 native library is loaded
            KZ3Solver(KContext()).close()
        }
    }
}
