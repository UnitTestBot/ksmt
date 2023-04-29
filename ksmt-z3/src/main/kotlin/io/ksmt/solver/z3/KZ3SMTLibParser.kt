package io.ksmt.solver.z3

import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Z3Exception
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.parser.KSMTLibParseException
import io.ksmt.parser.KSMTLibParser
import io.ksmt.sort.KBoolSort
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
        KZ3Context(ctx).use {
            it.convertAssertions(it.nativeContext.parser().toList())
        }
    } catch (ex: Z3Exception) {
        throw KSMTLibParseException(ex)
    }

    private fun KZ3Context.convertAssertions(assertions: List<BoolExpr>): List<KExpr<KBoolSort>> {
        val converter = KZ3ExprConverter(ctx, this)
        return with(converter) { assertions.map { nativeContext.unwrapAST(it).convertExpr() } }
    }

    companion object {
        init {
            // ensure z3 native library is loaded
            KZ3Solver(KContext()).close()
        }
    }
}
