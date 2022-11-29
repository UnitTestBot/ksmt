package org.ksmt.parser

import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import java.nio.file.Path

interface KSMTLibParser {
    /**
     * Parse the given file [smtLibFile] using the SMT-LIB2 parser.
     * @return a list of assertions.
     * @throws KSMTLibParseException if [smtLibFile] does not conform to the SMT-LIB2 format.
     *
     * Note. If the string contains push/pop commands,
     * the set of assertions returned are the ones in the last scope level.
     * */
    fun parse(smtLibFile: Path): List<KExpr<KBoolSort>>

    /**
     * Parse the given string [smtLibString] using the SMT-LIB2 parser.
     * @return a list of assertions.
     * @throws KSMTLibParseException if [smtLibString] does not conform to the SMT-LIB2 format.
     *
     * Note. If the string contains push/pop commands,
     * the set of assertions returned are the ones in the last scope level.
     * */
    fun parse(smtLibString: String): List<KExpr<KBoolSort>>
}
