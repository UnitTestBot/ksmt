package io.ksmt.solver.bitwuzla.bindings

import org.junit.jupiter.api.assertThrows
import kotlin.test.Test
import kotlin.test.assertEquals

class CheckBindings {
    @Test
    fun checkLinkage() {
        assertEquals("0.3.2-dev", Native.bitwuzlaVersion())
    }

    @Test
    fun checkAbortException() {
        assertThrows<BitwuzlaNativeException> {
            Native.bitwuzlaTermManagerDelete(termManager = 0L)
        }
    }

    @Test
    fun checkResultEnumValues() {
        for (result in BitwuzlaResult.values()) {
            val nativeStr = Native.bitwuzlaResultToString(result)
            val str = result.name.removePrefix("BITWUZLA_").lowercase()
            assertEquals(nativeStr, str)
        }
    }

    @Test
    fun checkRoundingModeEnumValues() {
        for (rm in BitwuzlaRoundingMode.values()) {
            val nativeStr = Native.bitwuzlaRmToString(rm)
            assertEquals(nativeStr, rm.name)
        }
        assertThrows<BitwuzlaNativeException> {
            Native.bitwuzlaRmToString(BitwuzlaRoundingMode.values().last().ordinal + 1)
        }
    }

    @Test
    fun checkKindEnumValues() {
        for (kind in BitwuzlaKind.values()) {
            val nativeStr = Native.bitwuzlaKindToString(kind)
            assertEquals(nativeStr, kind.name.replace("_OVERFLOW", "O"))
        }
        assertThrows<BitwuzlaNativeException> {
            Native.bitwuzlaKindToString(BitwuzlaKind.values().last().ordinal + 1)
        }
    }

    @Test
    fun checkOptionEnumValues() {
        for (option in BitwuzlaOption.values()) {
            val optionName = option.name.removePrefix("BITWUZLA_OPT_").replace('_', '-').lowercase()
            val nativeStr = Native.bitwuzlaOptionToString(option)
            assertEquals(nativeStr, optionName)
        }
        assertThrows<BitwuzlaNativeException> {
            Native.bitwuzlaOptionToString(BitwuzlaOption.values().last().ordinal + 1)
        }
    }
}
