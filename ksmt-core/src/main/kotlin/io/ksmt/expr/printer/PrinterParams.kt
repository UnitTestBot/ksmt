package io.ksmt.expr.printer

data class PrinterParams(
    val bvValuePrintMode: BvValuePrintMode = BvValuePrintMode.HEX,
)

/**
 * Customize printing of Bv values.
 *
 * [BINARY] --- print all values in binary format.
 * All values are prefixed with a binary representation marker `#b`, e.g. `#b1010`.
 *
 * [HEX] --- print values in hexadecimal format if the number of bits
 * is a multiple of 4 (the number of bits represented by one character in hex format).
 * Otherwise, values are printed as in [BINARY] mode.
 * All hexadecimal values are prefixed with a hex representation marker `#x`, e.g. `#xbfaa`.
 * */
enum class BvValuePrintMode {
    BINARY,
    HEX
}
