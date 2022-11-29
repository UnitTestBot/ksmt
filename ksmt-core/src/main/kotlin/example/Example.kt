package example

import org.ksmt.KContext
import org.ksmt.expr.KBitVec8Value
import org.ksmt.sort.KBv16Sort
import org.ksmt.utils.mkConst

@Suppress("UNUSED_VARIABLE")
fun main() = with(KContext()) {
    val a = boolSort.mkConst("a")
    val b = intSort.mkConst("b")
    val c = mkArraySort(intSort, intSort).mkConst("e")
    val x = mkArraySort(intSort, mkArraySort(intSort, intSort)).mkConst("e")
    val e1 = (c.select(b) + (c.store(b, b).select(b) + b) eq c.select(b)) and a or !a
    val e2 = x.store(b, c).select(b).select(10.expr) ge 11.expr
    val z = 3

    val bv8 = mkBv(0.toByte())
    val byteBits = Byte.SIZE_BITS
    val sameBv8 = mkBv("0".repeat(byteBits), byteBits.toUInt()) as KBitVec8Value
    check(bv8 === sameBv8)

    val bv16Sort = mkBv16Sort()
    val sameBv16Sort = mkBvSort(16.toUInt()) as KBv16Sort
    check(bv16Sort === sameBv16Sort)

    val bv32Decl = mkBvDecl(0)
    val sameBv32Decl = mkBvDecl("0".repeat(32), 32.toUInt())
    check(sameBv32Decl === bv32Decl)

    check(bv32Decl.name == "#b${"0".repeat(32)}")
}
