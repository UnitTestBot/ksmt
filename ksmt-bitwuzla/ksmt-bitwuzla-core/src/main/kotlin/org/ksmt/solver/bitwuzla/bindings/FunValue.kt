package org.ksmt.solver.bitwuzla.bindings

/**
 * @param args List of argument lists (nested lists) of size `size`. Each
 * argument list is of size `arity`.
 * @param arity Size of each argument list in `args`.
 * @param values List of values of size `size`.
 * @param size Size of `indices` and `values` list.
 *
 * **Usage**
 * ```
 * for (int i = 0; i < size; ++i)
 * {
 *   // args[i] are argument lists of size arity
 *   for (int j = 0; j < arity; ++j)
 *   {
 *     // args[i][j] corresponds to value of jth argument of function f
 *   }
 *   // values[i] corresponds to the value of
 *   // (f args[i][0] ... args[i][arity - 1])
 * }
 * ```
 */
class FunValue {
    @JvmField
    var size: Int = 0
    @JvmField
    var arity: Int = 0
    @JvmField
    var args: Array<LongArray>? = null //Array of BitwuzlaTerm
    @JvmField
    var values: LongArray? = null //Array of BitwuzlaTerm
}
