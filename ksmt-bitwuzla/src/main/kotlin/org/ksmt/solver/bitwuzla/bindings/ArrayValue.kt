package org.ksmt.solver.bitwuzla.bindings

/**
 * @param indices List of indices of size `size`. 1:1 mapping to `values`,
 * i.e., `index(i) -> value(i)`.
 * @param values List of values of size `size`.
 * @param size Size of `indices` and `values` list.
 * @param defaultValue The value of all other indices not in `indices` and
 * is set when base array is a constant array.
 */
class ArrayValue {
    @JvmField
    var size: Int = 0
    @JvmField
    var indices: LongArray? = null //Array of BitwuzlaTerms
    @JvmField
    var values: LongArray? = null //Array of BitwuzlaTerms
    @JvmField
    var defaultValue: BitwuzlaTerm = 0
}
