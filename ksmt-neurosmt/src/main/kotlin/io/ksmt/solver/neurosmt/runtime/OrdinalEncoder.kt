package io.ksmt.solver.neurosmt.runtime

import java.nio.file.Files
import java.nio.file.Path
import kotlin.streams.asSequence

const val UNKNOWN_VALUE = 1999

// wrapper for single-feature sklearn OrdinalEncoder (for each string we should provide its ordinal)
// used to convert strings to integers
class OrdinalEncoder(ordinalsPath: String, private val unknownValue: Int = UNKNOWN_VALUE) {
    private val lookup = HashMap<String, Int>()

    init {
        Files.lines(Path.of(ordinalsPath)).asSequence().forEachIndexed { index, s ->
            lookup[s] = index
        }
    }

    fun getOrdinal(s: String): Int {
        return lookup[s] ?: unknownValue
    }
}