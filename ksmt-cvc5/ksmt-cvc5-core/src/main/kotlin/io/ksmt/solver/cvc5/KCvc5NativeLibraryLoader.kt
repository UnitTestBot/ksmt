package io.ksmt.solver.cvc5

import io.ksmt.utils.library.NativeLibraryLoader

interface KCvc5NativeLibraryLoader : NativeLibraryLoader {
    override val libraryDir: String get() = "cvc5"
}
