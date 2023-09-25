package io.ksmt.solver.z3

import io.ksmt.utils.library.NativeLibraryLoader

interface KZ3NativeLibraryLoader : NativeLibraryLoader {
    override val libraryDir: String get() = "Z3"
}
