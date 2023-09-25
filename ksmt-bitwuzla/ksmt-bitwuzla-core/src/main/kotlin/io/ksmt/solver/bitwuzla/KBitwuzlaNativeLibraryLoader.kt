package io.ksmt.solver.bitwuzla

import io.ksmt.utils.library.NativeLibraryLoader

interface KBitwuzlaNativeLibraryLoader : NativeLibraryLoader {
    override val libraryDir: String get() = "Bitwuzla"
}
