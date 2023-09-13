package org.ksmt.solver.bitwuzla.bindings

import io.ksmt.utils.library.NativeLibraryLoader

interface KBitwuzlaNativeLibraryLoader : NativeLibraryLoader {
    override val libraryDir: String get() = "Bitwuzla"
}
