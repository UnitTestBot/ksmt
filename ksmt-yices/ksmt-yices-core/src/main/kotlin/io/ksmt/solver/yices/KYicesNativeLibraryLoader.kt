package io.ksmt.solver.yices

import io.ksmt.utils.library.NativeLibraryLoader

interface KYicesNativeLibraryLoader : NativeLibraryLoader {
    override val libraryDir: String get() = "yices"
}
