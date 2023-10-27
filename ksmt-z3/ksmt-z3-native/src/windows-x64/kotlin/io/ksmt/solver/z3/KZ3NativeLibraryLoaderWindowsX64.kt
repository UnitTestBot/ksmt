package io.ksmt.solver.z3

import io.ksmt.utils.library.NativeLibraryLoaderUtils
import io.ksmt.utils.library.NativeLibraryLoaderWindows
import io.ksmt.utils.library.NativeLibraryLoaderX64

@Suppress("unused")
class KZ3NativeLibraryLoaderWindowsX64 :
    KZ3NativeLibraryLoader,
    NativeLibraryLoaderWindows,
    NativeLibraryLoaderX64 {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("vcruntime140", "vcruntime140_1", "libz3", "libz3java")
    }
}
