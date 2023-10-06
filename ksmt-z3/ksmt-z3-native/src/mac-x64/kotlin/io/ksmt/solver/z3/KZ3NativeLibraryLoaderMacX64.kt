package io.ksmt.solver.z3

import io.ksmt.utils.library.NativeLibraryLoaderMac
import io.ksmt.utils.library.NativeLibraryLoaderUtils
import io.ksmt.utils.library.NativeLibraryLoaderX64

@Suppress("unused")
class KZ3NativeLibraryLoaderMacX64 :
    KZ3NativeLibraryLoader,
    NativeLibraryLoaderMac,
    NativeLibraryLoaderX64 {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libz3", "libz3java")
    }
}
