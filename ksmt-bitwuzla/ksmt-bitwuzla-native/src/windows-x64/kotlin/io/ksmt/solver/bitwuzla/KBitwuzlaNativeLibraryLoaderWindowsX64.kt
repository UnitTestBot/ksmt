package io.ksmt.solver.bitwuzla

import io.ksmt.utils.library.NativeLibraryLoaderUtils
import io.ksmt.utils.library.NativeLibraryLoaderWindows
import io.ksmt.utils.library.NativeLibraryLoaderX64

@Suppress("unused")
class KBitwuzlaNativeLibraryLoaderWindowsX64 :
    KBitwuzlaNativeLibraryLoader,
    NativeLibraryLoaderWindows,
    NativeLibraryLoaderX64 {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libbitwuzla", "libbitwuzla_jni")
    }
}
