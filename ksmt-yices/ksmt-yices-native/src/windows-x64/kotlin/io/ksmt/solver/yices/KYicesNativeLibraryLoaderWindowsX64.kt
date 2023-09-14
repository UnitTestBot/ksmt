package io.ksmt.solver.yices

import io.ksmt.utils.library.NativeLibraryLoaderUtils
import io.ksmt.utils.library.NativeLibraryLoaderWindows
import io.ksmt.utils.library.NativeLibraryLoaderX64

@Suppress("unused")
class KYicesNativeLibraryLoaderWindowsX64 :
    KYicesNativeLibraryLoader,
    NativeLibraryLoaderWindows,
    NativeLibraryLoaderX64 {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libyices", "libyices2java")
    }
}
