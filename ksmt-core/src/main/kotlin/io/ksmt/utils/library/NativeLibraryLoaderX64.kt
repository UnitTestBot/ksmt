package io.ksmt.utils.library

interface NativeLibraryLoaderX64 : NativeLibraryLoader {
    override val archName: String get() = NativeLibraryLoaderUtils.X64_ARCH_NAME
}
