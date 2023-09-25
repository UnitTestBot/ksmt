package io.ksmt.utils.library

interface NativeLibraryLoaderWindows : NativeLibraryLoader {
    override val osName: String get() = NativeLibraryLoaderUtils.WINDOWS_OS_NAME
    override val osLibraryExt: String get() = ".dll"
}
