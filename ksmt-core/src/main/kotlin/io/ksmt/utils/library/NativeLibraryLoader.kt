package io.ksmt.utils.library

interface NativeLibraryLoader {
    val libraryDir: String

    val osName: String

    val libraryExt: String

    val archName: String

    fun load()
}
