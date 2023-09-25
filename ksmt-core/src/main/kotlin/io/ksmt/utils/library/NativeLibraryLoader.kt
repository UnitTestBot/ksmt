package io.ksmt.utils.library

/**
 * Provides basic native library loading functionality.
 * See [NativeLibraryLoaderUtils].
 *
 * All properties are used to construct a native library resource file name.
 * See [NativeLibraryLoaderUtils.loadLibrariesFromResources].
 * */
interface NativeLibraryLoader {
    val libraryDir: String

    val osName: String

    val osLibraryExt: String

    val archName: String

    /**
     * Load native libraries.
     * See [NativeLibraryLoaderUtils.load] for the details.
     * */
    fun load()
}
