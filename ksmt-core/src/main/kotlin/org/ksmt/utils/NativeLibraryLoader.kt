package org.ksmt.utils

import java.nio.file.Files
import kotlin.io.path.deleteIfExists
import kotlin.io.path.outputStream

object NativeLibraryLoader {
    enum class OS {
        LINUX, WINDOWS, MACOS
    }

    private val supportedArchs = setOf("amd64", "x86_64")

    fun load(libraries: (OS) -> List<String>) {
        val arch = System.getProperty("os.arch")

        require(arch in supportedArchs) { "Not supported arch: $arch" }

        val osName = System.getProperty("os.name").lowercase()
        val (os, libraryExt) = when {
            osName.startsWith("linux") -> OS.LINUX to ".so"
            osName.startsWith("windows") -> OS.WINDOWS to ".dll"
            osName.startsWith("mac") -> OS.MACOS to ".dylib"
            else -> error("Unknown OS: $osName")
        }

        val librariesToLoad = libraries(os)

        for (libName in librariesToLoad) {
            val osLibName = libName + libraryExt
            val resourceName = "lib/x64/$osLibName"
            val libUri = NativeLibraryLoader::class.java.classLoader
                .getResource(resourceName)
                ?.toURI()
                ?: error("Can't find native library $osLibName")

            if (libUri.scheme == "file") {
                System.load(libUri.path)
                continue
            }

            val libFile = Files.createTempFile(libName, libraryExt)

            NativeLibraryLoader::class.java.classLoader
                .getResourceAsStream(resourceName)
                ?.use { libResourceStream ->
                    libFile.outputStream().use { libResourceStream.copyTo(it) }
                }

            System.load(libFile.toAbsolutePath().toString())

            libFile.deleteIfExists()
        }
    }
}
