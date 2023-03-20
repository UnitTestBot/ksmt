package org.ksmt.utils

import java.io.InputStream
import java.net.URL
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.outputStream

object NativeLibraryLoader {
    enum class OS {
        LINUX, WINDOWS, MACOS
    }

    private val supportedArchs = setOf("amd64", "x86_64", "aarch64")

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

        val destinationFolder = if (arch == "aarch64") "arm" else "x64"

        withLibraryUnpacker {
            for (libName in librariesToLoad) {
                val osLibName = libName + libraryExt
                val resourceName = "lib/$destinationFolder/$osLibName"
                val libraryResource = findResource(resourceName)
                    ?: error("Can't find native library $osLibName")

                val libUri = libraryResource.toURI()
                if (libUri.scheme == "file") {
                    System.load(libUri.path)
                    continue
                }

                val libFile = libraryResource.openStream().use { libResourceStream ->
                    unpackLibrary(
                        name = libName + libraryExt,
                        libraryData = libResourceStream
                    )
                }

                System.load(libFile.toAbsolutePath().toString())
            }
        }
    }

    private fun findResource(resourceName: String): URL? =
        NativeLibraryLoader::class.java.classLoader
            .getResource(resourceName)

    private inline fun withLibraryUnpacker(body: TmpDirLibraryUnpacker.() -> Unit) {
        val unpacker = TmpDirLibraryUnpacker()
        try {
            body(unpacker)
        } finally {
            unpacker.cleanup()
        }
    }

    private class TmpDirLibraryUnpacker {
        private val libUnpackDirectory = Files.createTempDirectory("ksmt")
        private val unpackedFiles = mutableListOf<Path>()

        fun unpackLibrary(name: String, libraryData: InputStream): Path {
            val libFile = libUnpackDirectory.resolve(name).also { unpackedFiles.add(it) }
            libFile.outputStream().use { libraryData.copyTo(it) }
            return libFile
        }

        fun cleanup() {
            val notDeletedFiles = unpackedFiles.filterNot { it.toFile().delete() }

            if (notDeletedFiles.isEmpty() && libUnpackDirectory.toFile().delete()) {
                return
            }

            // Something was not deleted --> register for deletion in reverse order (according to the API)
            libUnpackDirectory.toFile().deleteOnExit()
            notDeletedFiles.forEach { it.toFile().deleteOnExit() }
        }
    }
}
