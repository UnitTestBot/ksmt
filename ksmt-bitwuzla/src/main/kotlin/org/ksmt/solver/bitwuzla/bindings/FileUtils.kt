package org.ksmt.solver.bitwuzla.bindings

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Platform
import com.sun.jna.Pointer
import java.nio.file.Path
import kotlin.io.path.absolutePathString

object FileUtils {
    fun openRead(path: Path): FilePtr = FilePtr(lib.fopen(path.absolutePathString(), "r"))
    fun openWrite(path: Path): FilePtr = FilePtr(lib.fopen(path.absolutePathString(), "w+"))
    fun stdout(): FilePtr = FilePtr(lib.fdopen(1, "w"))
    fun close(ptr: Pointer) = lib.fclose(ptr)

    interface CLib : Library {
        fun fopen(path: String, mode: String): Pointer
        fun fdopen(fd: Int, mode: String): Pointer
        fun fclose(ptr: Pointer)
    }

    private val lib: CLib by lazy {
        Native.load(if (Platform.isWindows()) "msvcrt" else "c", CLib::class.java)
    }

}

class FilePtr(val ptr: Pointer) : AutoCloseable {
    override fun close() {
        FileUtils.close(ptr)
    }
}

