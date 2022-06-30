package org.ksmt.solver.bitwuzla.bindings

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Platform
import com.sun.jna.Pointer
import java.nio.file.Path
import kotlin.io.path.absolutePathString

object FilePtrUtils {
    fun openRead(path: Path): FilePtr = FilePtr(lib.fopen(path.absolutePathString(), "r"))
    fun openWrite(path: Path): FilePtr = FilePtr(lib.fopen(path.absolutePathString(), "w+"))
    fun stdout(): FilePtr = stdout
    fun flush(ptr: FilePtr) = lib.fflush(ptr.ptr)
    fun close(ptr: FilePtr) = lib.fclose(ptr.ptr)


    private interface CLib : Library {
        fun fopen(path: String, mode: String): Pointer
        fun fdopen(fd: Int, mode: String): Pointer
        fun fclose(ptr: Pointer)
        fun fflush(ptr: Pointer): Int
    }

    private val lib: CLib by lazy {
        Native.load(if (Platform.isWindows()) "msvcrt" else "c", CLib::class.java)
    }

    private val stdout: FilePtr by lazy {
        FilePtr(lib.fdopen(1, "w"), closable = false)
    }

}

class FilePtr internal constructor(val ptr: Pointer, private val closable: Boolean = true) : AutoCloseable {
    override fun close() {
        FilePtrUtils.flush(this)
        if (closable) {
            FilePtrUtils.close(this)
        }
    }
}
