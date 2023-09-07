### Build details

We build Bitwuzla with the following configuration:
```shell
./configure.py release --shared
```
To make our distribution portable we statically link Bitwuzla against:
* `libgmp` - in [src/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/meson.build)
* `bitwuzlabv` (`bitvector_lib`) - in [src/lib/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/lib/meson.build)
* `bitwuzlabb` (`bitblast_lib`) - in [src/lib/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/lib/meson.build)
* `bitwuzlals` (`local_search_lib`) - in [src/lib/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/lib/meson.build)

Jni bindings `bitwuzla_jni` are built without any tricks.

### Building on windows
To produce Windows builds we use MSYS2 environment prepared as described [here](https://github.com/aytey/bitwuzla/blob/refreshed_windows_instructions/docs/building_on_windows.rst).
Also, it is important to link statically with `libwinpthread`.

### Expected dynamic dependencies
To ensure that our distribution is portable, we verify that the produced binaries have no dependencies that might not be present on the user's machine. 

#### Linux x64
```shell
$ ldd libbitwuzla.so 
        linux-vdso.so.1 (0x00007ffcaf18b000)
        libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f1c42c00000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f1c43442000)
        libgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x00007f1c4341d000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f1c42800000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f1c4355b000)
$ ldd libbitwuzla_jni.so 
        linux-vdso.so.1 (0x00007ffcbf36f000)
        libbitwuzla.so => #PathToCompiledBitwuzlaLibrary
        libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f45d3200000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f45d3aa5000)
        libgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x00007f45d3a80000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f45d2e00000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f45d3bd5000)
```

#### Windows x64
```shell
$ ldd libbitwuzla.dll
        ntdll.dll => /c/Windows/SYSTEM32/ntdll.dll (0x7ffb7a190000)
        KERNEL32.DLL => /c/Windows/System32/KERNEL32.DLL (0x7ffb78240000)
        KERNELBASE.dll => /c/Windows/System32/KERNELBASE.dll (0x7ffb77a70000)
        msvcrt.dll => /c/Windows/System32/msvcrt.dll (0x7ffb79410000)
        CRYPTBASE.DLL => /c/Windows/SYSTEM32/CRYPTBASE.DLL (0x7ffb77190000)
        bcryptPrimitives.dll => /c/Windows/System32/bcryptPrimitives.dll (0x7ffb77de0000)
$ ldd libbitwuzla_jni.dll
        ntdll.dll => /c/Windows/SYSTEM32/ntdll.dll (0x7ffb7a190000)
        KERNEL32.DLL => /c/Windows/System32/KERNEL32.DLL (0x7ffb78240000)
        KERNELBASE.dll => /c/Windows/System32/KERNELBASE.dll (0x7ffb77a70000)
        msvcrt.dll => /c/Windows/System32/msvcrt.dll (0x7ffb79410000)
        libbitwuzla.dll => not found
        CRYPTBASE.DLL => /c/Windows/SYSTEM32/CRYPTBASE.DLL (0x7ffb77190000)
        bcryptPrimitives.dll => /c/Windows/System32/bcryptPrimitives.dll (0x7ffb77de0000)
```

#### MacOS aarch64
```shell
$ otool -L libbitwuzla.dylib    
libbitwuzla.dylib:
        @rpath/libbitwuzla.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1300.36.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)
$ otool -L libbitwuzla_jni.dylib 
libbitwuzla_jni.dylib:
        @rpath/libbitwuzla_jni.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libbitwuzla.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1300.36.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)
```
