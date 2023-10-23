### Build details

We build Bitwuzla with the following configuration:
```shell
./configure.py release --shared
```

and for Windows:
```shell
./configure.py release --shared --win64
```

To make our distribution portable we statically link Bitwuzla against:
* `gmp` - in [src/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/meson.build)
* `bitwuzlabv` (`bitvector_lib`) - in [src/lib/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/lib/meson.build)
* `bitwuzlabb` (`bitblast_lib`) - in [src/lib/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/lib/meson.build)
* `bitwuzlals` (`local_search_lib`) - in [src/lib/meson.build](https://github.com/dee-tree/bitwuzla/blob/main/src/lib/meson.build)

Jni bindings `bitwuzla_jni` are built without any tricks.

### Building on windows
To produce Windows builds we use MSYS2 (MinGW) environment prepared as described [here](https://github.com/aytey/bitwuzla/blob/refreshed_windows_instructions/docs/building_on_windows.rst).
Also, it is important to link statically with `libwinpthread`, `libgcc`, `libstdc++`.

### Expected dynamic dependencies
To ensure that our distribution is portable, we verify that the produced binaries have no dependencies that might not be present on the user's machine. 

#### Linux x64
```shell
$ ldd libbitwuzla.so 
        linux-vdso.so.1 (0x00007fff453c9000)
        libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f8ca6400000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f8ca6713000)
        libgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x00007f8ca6c96000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f8ca621e000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f8ca6ce9000)
$ ldd libbitwuzla_jni.so 
        linux-vdso.so.1 (0x00007ffc53530000)
        libbitwuzla.so => #PathToCompiledBitwuzlaLibrary
        libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007fbeaec00000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007fbeaef13000)
        libgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x00007fbeaf486000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007fbeaea1e000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007fbeaf4f2000)
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
