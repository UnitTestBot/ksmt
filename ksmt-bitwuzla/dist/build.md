### Build details

We build Bitwuzla with the following configuration:
```shell
./configure production --shared --only-cadical
```
To make our distribution portable we statically link Bitwuzla against `libgmp`.

Jni bindings `bitwuzla_jni` are built without any tricks.

### Linux x64 rebuild with the beta-reduction fix

`bitwuzla-linux64-1.0.zip` carries a `libbitwuzla.so` rebuilt from Bitwuzla commit `1230d80a5275ddf525da9c218c2bcd29c3130e49` with `bitwuzla-gh31.patch` applied.
The patch stops `beta_reduce` and `beta_reduce_partial_aux` in `src/bzlabeta.c` from replacing a lambda by its body when the lambda's parameter was not instantiated.
Without it an array `update` over such a lambda took a bitvector sort, and the next `bzla_node_create_apply` crashed in `bzla_sort_fun_get_codomain`.
The crash is reached from the fun solver's lemma loop, so no option avoids it.

`Dockerfile.bitwuzla-linux64` reproduces the build: GMP 6.3.0 static with `--with-pic --enable-fat`, the pinned Bitwuzla tarball, the patch, the `contrib/setup-*.sh` dependencies, then `./configure.sh production --shared --only-cadical`.
It checks that the result needs only `libc.so.6` and `libm.so.6` and defines every `bzla_*` and `bitwuzla_*` symbol the unchanged `libbitwuzla_jni.so` imports, then repacks the zip.

```shell
BASE_REGISTRY=mirror.gcr.io ./build-linux64.sh
```

`libbitwuzla_jni.so` in the zip is still the original binary.
The rebuilt `libbitwuzla.so` has sha256 `6b2ee8e5efed1af68f86474dfeff7c8f9685f3999472073ce201529fb4486252`.
It also lists `ld-linux-x86-64.so.2` as a dependency and needs glibc 2.33 or newer, because the build image is Ubuntu 22.04; the 2023 binary needed glibc 2.14.

### Building on windows
To produce Windows builds we use MSYS2 environment prepared as described [here](https://github.com/aytey/bitwuzla/blob/refreshed_windows_instructions/docs/building_on_windows.rst).
Also, it is important to link statically with `libwinpthread`.

### Expected dynamic dependencies
To ensure that our distribution is portable, we verify that the produced binaries have no dependencies that might not be present on the user's machine. 

#### Linux x64
```shell
$ ldd libbitwuzla.so 
        linux-vdso.so.1 (0x00007ffce92d0000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f2fdad18000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f2fda819000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f2fdb1e0000)
$ ldd libbitwuzla_jni.so 
        libbitwuzla.so => not found
        linux-vdso.so.1 (0x00007ffd1ddc6000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f6188b18000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f6188619000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f618906d000)
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
