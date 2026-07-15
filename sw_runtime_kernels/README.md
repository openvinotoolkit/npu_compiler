# How to build kernel binaries

To build the software kernels, MoviTools must be available in the development environment, they are downloaded automatically when you enable the `ENABLE_SHAVE_BINARIES_BUILD` cmake option.

The downloaded tools link is declared in `(vpux_plugin_path)/artifacts/vpuip2/revisions.json`, if you want to use other version, you can change the link or use the `IE_NPU_FORCE_MV_TOOLS_PATH` environment variable, which should contain the full path to a particular MoviTools version. Changing the tools version is recommended for debugging purposes only.

To build the management kernels and the kernels using firmware resources please use the following environment variable: `FIRMWARE_VPU_DIR` - absolute path to firmware.vpu.client workspace.

## CMake options used in vpux-plugin build

- `ENABLE_SHAVE_BINARIES_BUILD=ON|OFF`
    - `ON` - build sw kernels
    - `OFF` (default) - prebuilt binaries will be used
- `ENABLE_MANAGEMENT_KERNEL_BUILD=ON|OFF`
    - management kernels are treated separately because they also require dependencies from `vpu.firmware.client` repository
    - `ON` - build management kernels
    - `OFF` (default) - prebuilt binaries will be used
- `ENABLE_FIRMWARE_SOURCES_KERNEL_BUILD=ON|OFF`
    - kernels using firmware resources require dependencies from `vpu.firmware.client` repository
    - `ON` - build kernels using firmware sources
    - `OFF` (default) - prebuilt binaries will be used

## Kernel description files (descrip/*.txt)

Description file is the text file, which will be included into cmake script by include() statement and customize build options for a particular kernel.
Usually but not necessarily description file contains one or more set() or list() cmake statements which assign a values to a dedicated variables.
As for cmake scripts, '#' character marks a comment line.

CAUTION: since description file is included into cmake script, it can't be isolated from cmake script context.
So, you can unintentionally change the behaviour of cmake and even make it wrong. Be careful!

Variables intended to be used/set in description files:

- `kernel_entry`
  a string which specifies kernel entry point name. Optional; default is `"${kernel_src}"` without path and filename suffixes (extensions).
- `kernel_src`
  a string which specifies kernel source file name, without path and relative to `"${kernel_src_dir}"` . Required.
- `kernel_src_dir`
  a string which specifies a path to directory containing `"${kernel_src}"` file.
  Can be absolute, or relative to [sw_runtime_kernels/kernels](.). Optional; default is `"src"` .
- `kernel_cpunum`
  a string which specifies a target chip level. Optional; default is `"3720"`.
  Sources are compiled with `"-mcpu=${kernel_cpunum}xx"` option.
  Built binaries' names have suffix in the form of `".${kernel_cpunum}xx"` .
- `optimization_opts`
  a string which specifies an optimization option for compilation of source files. Optional; default is `"-O3"` .
- `include_dirs_list`
  a string which specifies an additional C/C++ include directories list in the cmake form (`"dir1;dir2;etc"`).
  The list is parsed and each directory is prepended by `"-I"` prefix automatically. Optional.
- `define_symbols_list`
  a string which specifies an additional C/C++ #define symbols list in the cmake form (`"sym1;sym2;etc"`).
  The list is parsed and each symbol is prepended by `"-D"` prefix automatically. Optional.
- `always_inline`
  a string which specifies whether compilation use inlined code (`-DCONFIG_ALWAYS_INLINE`) or not. Optional; default is `"no"` .
  Can also be checked in description code in if() statement(s) to change compile/link behaviour (e.g. add extra source files).
- `extra_src_list`
  a string which specifies an additional C/C++ source files (absolute paths) which will be compiled and linked together with `"${kernel_src}"` file to form the output binary.
  The list must be in the cmake form (`"src1;src2;etc"`), it is parsed automatically.
- `link_script_file`
  a string which specifies custom link 'ldscript' file. Optional; default is `"${CMAKE_SOURCE_DIR}/prebuild/shave_kernel.ld"` .
  For existing kernels only ManagementKernel (nnActEntry) uses different ldscript; for the rest of kernels default ldscript should be enough.
- `kernel_descrip_path`
  a string which specifies an absolute path to description file directory; can be used to include another (e.g. 'common') description file.
  Prepared by cmake script automatically; description file can use it.

Examples:

### dummy.txt

```
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

set(kernel_src "dummy.cpp")

set(kernel_cpunum "3720")

set(always_inline "yes")
```

### singleShaveSoftmax.txt

```
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

set(kernel_src "singleShaveSoftmax.cpp")

set(kernel_cpunum "3720")

set(optimization_opts "") # -O3
set(always_inline "yes")
```
