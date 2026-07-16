# Pass disabling mechanism
During development and debugging, it might be necessary or desirable to disable some compiler passes. For that purpose,
there is the `disabled-passes` option in the compiler.

## The `disabled-passes` option
The `disabled-passes` option takes a regex that matches all passes that should be disabled. A pass is disabled if either
its pass name or its pass argument name match the regex. Therefore, these four settings are equivalent:
```
disabled-passes=(optimize-concat|fuse-convert)
disabled-passes=(optimize-concat|FuseConvertPass)
disabled-passes=(OptimizeConcat|fuse-convert)
disabled-passes=(OptimizeConcat|FuseConvertPass)
```

This option can be set in `NPU_COMPILATION_MODE_PARAMS`:
```
NPU_COMPILATION_MODE_PARAMS disabled-passes=FuseConvertPass
```

If it is not possible to use compilation mode params (e.g. in LIT tests), this option can also be passed in the init
compiler pipeline options. Example from [pass_disabling.mlir](/tests/lit/NPU/utils/pass_disabling.mlir):
```bash
vpux-opt --split-input-file --init-compiler="platform=%platform% disabled-passes=set-memory-space" --set-memory-space="memory-space=DDR" %s
```

## Implementation
This option is only available in developer builds, i.e. if `VPUX_DEVELOPER_BUILD` is defined.

If the `disabled-passes` option is set, the init compiler pipeline sets the [PassDisablingCallback](/src/vpux_compiler/include/vpux/compiler/utils/pass_disabling_callback.hpp)
as the callback for the `NpuActionHandler`. Before each pass is executed, this callback checks whether the pass name
or pass argument name match the `disabled-passes` regex and skips the pass execution if so.

## Related options
The compiler also has several options that enable or disable specific passes or pipelines. These options might be enabled-by-default
on some platforms and disabled on others. An example is the `enable-ops-as-dma` which is disabled on NPU 40XX and 50XX and enabled
on all other platforms.

It is preferable to use the `disabled-passes` option for debugging purposes as it provides a clear list of passes that are being disabled.
