//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONFIG
#define VPUX_COMPILER_DIALECT_CONFIG

include "mlir/IR/OpBase.td"

def Config_Dialect : Dialect {
    let summary = "Config Dialect";

    let description = [{
        The `config` dialect contains items related to the configuration of the compilation.
        For example, operations which describe pipeline options or operations which describe
        the amount of NPU resources that are allocated for the current compilation.

    The **Config Dialect** provides separate operations to describe the available and used run-time resources.
    It deals with the following resource types:

    * Memory space.
    * Executor (CPU, HW module, DMA).

    ```MLIR
    config.Resources 4 of @NCE {
        config.ExecutorResource 5 of @DPU
        config.ExecutorResource 5 of @SHAVE_NN
        config.ExecutorResource 1 of @SHAVE_ACT
        config.MemoryResource 1048576 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.ExecutorResource 1 of @DMA_NN

    config.MemoryResource 31457280 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
    ```

    The `config.ExecutorResource` and `config.MemoryResource` are added by underlying low-level dialect to provide information about HW-specific resources.
    }];

    let name = "config";

    let cppNamespace = "vpux::config";

    let extraClassDeclaration = [{
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

#endif
