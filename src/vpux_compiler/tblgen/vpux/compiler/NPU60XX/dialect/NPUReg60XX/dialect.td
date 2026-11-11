//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUReg60XX
#define VPUX_COMPILER_DIALECT_NPUReg60XX

include "mlir/IR/OpBase.td"

def NPUReg60XX_Dialect : Dialect {
    let summary = "VPU 60XX HW specifics";

    let description = [{
        The **60XX** encapsulates NPU6/60XX specific information.

        The NPUReg60XX dialect is the final dialect for NPUReg60XX ops
        before serialization. Ops in this dialect also use symbolic representation.

    }];

    let name = "NPUReg60XX";

    let cppNamespace = "vpux::NPUReg60XX";

    let dependentDialects = [
        "vpux::VPURegMapped::VPURegMappedDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

#endif
