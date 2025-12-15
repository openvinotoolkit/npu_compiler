//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NPUREG50XX
#define VPUX_COMPILER_DIALECT_NPUREG50XX

include "mlir/IR/OpBase.td"

def NPUReg50XX_Dialect : Dialect {
    let summary = "VPU 50XX HW specifics";

    let description = [{
        The **50XX** encapsulates VPU5/50XX specific information.

        The NPUReg50XX dialect is the final dialect for NPUReg50XX ops
        before serialization. Ops in this dialect also use symbolic representation.
    }];

    let name = "NPUReg50XX";

    let cppNamespace = "vpux::NPUReg50XX";

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
