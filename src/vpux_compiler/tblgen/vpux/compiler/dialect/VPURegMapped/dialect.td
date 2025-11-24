//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED

include "mlir/IR/OpBase.td"

def VPURegMapped_Dialect : Dialect {
    let summary = "VPU NN Register Mapped Dialect";

    let description = [{
        The **VPURegMapped Dialect** defines attributes and types common
        for all supported VPU HW architectures. It is used combined with
        arch specific register map dialects (e.g. NPU37XX, VPU40XX)
        to represents NN RunTime IR constructs in terms of the MLIR framework.

        There are ops which do not require symbolic representation or strictly
        follow the typical lowering pass rules. As a result, they should not be
        placed in the VPUMI40XX or VPUASM dialects. Instead, they can be included
        in the VPURegMapped dialect, if they are architecture-agnostic. These operations
        are partially involved in the lowering pass and serve as utility operations to
        assist with compilation. In addition, these operations are not serialized into
        the final binary output.

        For example, the **TaskBufferLayoutOp** operation is placed in the VPURegMapped
        dialect because it is only used to create task location buffers in the
        VPUMI40XX dialect's `CreateResolveTaskLocationPass`. Since it is an
        architecture-agnostic operation and only partially involved in the lowering pass,
        it is appropriate to include it in the VPURegMapped dialect.
        }];



    let name = "VPURegMapped";

    let cppNamespace = "vpux::VPURegMapped";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::ELFNPU37XX::ELFNPU37XXDialect",
        "mlir::func::FuncDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

#endif
