//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI40XX
#define VPUX_COMPILER_DIALECT_VPUMI40XX

include "mlir/IR/OpBase.td"

def VPUMI40XX_Dialect : Dialect {
    let summary = "VPU NN Mapped Inference RunTime Dialect for arch VPUX40XX";

    let description = [{
        The **VPUMI40XX Dialect** represents NN RunTime IR together with MappedInference
        constructs in terms of the MLIR framework.

        If we need to determine whether an op should be placed in this dialect, we should
        consider from the perspective of the typical lowering pass and dialect difference
        in the backend.

        1. Typical lowering pass:
            VPUMI40XX   VPURegMapped
                |
             VPUASM                         arch-agnostic
            ----|-----------------------------------------
             NPUReg*                        arch-dependent

            **Note**: The VPURegMapped dialect does not strictly follow the typical lowering pass rules.
            However, it is a backend dialect that includes certain specialized ops.

            There is a clear separation between architecture-agnostic and architecture-dependent dialects:
            - **Arch-agnostic dialects**: VPUMI40XX, VPURegMapped, and VPUASM.
            - **Arch-dependent dialects**: NPUReg*.

            If an op depends on a specific architecture, it should be placed in the NPUReg* dialect
            to maintain a clean separation between these categories.

        2. The VPUMI40XX Dialect:
           This dialect has a SSA-based type system and the ops in this dialect should further be lowered
           to the VPUASM dialect.

        3. The VPUASM Dialect:
           The dialect has the symbolic type system. In addition, it is considered as the last architecture-independent dialect
           and is followed by architecture-dependent dialects, such as NPUReg40XX.

        4. The VPURegMapped Dialect:
           There are ops which do not require symbolic representation or strictly follow the typical lowering pass rules.
           As a result, they should not be placed in the VPUMI40XX or VPUASM dialects. Instead, they can be included in the VPURegMapped dialect
           if they are architecture-agnostic. These operations are partially involved in the lowering pass and serve as utility operations
           to assist with compilation. In addition, these operations are not serialized into the final binary output.

        5. The NPUReg* dialect is represents architecture-specific dialect, Ops that are related to a specific architecture should be
        placed here. The NPUReg* dialect is the final dialect for architecture-dependent ops before serialization. Ops in this
        dialect also use symbolic representation.

        ### Placement Guidelines:
        - If an op follows the typical lowering pass, it should exist in at least the VPUMI40XX
        or VPUASM dialects, as it follows the lowering pass rules.
        - If an op requires serialization into the ELF format, its placement in the NPUReg* dialect
        depends on its architecture dependency:
            - If architecture-dependent, it should be duplicated in the NPUReg* dialect.
            - Otherwise, it can remain in the VPUASM dialect.
        - If an op serves only as a utility for compilation and is not involved in the lowering
        pass logic, it should be placed in the VPURegMapped dialect. For example, the **TaskBufferLayoutOp**
        operation is placed in the VPURegMapped dialect because it is only used to create task location buffers in the
        VPUMI40XX dialect's `CreateResolveTaskLocationPass`. Since it is an architecture-agnostic operation and only
        partially involved in the lowering pass, so it is appropriate to include it in the VPURegMapped dialect.
    }];

    let name = "VPUMI40XX";

    let cppNamespace = "vpux::VPUMI40XX";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "mlir::func::FuncDialect",
        "mlir::quant::QuantDialect",
        "vpux::VPURegMapped::VPURegMappedDialect",
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

#endif
