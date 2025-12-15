//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU
#define VPUX_COMPILER_DIALECT_VPU

include "mlir/IR/OpBase.td"

def VPU_Dialect : Dialect {
    let summary = "VPU high-level ISA Dialect";

    let description = [{
        The **VPU Dialect** represents high level ISA for VPU architectures.

        It is type compatible with **IE Dialect**, implements its interfaces and can be used along side with it.
        The operations in the **VPU Dialect** are pure functional and works on tensor level.
    }];

    let name = "VPU";

    let cppNamespace = "vpux::VPU";

    let extraClassDeclaration = [{
        void registerAttributes();
        void registerTypes();
    }];

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::Core::CoreDialect",
        "vpux::net::NetDialect",
        "vpux::IE::IEDialect",
        "mlir::func::FuncDialect",
        "mlir::quant::QuantDialect",
        "mlir::scf::SCFDialect"
    ];

    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let usePropertiesForAttributes = 1;
}

#endif
