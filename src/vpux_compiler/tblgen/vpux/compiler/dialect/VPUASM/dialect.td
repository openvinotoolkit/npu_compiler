//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUASM
#define VPUX_COMPILER_DIALECT_VPUASM

include "mlir/IR/OpBase.td"

def VPUASM_Dialect : Dialect {
    let summary = "VPU NN Assembly dialect";

    let description = [{
        The **VPUASM Dialect** represents the inference with the view and semantics of the NN RunTime.
        Unlike **VPUMI Dialect**, **VPUASM** imposes a **symbolic** representation, instead of a
        **SSA use-def chain** one.

        In addition, it is considered as the last architecture-independent dialect and is followed by
        architecture-dependent dialects, such as NPUReg40XX.
    }];

    let name = "VPUASM";

    let cppNamespace = "vpux::VPUASM";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::ELF::ELFDialect",
        "vpux::VPUIP::VPUIPDialect",
        "vpux::VPUMI40XX::VPUMI40XXDialect",
        "mlir::func::FuncDialect",
        "mlir::quant::QuantDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];

    let useDefaultTypePrinterParser = 1;
}

#endif
