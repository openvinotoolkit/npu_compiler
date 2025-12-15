//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP
#define VPUX_COMPILER_DIALECT_VPUIP

include "mlir/IR/OpBase.td"

def VPUIP_Dialect : Dialect {
    let summary = "VPU NN Execution RunTime Dialect";

    let description = [{
The **VPUIP Dialect** represents NN RunTime IR in terms of MLIR framework.

It allows to work with the graph schema inside MLIR framework:

* Validate it.
* Perform additional low level transformations/optimizations.

It handles such VPU-specifics as:

* Memory/executors hierarchy.
* Supported operation set.

[./VPUIP/_ops_interfaces.md]
    }];

    let name = "VPUIP";

    let cppNamespace = "vpux::VPUIP";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::Core::CoreDialect",
        "vpux::net::NetDialect",
        "vpux::IE::IEDialect",
        "vpux::VPU::VPUDialect",
        "mlir::func::FuncDialect",
        "mlir::quant::QuantDialect",
        "mlir::async::AsyncDialect",
        // Note: required because VPUIP actually relies on one-shot "buffers"
        "mlir::bufferization::BufferizationDialect",
        "mlir::memref::MemRefDialect"
    ];

    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let usePropertiesForAttributes = 1;

    let extraClassDeclaration = [{
        void registerAttributes();
        void registerTypes();

        static void setExecutor(mlir::async::ExecuteOp execOp, vpux::IndexedSymbolAttr executor);
        static void setExecutorInstanceMask(mlir::async::ExecuteOp execOp, mlir::ArrayAttr executorInstanceMask);
        static llvm::StringLiteral getExecutorAttrName();
        static vpux::IndexedSymbolAttr getExecutor(mlir::async::ExecuteOp execOp);
        static VPU::ExecutorKind getExecutorKind(mlir::async::ExecuteOp execOp);
        static bool hasExecutorInstanceMask(mlir::async::ExecuteOp execOp);
        static bool isComputeExecutorKind(VPU::ExecutorKind executorKind);
        static mlir::ArrayAttr getExecutorInstanceMask(mlir::async::ExecuteOp execOp);

        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
        static void setupExtraInterfacesAdditional(mlir::DialectRegistry& registry);
    }];
}

#endif
