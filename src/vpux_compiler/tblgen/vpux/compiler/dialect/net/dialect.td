//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_NET
#define VPUX_COMPILER_DIALECT_NET

include "mlir/IR/OpBase.td"

def Net_Dialect : Dialect {
    let summary = "Net Dialect";

    let description = [{
        The `net` dialect contains items related to the neural network itself.
        For example, operations which describe the inputs and outputs of the network
        (e.g. their shape, precision and layout).

        The neural network may contain more information which is relevant to the
        compilation and which could be stored into the IR. For example, information
        from the `rt_info` section of an OpenVINO model (such as per-layer sparsity
        statistics). Such data could be represented via items from this dialect.
    }];

    let name = "net";

    let cppNamespace = "vpux::net";

    let dependentDialects = [
        "vpux::Core::CoreDialect",
        // Note: required because relies on one-shot "tensors" and "buffers"
        "mlir::bufferization::BufferizationDialect"
    ];

    let extraClassDeclaration = [{
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];
}

#endif
