//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_PASSES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_PASSES

include "mlir/Pass/PassBase.td"

//
// DeduceDynamicMappedInferenceVersion
//

def DeduceDynamicMappedInferenceVersion : PassBase<"deduce-dynamic-mi", "vpux::ModulePass"> {
    let summary = "Set dynamic version pass";

    let description = [{
        Deduce version of mapped inference (MI version) as maximum version required across all ops it contains.
        If operations has a descriptor, its version is defined as maximum version across all fields set
        in the descriptor, where field is set if it has non-zero value.
    }];

    let constructor = "vpux::VPURegMapped::createDeduceDynamicMappedInferenceVersionPass()";

    let dependentDialects = [
        "vpux::ELF::ELFDialect"
    ];
}

#endif
