//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>

using namespace vpux;

//
// registerBufferizableOpInterfaces
//

void vpux::arch37xx::registerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    vpux::registerTensorSliceBufferizableOpInterfaces(registry);
    vpux::registerConstDeclareBufferizableOpInterfaces(registry);
    vpux::registerCoreBufferizableOpInterfaces(registry);
    vpux::registerSoftwareLayerBufferizableOpInterfaces(registry);
    vpux::registerVpuNceBufferizableOpInterfaces(registry);
    vpux::registerVPUBufferizableOpInterfaces(registry);

    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
}
