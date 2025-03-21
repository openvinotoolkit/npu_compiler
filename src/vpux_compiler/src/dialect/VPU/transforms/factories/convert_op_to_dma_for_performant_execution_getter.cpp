//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/convert_op_to_dma_for_performant_execution_getter.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/convert_ops_to_dma_for_performant_execution_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace vpux::VPU;

std::unique_ptr<vpux::IConversionPassStrategy> vpux::VPU::createConvertOpToDMAForPerformantExecutionStrategy(
        ArchKind arch) {
    if (arch >= VPU::ArchKind::NPU40XX) {
        return std::make_unique<arch40xx::ConvertOpToDMAForPerformantExecutionStrategy>();
    }
    // TODO : E#-118296 Other ops and architectures will be enabled.
    VPUX_THROW("Currently ConvertOpToDMAForPerformantExecutionStrategy is available for NPU40XX arch");
}
