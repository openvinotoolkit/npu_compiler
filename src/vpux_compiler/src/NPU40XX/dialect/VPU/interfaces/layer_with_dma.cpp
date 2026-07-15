//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_utils/layer_with_dma.hpp"

using namespace vpux;

void vpux::VPU::arch40xx::registerLayerWithDmaInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::AtanOp::attachInterface<VPU::LayerWithDmaInterfaceActivation<IE::AtanOp>>(*ctx);
        IE::InterpolateOp::attachInterface<VPU::LayerWithDmaInterfaceFwlm<IE::InterpolateOp>>(*ctx);
        IE::ScatterUpdateOp::attachInterface<VPU::LayerWithDmaInterfaceScatterUpdate>(*ctx);
    });
}
