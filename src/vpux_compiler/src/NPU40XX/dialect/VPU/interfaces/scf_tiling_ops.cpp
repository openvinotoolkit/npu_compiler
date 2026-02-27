//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_viewlike_interfaces.hpp"

void vpux::VPU::arch40xx::registerSCFTilingOpsInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEEltwiseOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<vpux::VPU::SCFAvgPoolOpModel>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<vpux::VPU::SCFMaxPoolOpModel>(*ctx);
        VPU::NCEConvolutionOp::attachInterface<vpux::VPU::SCFConvOpModel>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<vpux::VPU::SCFTilingDepthConvModelOp>(*ctx);
        VPU::NCEPermuteOp::attachInterface<vpux::VPU::SCFTilingPermuteModelOp>(*ctx);

        VPU::DepthToSpaceOp::attachInterface<vpux::VPU::SCFDepthToSpaceModelOp>(*ctx);
        VPU::ConvertOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::ConvertOp>>(*ctx);

        VPU::LayoutCastOp::attachInterface<vpux::VPU::SCFGenericViewLikeTilingModelOp<VPU::LayoutCastOp>>(*ctx);
        VPU::PermuteCastOp::attachInterface<vpux::VPU::SCFPermuteCastTilingModelOp>(*ctx);
        VPU::QuantizeCastOp::attachInterface<vpux::VPU::SCFGenericViewLikeTilingModelOp<VPU::QuantizeCastOp>>(*ctx);
    });
}
