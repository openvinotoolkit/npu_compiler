//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/mpe_engine_info_utils.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

namespace vpux::IE::arch37xx {

void registerMPEEngineInfoOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<MPEEngineInfoOpModel<IE::ConvolutionOp>>(*ctx);
        IE::MatMulOp::attachInterface<MPEEngineInfoOpModel<IE::MatMulOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<MPEEngineInfoOpModel<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<MPEEngineInfoOpModel<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<MPEEngineInfoOpModel<IE::AddOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<MPEEngineInfoOpModel<IE::GroupConvolutionOp>>(*ctx);
        IE::PermuteQuantizeOp::attachInterface<MPEEngineInfoOpModel<IE::PermuteQuantizeOp>>(*ctx);
        IE::InterpolateOp::attachInterface<MPEEngineInfoOpModel<IE::InterpolateOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<MPEEngineInfoOpModel<IE::TransposedConvolutionOp>>(*ctx);
        IE::RollOp::attachInterface<MPEEngineInfoOpModel<IE::RollOp>>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::GroupConvolutionOp::attachInterface<MPEEngineInfoOpModel<VPU::GroupConvolutionOp>>(*ctx);
        VPU::PadOp::attachInterface<MPEEngineInfoOpModel<VPU::PadOp>>(*ctx);
        VPU::RollOp::attachInterface<MPEEngineInfoOpModel<VPU::RollOp>>(*ctx);
        VPU::InterpolateOp::attachInterface<MPEEngineInfoOpModel<VPU::InterpolateOp>>(*ctx);
        VPU::TransposedConvolutionOp::attachInterface<MPEEngineInfoOpModel<VPU::TransposedConvolutionOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<MPEEngineInfoOpModel<VPU::NCEAveragePoolOp>>(*ctx);
    });
}

}  // namespace vpux::IE::arch37xx
