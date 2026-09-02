//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/utils/mpe_engine_info_utils.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
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

namespace vpux::IE::arch50xx {

void registerMPEEngineInfoOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::ConvolutionOp>>(*ctx);
        IE::MatMulOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::MatMulOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::AddOp>>(*ctx);
        IE::MultiplyOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::MultiplyOp>>(*ctx);
        IE::SubtractOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::SubtractOp>>(*ctx);
        IE::ReduceMeanOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::ReduceMeanOp>>(*ctx);
        IE::ReduceSumOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::ReduceSumOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::GroupConvolutionOp>>(*ctx);
        IE::PermuteQuantizeOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::PermuteQuantizeOp>>(*ctx);
        IE::RollOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::RollOp>>(*ctx);
        IE::InterpolateOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::InterpolateOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<IE::TransposedConvolutionOp>>(
                *ctx);
    });
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::GroupConvolutionOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<VPU::GroupConvolutionOp>>(*ctx);
        VPU::PadOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<VPU::PadOp>>(*ctx);
        VPU::RollOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<VPU::RollOp>>(*ctx);
        VPU::InterpolateOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<VPU::InterpolateOp>>(*ctx);
        VPU::TransposedConvolutionOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<VPU::TransposedConvolutionOp>>(
                *ctx);
        VPU::NCEAveragePoolOp::attachInterface<IE::arch37xx::MPEEngineInfoOpModel<VPU::NCEAveragePoolOp>>(*ctx);
    });
}
}  // namespace vpux::IE::arch50xx
