//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/scf/scf_tiling_viewlike_interfaces.hpp"

void vpux::VPU::arch40xx::registerSCFTilingOpsInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEEltwiseOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::NCEEltwiseOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<vpux::VPU::SCFAvgPoolOpModel>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<vpux::VPU::SCFMaxPoolOpModel>(*ctx);
        VPU::NCEConvolutionOp::attachInterface<vpux::VPU::SCFConvOpModel>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<vpux::VPU::SCFCompressConvOpModel>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<vpux::VPU::SCFTilingDepthConvModelOp>(*ctx);
        VPU::NCEPermuteOp::attachInterface<vpux::VPU::SCFTilingPermuteModelOp>(*ctx);
        VPU::NCEReduceOp::attachInterface<vpux::VPU::SCFNCEReduceModelOp>(*ctx);

        VPU::DepthToSpaceOp::attachInterface<vpux::VPU::SCFDepthToSpaceModelOp>(*ctx);
        VPU::InterpolateOp::attachInterface<vpux::VPU::SCFInterpolateModelOp>(*ctx);
        VPU::ConvertOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::ConvertOp>>(*ctx);
        VPU::YuvToRgbOp::attachInterface<vpux::VPU::SCFYuvToRgbModelOp>(*ctx);
        VPU::GeluOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::GeluOp>>(*ctx);
        VPU::DequantizeOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::DequantizeOp>>(*ctx);
        VPU::SoftMaxOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SoftMaxOp>>(*ctx);
        VPU::AbsOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::AbsOp>>(*ctx);
        VPU::AcosOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::AcosOp>>(*ctx);
        VPU::AcoshOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::AcoshOp>>(*ctx);
        VPU::AsinOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::AsinOp>>(*ctx);
        VPU::AsinhOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::AsinhOp>>(*ctx);
        VPU::AtanOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::AtanOp>>(*ctx);
        VPU::AtanhOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::AtanhOp>>(*ctx);
        VPU::CeilingOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::CeilingOp>>(*ctx);
        VPU::ClampOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::ClampOp>>(*ctx);
        VPU::CosOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::CosOp>>(*ctx);
        VPU::CoshOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::CoshOp>>(*ctx);
        VPU::CumSumOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::CumSumOp>>(*ctx);
        VPU::ErfOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::ErfOp>>(*ctx);
        VPU::ExpOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::ExpOp>>(*ctx);
        VPU::FloorOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::FloorOp>>(*ctx);
        VPU::LogOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::LogOp>>(*ctx);
        VPU::NegativeOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::NegativeOp>>(*ctx);
        VPU::RoundOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::RoundOp>>(*ctx);
        VPU::SignOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SignOp>>(*ctx);
        VPU::SinhOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SinhOp>>(*ctx);
        VPU::TanOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::TanOp>>(*ctx);
        VPU::SinOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SinOp>>(*ctx);
        VPU::SqrtOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SqrtOp>>(*ctx);
        VPU::TanhOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::TanhOp>>(*ctx);
        VPU::MishOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::MishOp>>(*ctx);
        VPU::ReLUOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::ReLUOp>>(*ctx);
        VPU::SigmoidOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SigmoidOp>>(*ctx);
        VPU::SoftPlusOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SoftPlusOp>>(*ctx);
        VPU::SwishOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SwishOp>>(*ctx);
        VPU::EluOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::EluOp>>(*ctx);
        VPU::HardSigmoidOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::HardSigmoidOp>>(*ctx);
        VPU::HSigmoidOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::HSigmoidOp>>(*ctx);
        VPU::HSwishOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::HSwishOp>>(*ctx);
        VPU::LeakyReluOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::LeakyReluOp>>(*ctx);
        VPU::PReluOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::PReluOp>>(*ctx);
        VPU::SeluOp::attachInterface<vpux::VPU::SCFTilingEltwiseLikeModelOp<VPU::SeluOp>>(*ctx);

        VPU::LayoutCastOp::attachInterface<vpux::VPU::SCFGenericViewLikeTilingModelOp<VPU::LayoutCastOp>>(*ctx);
        VPU::PermuteCastOp::attachInterface<vpux::VPU::SCFPermuteCastTilingModelOp>(*ctx);
        VPU::SliceOp::attachInterface<vpux::VPU::SCFSliceTilingModelOp>(*ctx);
        VPU::ExpandOp::attachInterface<vpux::VPU::SCFExpandTilingModelOp>(*ctx);
        VPU::QuantizeCastOp::attachInterface<vpux::VPU::SCFGenericViewLikeTilingModelOp<VPU::QuantizeCastOp>>(*ctx);

        VPU::ReduceLogicalOrOp::attachInterface<vpux::VPU::SCFReduceLogicalOrModelOp>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<vpux::VPU::SCFReduceLogicalAndModelOp>(*ctx);
        VPU::ReduceMeanOp::attachInterface<vpux::VPU::SCFReduceMeanModelOp>(*ctx);
        VPU::ReduceSumOp::attachInterface<vpux::VPU::SCFReduceSumModelOp>(*ctx);
        VPU::ReduceL2Op::attachInterface<vpux::VPU::SCFReduceL2ModelOp>(*ctx);
        VPU::ReduceL1Op::attachInterface<vpux::VPU::SCFReduceL1ModelOp>(*ctx);
        VPU::ReduceSquareOp::attachInterface<vpux::VPU::SCFReduceSquareModelOp>(*ctx);
        VPU::ReduceMinOp::attachInterface<vpux::VPU::SCFReduceMinModelOp>(*ctx);
        VPU::ReduceMaxOp::attachInterface<vpux::VPU::SCFReduceMaxModelOp>(*ctx);
        VPU::ReduceProdOp::attachInterface<vpux::VPU::SCFReduceProdModelOp>(*ctx);
        VPU::LSTMGatesOp::attachInterface<vpux::VPU::SCFLSTMGatesModelOp>(*ctx);
        VPU::TopKOp::attachInterface<vpux::VPU::SCFTopKModelOp>(*ctx);
        VPU::MVN1NormalizeOp::attachInterface<vpux::VPU::SCFMVN1NormalizeModelOp>(*ctx);
    });
}
