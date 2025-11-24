//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

namespace {

std::vector<unsigned int> getNTHWNTKGrid(VPU::MPEMode mode) {
    switch (mode) {
    case VPU::MPEMode::CUBOID_4x16:
        return {8, 8, 256};
    case VPU::MPEMode::CUBOID_8x16:
        return {16, 8, 128};
    default:
        return {16, 16, 64};
    }
}
VPU::MPEMode getMpeModeForConv([[maybe_unused]] config::ArchKind arch, ShapeRef shape) {
    std::vector<std::pair<double, VPU::MPEMode>> MPECost = {{0.0, VPU::MPEMode::CUBOID_4x16},
                                                            {0.0, VPU::MPEMode::CUBOID_8x16},
                                                            {0.0, VPU::MPEMode::CUBOID_16x16}};

    for (unsigned int idx = 0; idx < MPECost.size(); idx++) {
        auto grid = getNTHWNTKGrid(MPECost[idx].second);
        // Get the number of weights and activation grid reads
        double numWtGrids = std::ceil((double)shape[Dims4D::Act::C] / (double)grid[2]);
        double numActGrids = std::ceil((double)shape[Dims4D::Act::H] / (double)grid[1]) *
                             std::ceil((double)shape[Dims4D::Act::W] / (double)grid[0]);

        // Compute the simplfied number of reads
        double actReads = numWtGrids * shape[Dims4D::Act::H] * shape[Dims4D::Act::W];
        double wtReads = numActGrids * shape[Dims4D::Act::C];

        MPECost[idx].first = actReads + wtReads;
    }

    // Select the one that has min number of reads
    return (*std::min_element(MPECost.begin(), MPECost.end())).second;
}

class ConvMpeModeModel {
public:
    VPU::MPEMode getMpeModeImpl(mlir::Operation* op, mlir::Type, mlir::Type, ShapeRef shape) const {
        auto archKind = config::getArch(op);
        return getMpeModeForConv(archKind, shape);
    }
};

class Cuboid8MpeModeModel {
public:
    VPU::MPEMode getMpeModeImpl(mlir::Operation*, mlir::Type, mlir::Type, ShapeRef) const {
        return VPU::MPEMode::CUBOID_8x16;
    }
};

class Cuboid16MpeModeModel {
public:
    VPU::MPEMode getMpeModeImpl(mlir::Operation*, mlir::Type, mlir::Type, ShapeRef) const {
        return VPU::MPEMode::CUBOID_16x16;
    }
};

class ConvolutionOpModel :
        public VPU::NCEConvolutionOpModel<ConvolutionOpModel, VPU::NCEConvolutionOp>,
        public ConvMpeModeModel {
public:
    std::optional<vpux::VPU::MPEEngineAttr> getMpeEngine(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEConvolutionOp>(op).getMpeEngine();
    }
};
class ReduceOpModel : public VPU::NCEReduceOpModel<ReduceOpModel, VPU::NCEReduceOp>, public ConvMpeModeModel {};
class DepthConvolutionOpModel :
        public VPU::NCEConvolutionOpModel<DepthConvolutionOpModel, VPU::NCEDepthConvolutionOp>,
        public Cuboid16MpeModeModel {};
class CompressConvolutionOpModel :
        public VPU::NCECompressConvolutionOpModel<CompressConvolutionOpModel, VPU::NCECompressConvolutionOp>,
        public ConvMpeModeModel {};
class InterpolateOpModel :
        public VPU::NCEInterpolateOpModel<InterpolateOpModel, VPU::NCEInterpolateOp>,
        public ConvMpeModeModel {};

class MatMulOpModel : public VPU::NCEMatMulOpModel<MatMulOpModel, VPU::NCEMatMulOp>, public ConvMpeModeModel {
public:
    std::optional<vpux::VPU::MPEEngineAttr> getMpeEngine(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEMatMulOp>(op).getMpeEngine();
    }
};
class AveragePoolOpModel :
        public VPU::NCEAveragePoolOpModel<AveragePoolOpModel, VPU::NCEAveragePoolOp>,
        public Cuboid16MpeModeModel {};

class MaxPoolOpModel : public VPU::NCEMaxPoolOpModel<MaxPoolOpModel, VPU::NCEMaxPoolOp>, public Cuboid16MpeModeModel {};
class EltwiseOpModel : public VPU::NCEEltwiseOpModel<EltwiseOpModel, VPU::NCEEltwiseOp>, public Cuboid8MpeModeModel {};
class PermuteOpModel : public VPU::NCEEltwiseOpModel<PermuteOpModel, VPU::NCEPermuteOp>, public Cuboid16MpeModeModel {};

}  // namespace

void vpux::VPU::arch37xx::registerNCEOpInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEConvolutionOp::attachInterface<ConvolutionOpModel>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<CompressConvolutionOpModel>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<DepthConvolutionOpModel>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<MaxPoolOpModel>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<AveragePoolOpModel>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<EltwiseOpModel>(*ctx);
        VPU::NCEPermuteOp::attachInterface<PermuteOpModel>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<InterpolateOpModel>(*ctx);
        VPU::NCEMatMulOp::attachInterface<MatMulOpModel>(*ctx);
        VPU::NCEReduceOp::attachInterface<ReduceOpModel>(*ctx);
    });
}
