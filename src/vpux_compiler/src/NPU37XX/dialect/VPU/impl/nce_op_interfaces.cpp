//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"

using namespace vpux;

namespace {

class ConvMpeModeModel {
public:
    VPU::MPEMode getMpeModeImpl(mlir::Operation*, mlir::Type, mlir::Type, ShapeRef shape) const {
        return VPU::getMpeModeForConv(shape, VPU::getCuboidMpeGrids());
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
        public VPU::NCEDepthConvolutionOpModel<DepthConvolutionOpModel, VPU::NCEDepthConvolutionOp>,
        public Cuboid16MpeModeModel {};
class CompressConvolutionOpModel :
        public VPU::NCECompressConvolutionOpModel<CompressConvolutionOpModel, VPU::NCECompressConvolutionOp>,
        public ConvMpeModeModel {};
class InterpolateOpModel :
        public VPU::NCEInterpolateOpModel<InterpolateOpModel, VPU::NCEInterpolateOp>,
        public ConvMpeModeModel {};

class MatMulOpModel : public VPU::NCEMatMulOpModel<MatMulOpModel, VPU::NCEMatMulOp>, public ConvMpeModeModel {
public:
    VPU::MPEMode getMpeModeImpl(mlir::Operation* op, mlir::Type inElemType, mlir::Type outElemType,
                                ShapeRef shape) const {
        if (shape.size() == DimsGroups5D::Act::numDims) {
            return VPU::MPEMode::CUBOID_16x16;
        }
        return ConvMpeModeModel::getMpeModeImpl(op, inElemType, outElemType, shape);
    }

    std::optional<vpux::VPU::MPEEngineAttr> getMpeEngine(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEMatMulOp>(op).getMpeEngine();
    }
};
class AveragePoolOpModel :
        public VPU::NCEAveragePoolOpModel<AveragePoolOpModel, VPU::NCEAveragePoolOp>,
        public Cuboid16MpeModeModel {};

class MaxPoolOpModel : public VPU::NCEMaxPoolOpModel<MaxPoolOpModel, VPU::NCEMaxPoolOp>, public Cuboid16MpeModeModel {};
class EltwiseOpModel : public VPU::NCEEltwiseOpModel<EltwiseOpModel, VPU::NCEEltwiseOp>, public Cuboid8MpeModeModel {
public:
    mlir::Value getWeightTableScaleOperand(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEEltwiseOp>(op).getWeightTableScale();
    }
    mlir::Value getWeightTableBiasOperand(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEEltwiseOp>(op).getWeightTableBias();
    }
    mlir::Value getReduceXyMax(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEEltwiseOp>(op).getReduceXyMax();
    }
    mlir::Value getReduceXyMin(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEEltwiseOp>(op).getReduceXyMin();
    }
    mlir::Value getReduceTensorMinMax(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEEltwiseOp>(op).getReduceTensorMinMax();
    }
    bool hasReduceOutput(mlir::Operation* op) const {
        return VPU::hasAnyReduceOutput<VPU::NCEEltwiseOp>(op);
    }
};
class PermuteOpModel : public VPU::NCEEltwiseOpModel<PermuteOpModel, VPU::NCEPermuteOp>, public Cuboid16MpeModeModel {};

}  // namespace

void vpux::VPU::arch37xx::registerNCEOpInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEConvolutionOp::attachInterface<ConvolutionOpModel>(*ctx);
        VPU::NCEMatMulOp::attachInterface<MatMulOpModel>(*ctx);
        VPU::NCEReduceOp::attachInterface<ReduceOpModel>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<InterpolateOpModel>(*ctx);
        VPU::NCECompressConvolutionOp::attachInterface<CompressConvolutionOpModel>(*ctx);
    });
    registerCommonNCEOpInterfaces(registry);
}

void vpux::VPU::arch37xx::registerCommonNCEOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEDepthConvolutionOp::attachInterface<DepthConvolutionOpModel>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<MaxPoolOpModel>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<AveragePoolOpModel>(*ctx);
        VPU::NCEEltwiseOp::attachInterface<EltwiseOpModel>(*ctx);
        VPU::NCEPermuteOp::attachInterface<PermuteOpModel>(*ctx);
    });
}
