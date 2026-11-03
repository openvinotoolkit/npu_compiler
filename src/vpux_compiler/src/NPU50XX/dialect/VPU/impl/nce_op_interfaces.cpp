//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"

using namespace vpux;

namespace {

class ConvMpeModeModel {
public:
    VPU::MPEMode getMpeModeImpl(mlir::Operation*, mlir::Type, mlir::Type, ShapeRef shape) const {
        return VPU::getMpeModeForConv(shape, VPU::getCuboidMpeGridsWithReducedAccumulators());
    }
};

class ConvolutionOpModel :
        public VPU::NCEConvolutionOpModel<ConvolutionOpModel, VPU::NCEConvolutionOp>,
        public ConvMpeModeModel {
public:
    std::optional<VPU::MPEEngineAttr> getMpeEngine(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEConvolutionOp>(op).getMpeEngine();
    }
};
class MatMulOpModel : public VPU::NCEMatMulOpModel<MatMulOpModel, VPU::NCEMatMulOp>, public ConvMpeModeModel {
public:
    VPU::MPEMode getMpeModeImpl(mlir::Operation* op, mlir::Type inElemType, mlir::Type outElemType,
                                ShapeRef shape) const {
        if (shape.size() == DimsGroups5D::Act::numDims) {
            // Keep the grouped MatMul fallback: VPUNN pre-split support was reverted because folding G into N
            // loses the group count when VPUNN treats N as 1, producing inaccurate costs E#160854.
            // On NPU50XX, select CUBOID_4x16 for 1x1 kernels to allow wt_dense_opt_en for two-bit palettized
            // weights E#232349, other kernels retain CUBOID_16x16.
            const auto kernel = VPU::getKernelSize(mlir::cast<VPU::NCEMatMulOp>(op));
            const auto isOneByOneKernel =
                    kernel[DimsGroups5D::Kernel::Y.ind()] == 1 && kernel[DimsGroups5D::Kernel::X.ind()] == 1;
            return isOneByOneKernel ? VPU::MPEMode::CUBOID_4x16 : VPU::MPEMode::CUBOID_16x16;
        }
        return ConvMpeModeModel::getMpeModeImpl(op, inElemType, outElemType, shape);
    }

    std::optional<VPU::MPEEngineAttr> getMpeEngine(mlir::Operation* op) const {
        return mlir::cast<VPU::NCEMatMulOp>(op).getMpeEngine();
    }
};
class CompressConvolutionOpModel :
        public VPU::NCECompressConvolutionOpModel<CompressConvolutionOpModel, VPU::NCECompressConvolutionOp>,
        public ConvMpeModeModel {};
class InterpolateOpModel :
        public VPU::NCEInterpolateOpModel<InterpolateOpModel, VPU::NCEInterpolateOp>,
        public ConvMpeModeModel {};
class ReduceOpModel : public VPU::NCEReduceOpModel<ReduceOpModel, VPU::NCEReduceOp>, public ConvMpeModeModel {};

}  // namespace

void vpux::VPU::arch50xx::registerNCEOpInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEConvolutionOp::attachInterface<ConvolutionOpModel>(*ctx);
        VPU::NCEMatMulOp::attachInterface<MatMulOpModel>(*ctx);
    });
    registerConvolutionLikeNCEOpInterfaces(registry);
    arch37xx::registerCommonNCEOpInterfaces(registry);
}

void vpux::VPU::arch50xx::registerConvolutionLikeNCEOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCECompressConvolutionOp::attachInterface<CompressConvolutionOpModel>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<InterpolateOpModel>(*ctx);
        VPU::NCEReduceOp::attachInterface<ReduceOpModel>(*ctx);
    });
}
