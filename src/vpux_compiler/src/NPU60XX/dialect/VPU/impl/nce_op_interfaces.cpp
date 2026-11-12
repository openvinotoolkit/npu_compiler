//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/nce_op_interfaces.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"

namespace {

template <typename ConcreteOp>
class MpeModeModel {
public:
    vpux::VPU::MPEMode getMpeModeImpl(mlir::Operation* op, mlir::Type, mlir::Type, vpux::ShapeRef shape) const {
        const auto engine =
                mlir::dyn_cast_if_present<vpux::VPU::MPEEngine60XXAttr>(mlir::cast<ConcreteOp>(op).getMpeEngineAttr());
        if (engine != nullptr && engine.getMode().getValue() == vpux::VPU::MPEEngine60XXMode::DCIM) {
            return vpux::VPU::MPEMode::DCIM_IC128;
        }
        // NPU60XX SCL uses the same reduced accumulator grid as NPU50XX.
        return vpux::VPU::getMpeModeForConv(shape, vpux::VPU::getCuboidMpeGridsWithReducedAccumulators());
    }
};

class ConvolutionOpModel :
        public vpux::VPU::NCEConvolutionOpModel<ConvolutionOpModel, vpux::VPU::NCEConvolutionOp>,
        public MpeModeModel<vpux::VPU::NCEConvolutionOp> {
public:
    std::optional<vpux::VPU::MPEEngineAttr> getMpeEngine(mlir::Operation* op) const {
        return mlir::cast<vpux::VPU::NCEConvolutionOp>(op).getMpeEngine();
    }
};

class MatMulOpModel :
        public vpux::VPU::NCEMatMulOpModel<MatMulOpModel, vpux::VPU::NCEMatMulOp>,
        public MpeModeModel<vpux::VPU::NCEMatMulOp> {
public:
    vpux::VPU::MPEMode getMpeModeImpl(mlir::Operation* op, mlir::Type inElemType, mlir::Type outElemType,
                                      vpux::ShapeRef shape) const {
        const auto engine = mlir::dyn_cast_if_present<vpux::VPU::MPEEngine60XXAttr>(
                mlir::cast<vpux::VPU::NCEMatMulOp>(op).getMpeEngineAttr());
        if (engine != nullptr && engine.getMode().getValue() == vpux::VPU::MPEEngine60XXMode::DCIM) {
            return vpux::VPU::MPEMode::DCIM_IC128;
        }
        if (shape.size() == vpux::DimsGroups5D::Act::numDims) {
            return vpux::VPU::MPEMode::CUBOID_16x16;
        }
        return MpeModeModel<vpux::VPU::NCEMatMulOp>::getMpeModeImpl(op, inElemType, outElemType, shape);
    }

    std::optional<vpux::VPU::MPEEngineAttr> getMpeEngine(mlir::Operation* op) const {
        return mlir::cast<vpux::VPU::NCEMatMulOp>(op).getMpeEngine();
    }
};

}  // namespace

void vpux::VPU::arch60xx::registerNCEOpInterface(mlir::DialectRegistry& registry) {
    // The other NCE operations retain their shared models; convolution and matmul select
    // their execution modes according to the NPU60XX MPE engine.
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEConvolutionOp::attachInterface<ConvolutionOpModel>(*ctx);
        VPU::NCEMatMulOp::attachInterface<MatMulOpModel>(*ctx);
    });
    arch50xx::registerConvolutionLikeNCEOpInterfaces(registry);
    arch37xx::registerCommonNCEOpInterfaces(registry);
}
