//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"

namespace vpux::IE::arch60xx {
class ElemTypeInfoMaxPoolOpModel final : public IE::ElemTypeInfoOpInterface::FallbackModel<ElemTypeInfoMaxPoolOpModel> {
public:
    static void inferElemTypeInfo(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected MaxPoolOp, got {0} at loc {1}", op->getName(), op->getLoc());

        if (VPU::NCEMaxPoolOp::isSupported(origOp, emptyLogCb, /*checkLayout=*/false,
                                           /*checkChannelAlignment=*/false)) {
            propagateElementTypeDown(info);
        }
    }

    static void inferElemTypeInfoUp(mlir::Operation* op, LayerDataInfo<mlir::Type>& info) {
        auto origOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
        VPUX_THROW_WHEN(origOp == nullptr, "Expected MaxPoolOp, got {0} at loc {1}", op->getName(), op->getLoc());
        const auto outputElemType = info.getOutputs()[0];
        if (auto qElementType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputElemType)) {
            if (qElementType.getStorageType().isInteger(16)) {
                return;
            }
        }

        if (VPU::NCEMaxPoolOp::isSupported(origOp, emptyLogCb, /*checkLayout=*/false,
                                           /*checkChannelAlignment=*/false)) {
            propagateElementTypeUp(info);
        }
    }
    static LayerDataInfo<mlir::Type> getElemTypeInfo(mlir::Operation* op) {
        return vpux::IE::getElemTypeInfo(op);
    }
};

}  // namespace vpux::IE::arch60xx
void vpux::IE::arch60xx::registerElemTypeInfoOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::AffineReshapeOp::attachInterface<ElemTypeInfoAffineReshapeOpModel>(*ctx);
        IE::ClampOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ConcatOp::attachInterface<ElemTypeInfoConcatOpModel>(*ctx);
        IE::DepthToSpaceOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ExpandOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ExpandDilatedOp::attachInterface<ElemTypeInfoExpandDilatedOpModel>(*ctx);
        IE::InterpolateOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::MaxPoolOp::attachInterface<vpux::IE::arch60xx::ElemTypeInfoMaxPoolOpModel>(*ctx);
        IE::ReduceMaxOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ReorderOp::attachInterface<ElemTypeInfoReorderOpModel>(*ctx);
        IE::ReshapeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SliceOp::attachInterface<ElemTypeInfoSliceOpModel>(*ctx);
        IE::SpaceToDepthOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SplitOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::SqueezeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::TileOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::TransposeOp::attachInterface<ElemTypeInfoTransposeOpModel>(*ctx);
        IE::UnsqueezeOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::UpsamplingOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::PadOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::LayoutCastOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::PermuteCastOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
        IE::ShapeCastOp::attachInterface<PerTensorElemTypeInfoOpModel>(*ctx);
    });
}
