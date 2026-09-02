//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/interfaces/se_op_models.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;
using namespace IE;

namespace {

// While adjusting the layout, an intermediate Reorder operation can be introduced, before it gets fused into the
// filter constant
bool isFilterConst(mlir::Value filter) {
    if (auto reorderOp = filter.getDefiningOp<IE::ReorderOp>()) {
        filter = reorderOp.getInput();
    }

    auto constOp = filter.getDefiningOp<Const::DeclareOp>();
    if (auto fqOp = filter.getDefiningOp<IE::FakeQuantizeOp>()) {
        constOp = fqOp.getInput().getDefiningOp<Const::DeclareOp>();
    }

    if (auto dequantOp = filter.getDefiningOp<IE::DequantizeOp>()) {
        constOp = dequantOp.getInput().getDefiningOp<Const::DeclareOp>();
    }

    return constOp != nullptr;
}

// TransposedConvolution / GroupTransposedConvolution: validates shapes, pads, dilations,
// then delegates to NCE conv check. Supports both 4D filters (TransposedConv)
// and 5D filters (GroupTransposedConv).
template <class MainOpType>
class SETransposedConvOpModel final :
        public IE::SEOpInterface::ExternalModel<SETransposedConvOpModel<MainOpType>, MainOpType> {
public:
    bool isSupported(mlir::Operation* op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                     bool /*checkBatch*/) const {
        auto concreteOp = mlir::cast<MainOpType>(op);

        auto inputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getInput().getType());
        auto filterType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getFilter().getType());
        auto outputType = mlir::cast<vpux::NDTypeInterface>(concreteOp.getOutput().getType());

        if (inputType.getShape().size() != 4) {
            logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
            return false;
        }
        const auto filterRank = filterType.getShape().size();
        if (filterRank != 4 && filterRank != 5) {
            logCb(formatv("Only 4D or 5D filters are supported, got {0} dimensions", filterRank));
            return false;
        }
        if (outputType.getShape().size() != 4) {
            logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
            return false;
        }
        // For 4D filters (TransposedConv), verify input/filter channel consistency
        if (filterRank == 4 && inputType.getShape()[Dims4D::Act::C] != filterType.getShape()[Dims4D::Filter::IC]) {
            logCb(formatv("The filter channels are inconsistent with activation channels"));
            return false;
        }
        // For 5D filters (GroupTransposedConv), the filter must be a constant
        if (filterRank == 5 && !isFilterConst(concreteOp.getFilter())) {
            return false;
        }
        if (concreteOp.getPadsBegin().size() != 2 || concreteOp.getPadsEnd().size() != 2) {
            logCb(formatv("Pads begin and pads end should have a 2D shape, but got {0}D and {1}D",
                          concreteOp.getPadsBegin().size(), concreteOp.getPadsEnd().size()));
            return false;
        }

        const auto dilationsAttr = concreteOp.getDilations();
        const auto dilations = parseIntArrayAttr<int64_t>(dilationsAttr);
        if (dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1) {
            logCb(formatv("Dilated transposed convolution is not supported"));
            return false;
        }

        auto origPads = PadInfo(concreteOp.getPadsBegin(), concreteOp.getPadsEnd());
        if (origPads.left < 0 || origPads.top < 0 || origPads.right < 0 || origPads.bottom < 0) {
            logCb(formatv("Negative padding is unsupported"));
            return false;
        }

        const auto filterShape = filterType.getShape().raw();
        const auto KY = filterShape[filterShape.size() - 2];
        const auto KX = filterShape[filterShape.size() - 1];

        const auto outputPadding = Shape(parseIntArrayAttr<int64_t>(concreteOp.getSpatialOutputPadding()));
        const auto inputShape = getBoundedShape(inputType);
        const auto origKernelStrides = Shape(parseIntArrayAttr<int64_t>(concreteOp.getStrides()));
        const auto zerosY = origKernelStrides[Dims4D::Strides::Y] - 1;
        const auto zerosX = origKernelStrides[Dims4D::Strides::X] - 1;
        const auto newPadTop = KY - 1;
        const auto newPadBottom = KY - 1 + outputPadding[Dims4D::PadsOutput::Y];
        const auto newPadLeft = KX - 1;
        const auto newPadRight = KX - 1 + outputPadding[Dims4D::PadsOutput::X];
        const auto newY =
                inputShape[Dims4D::Act::H] + zerosY * (inputShape[Dims4D::Act::H] - 1) + newPadTop + newPadBottom;
        const auto newX =
                inputShape[Dims4D::Act::W] + zerosX * (inputShape[Dims4D::Act::W] - 1) + newPadLeft + newPadRight;

        const Shape newInputShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], newY, newX};

        // In case of dynamic bounded types, check that the NCEConv is legal on the bounded shape
        mlir::Type convInputType = inputType;
        if (mlir::isa<Core::BoundedTensorType>(convInputType)) {
            convInputType = vpux::getTensorType(newInputShape, inputType.getElementType(), inputType.getDimsOrder(),
                                                inputType.getMemSpace(), /*Bounds=*/{}, /*DynamicDimsMask=*/{});
        } else {
            convInputType = mlir::cast<NDTypeInterface>(convInputType).changeShape(newInputShape);
        }

        mlir::Type convFilterType = filterType;
        if (mlir::isa<Core::BoundedTensorType>(convFilterType)) {
            convFilterType = vpux::getTensorType(getBoundedShape(convFilterType), filterType.getElementType(),
                                                 filterType.getDimsOrder(), filterType.getMemSpace(), /*Bounds=*/{},
                                                 /*DynamicDimsMask=*/{});
        }

        mlir::Type convOutputType = outputType;
        if (mlir::isa<Core::BoundedTensorType>(convOutputType)) {
            convOutputType = vpux::getTensorType(getBoundedShape(convOutputType), outputType.getElementType(),
                                                 outputType.getDimsOrder(), outputType.getMemSpace(), /*Bounds=*/{},
                                                 /*DynamicDimsMask=*/{});
        }

        const int64_t SY = 1;
        const int64_t SX = 1;

        PadInfo pads(0, 0, 0, 0);

        return VPU::isNCEConvSupported(op, convInputType, convFilterType, convOutputType, dilations, KY, KX, SY, SX,
                                       pads, checkLayout, checkChannelAlignment, logCb);
    }
};

}  // namespace

void vpux::IE::arch37xx::registerSEOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::InterpolateOp::attachInterface<SEInterpolateOpModel<IE::InterpolateOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<SETransposedConvOpModel<IE::TransposedConvolutionOp>>(*ctx);
        IE::GroupTransposedConvolutionOp::attachInterface<SETransposedConvOpModel<IE::GroupTransposedConvolutionOp>>(
                *ctx);
        IE::PadOp::attachInterface<SEPadOpModel<IE::PadOp, /*HasSparsityMapSupport=*/true>>(*ctx);
        IE::RollOp::attachInterface<SERollOpModel<IE::RollOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<SEDilatedGroupConvOpModel<IE::GroupConvolutionOp>>(*ctx);
    });

    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::InterpolateOp::attachInterface<SEInterpolateOpModel<VPU::InterpolateOp>>(*ctx);
        VPU::TransposedConvolutionOp::attachInterface<SETransposedConvOpModel<VPU::TransposedConvolutionOp>>(*ctx);
        VPU::PadOp::attachInterface<SEPadOpModel<VPU::PadOp, /*HasSparsityMapSupport=*/true>>(*ctx);
        VPU::RollOp::attachInterface<SERollOpModel<VPU::RollOp>>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<SEDilatedGroupConvOpModel<VPU::GroupConvolutionOp>>(*ctx);
    });
}
