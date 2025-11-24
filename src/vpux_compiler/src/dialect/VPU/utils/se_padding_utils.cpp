//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/se_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

namespace {

// Empirical threshold based on profiling traces and based on RTL simulations.
// VPUNN CostModel does not currently model differences in performance coming from SEP usage.
constexpr int64_t SEP_PAD_IC_NUM_PERF_THRESHOLD = 32;

bool isSupportedSEPPadImpl(mlir::Operation* op, NDTypeInterface inputType, NDTypeInterface outputType,
                           IE::PadMode padMode, mlir::ArrayAttr padsBeginAttr, mlir::ArrayAttr padsEndAttr,
                           mlir::FloatAttr padValueAttr, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                           bool supportsInputActCompression, mlir::MLIRContext* ctx) {
    if (inputType.getShape().size() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
        return false;
    }
    if (outputType.getShape().size() != 4) {
        logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
        return false;
    }

    if (padsBeginAttr == nullptr || padsEndAttr == nullptr) {
        logCb(formatv("Only constant pads begin and pads end are supported"));
        return false;
    }

    if (padMode == IE::PadMode::CONSTANT) {
        if (padValueAttr == nullptr) {
            logCb(formatv("PadMode with CONSTANT should have constant pad value"));
            return false;
        }
        const auto padValue = padValueAttr.getValue().convertToDouble();
        if (!isDoubleEqual(padValue, 0.f)) {
            logCb(formatv("Only CONSTANT mode with pad value '0' is supported"));
            return false;
        }
    }

    const auto inputShape = inputType.getShape();
    const auto padsBegin = parseIntArrayAttr<int64_t>(padsBeginAttr);
    const auto padsEnd = parseIntArrayAttr<int64_t>(padsEndAttr);
    auto isSpatialPadding = [](ArrayRef<int64_t> padsValue) {
        return padsValue[Dims4D::Act::N.ind()] == 0 && padsValue[Dims4D::Act::C.ind()] == 0;
    };
    if (!isSpatialPadding(padsBegin) || !isSpatialPadding(padsEnd)) {
        logCb(formatv("Only spatial padding is supported"));
        return false;
    }

    const auto newY = inputShape[Dims4D::Act::H] + padsBegin[Dims4D::Act::H.ind()] + padsEnd[Dims4D::Act::H.ind()];
    const auto newX = inputShape[Dims4D::Act::W] + padsBegin[Dims4D::Act::W.ind()] + padsEnd[Dims4D::Act::W.ind()];
    const Shape newInputShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], newY, newX};
    inputType = inputType.changeShape(newInputShape);

    // SEP PadOp will get fused into the next Convolution, if that exists, or it will be converted
    // into a 1x1 Convolution, input sparsity will be enabled and storage element pointers will be used.
    // E-163345 shows that the overhead of storage element table reads from CMX will bring a significant performance
    // regression when Conv Input Channel number is low.
    if (newInputShape[Dims4D::Act::C] <= SEP_PAD_IC_NUM_PERF_THRESHOLD) {
        return false;
    }
    auto weightShape =
            Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::C], /*KY=*/1, /*KX=*/1});
    mlir::Type elemType = mlir::Float16Type::get(ctx);
    if (mlir::isa<mlir::quant::QuantizedType>(inputType.getElementType())) {
        elemType = mlir::quant::UniformQuantizedType::get(
                /*flags=*/0, /*storageType=*/getUInt8Type(ctx), /*expressedType=*/mlir::Float16Type::get(ctx),
                /*scale=*/static_cast<double>(1.0f), /*zeroPoint=*/0, /*storageTypeMin=*/0, /*storageTypeMax=*/255);
    }
    const auto tensorAttr = vpux::getTensorAttr(ctx, DimsOrder::OYXI, nullptr);
    const auto weightsType =
            mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(weightShape.raw(), elemType, tensorAttr));

    const auto dilations = SmallVector<int64_t>{1, 1};
    const auto pads = PadInfo(0, 0, 0, 0);

    // When SEP Pad Op is enabled, it will be converted to NCEConvolution
    // with kernel size, strides, and dilations set to [1, 1]
    // This is to verify that it can meet the NCEConvolution HW requirements, such as channel alignment and layout.
    return VPU::isNCEConvSupported(op, inputType, weightsType, outputType, dilations, /*KY=*/1, /*KX=*/1, /*SY=*/1,
                                   /*SX=*/1, pads, checkLayout, checkChannelAlignment, logCb,
                                   supportsInputActCompression);
}

}  // namespace

bool VPU::isSupportedSEPPadOp(IE::PadOp padOp, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                              bool supportsInputActCompression) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(padOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(padOp.getOutput().getType());
    return isSupportedSEPPadImpl(padOp.getOperation(), inputType, outputType, padOp.getMode(),
                                 padOp.getPadsBeginAttrAttr(), padOp.getPadsEndAttrAttr(), padOp.getPadValueAttrAttr(),
                                 logCb, checkLayout, checkChannelAlignment, supportsInputActCompression,
                                 padOp.getContext());
}

bool VPU::isSupportedSEPPadOp(VPU::PadOp padOp, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                              bool supportsInputActCompression) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(padOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(padOp.getOutput().getType());
    return isSupportedSEPPadImpl(padOp.getOperation(), inputType, outputType, padOp.getMode(),
                                 padOp.getPadsBeginAttrAttr(), padOp.getPadsEndAttrAttr(), padOp.getPadValueAttrAttr(),
                                 logCb, checkLayout, checkChannelAlignment, supportsInputActCompression,
                                 padOp.getContext());
}
