//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/conv_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;
using namespace VPU;

bool vpux::VPU::isNCEConvSupported(mlir::Operation* op, NDTypeInterface inputType, NDTypeInterface filterType,
                                   NDTypeInterface outputType, ArrayRef<int64_t> dilations, int64_t KY, int64_t KX,
                                   int64_t SY, int64_t SX, PadInfo pads, bool checkLayout, bool checkChannelAlignment,
                                   LogCb logCb, bool supportsInputActCompression) {
    if (outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (dilations.size() != 2) {
        logCb(formatv("Expected dilations size to be 2, got '{0}'", dilations.size()));
        return false;
    }
    if (dilations[0] != 1 || dilations[1] != 1) {
        logCb(formatv("Dilated convolution is not supported"));
        return false;
    }

    if (!NCEInvariant::isAttrsSupported(op, KY, KX, SY, SX, pads.top, pads.bottom, pads.left, pads.right, logCb)) {
        return false;
    }

    const auto inputOrder = inputType.getDimsOrder();
    const auto isChannelMajor = inputOrder == DimsOrder::NCHW;

    if (checkChannelAlignment) {
        auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op);
        auto inputAlignment = iface != nullptr ? iface.getInputChannelAlignment()
                                               : vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType());
        auto outputAlignment = iface != nullptr ? iface.getOutputChannelAlignment()
                                                : vpux::VPU::NCEInvariant::getAlignment(outputType.getElementType());
        if (!NCEInvariant::isInputActTypeSupported(inputType, !isChannelMajor ? inputAlignment : 1,
                                                   supportsInputActCompression) ||
            !NCEInvariant::isOutputActTypeSupported(outputType, outputAlignment)) {
            logCb(formatv("Misaligned tensor shape"));
            return false;
        }
    }

    if (checkLayout) {
        const auto filterOrder = filterType.getDimsOrder();

        if (inputOrder != DimsOrder::NHWC && inputOrder != DimsOrder::NCHW) {
            logCb(formatv("Unsupported input layout '{0}'", inputOrder));
            return false;
        }
        if (filterOrder != DimsOrder::OYXI) {
            logCb(formatv("Unsupported filter layout '{0}'", filterOrder));
            return false;
        }
    }

    return true;
}

bool vpux::VPU::isSupportedConv(IE::ConvolutionOp op, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                                bool supportsInputActCompression) {
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());

    const auto filterShape = getShape(op.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(op.getStrides()));
    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getFilter().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());

    if (op.getPadsBegin().size() != 2 || op.getPadsEnd().size() != 2) {
        logCb(formatv("Pads begin and pads end should have a 2D shape, but got {0}D and {1}D", op.getPadsBegin().size(),
                      op.getPadsEnd().size()));
        return false;
    }

    const auto pads = PadInfo(op.getPadsBegin(), op.getPadsEnd());

    return VPU::isNCEConvSupported(op, inputType, filterType, outputType, dilations, KY, KX, SY, SX, pads, checkLayout,
                                   checkChannelAlignment, logCb, supportsInputActCompression);
}

namespace {

bool isFilterConst(mlir::Value filter) {
    // While adjusting the layout, an intermediate Reorder operation can be introduced, before it gets fused into the
    // filter constant
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

bool isSupportedSEPTransposedConvImpl(mlir::Operation* op, NDTypeInterface inputType, NDTypeInterface filterType,
                                      NDTypeInterface outputType, mlir::ArrayAttr kernelStridesAttr,
                                      mlir::ArrayAttr dilationsAttr, mlir::ArrayAttr outputPaddingAttr,
                                      PadInfo origPads, LogCb logCb, bool checkLayout, bool checkChannelAlignment,
                                      bool supportsInputActCompression) {
    const auto dilations = parseIntArrayAttr<int64_t>(dilationsAttr);
    if (dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1) {
        logCb(formatv("Dilated transposed convolution is not supported"));
        return false;
    }

    if (origPads.left < 0 || origPads.top < 0 || origPads.right < 0 || origPads.bottom < 0) {
        logCb(formatv("Negative padding is unsupported"));
        return false;
    }

    const auto filterShape = filterType.getShape().raw();
    const auto KY = filterShape[filterShape.size() - 2];
    const auto KX = filterShape[filterShape.size() - 1];

    const auto outputPadding = Shape(parseIntArrayAttr<int64_t>(outputPaddingAttr));

    const auto inputShape = getBoundedShape(inputType);
    const auto origKernelStrides = Shape(parseIntArrayAttr<int64_t>(kernelStridesAttr));
    const auto zerosY = origKernelStrides[Dims4D::Strides::Y] - 1;
    const auto zerosX = origKernelStrides[Dims4D::Strides::X] - 1;
    const auto newPadTop = KY - 1;
    const auto newPadBottom = KY - 1 + outputPadding[Dims4D::PadsOutput::Y];
    const auto newPadLeft = KX - 1;
    const auto newPadRight = KX - 1 + outputPadding[Dims4D::PadsOutput::X];
    const auto newY = inputShape[Dims4D::Act::H] + zerosY * (inputShape[Dims4D::Act::H] - 1) + newPadTop + newPadBottom;
    const auto newX = inputShape[Dims4D::Act::W] + zerosX * (inputShape[Dims4D::Act::W] - 1) + newPadLeft + newPadRight;

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

    return VPU::isNCEConvSupported(op, convInputType, convFilterType, convOutputType, dilations, KY, KX, SY, SX, pads,
                                   checkLayout, checkChannelAlignment, logCb, supportsInputActCompression);
}

}  // namespace

bool VPU::isSupportedSEPTransposedConv(IE::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                       bool checkChannelAlignment, bool supportsInputActCompression) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getFilter().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    if (inputType.getShape().size() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
        return false;
    }
    if (filterType.getShape().size() != 4) {
        logCb(formatv("Only 4D filters are supported, got {0} dimensions", filterType.getShape().size()));
        return false;
    }
    if (outputType.getShape().size() != 4) {
        logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
        return false;
    }
    if (inputType.getShape()[Dims4D::Act::C] != filterType.getShape()[Dims4D::Filter::IC]) {
        logCb(formatv("The filter channels are inconsistent with activation channels"));
        return false;
    }
    if (op.getPadsBegin().size() != 2 || op.getPadsEnd().size() != 2) {
        logCb(formatv("Pads begin and pads end should have a 2D shape, but got {0}D and {1}D", op.getPadsBegin().size(),
                      op.getPadsEnd().size()));
        return false;
    }

    auto origPads = PadInfo(op.getPadsBegin(), op.getPadsEnd());
    return isSupportedSEPTransposedConvImpl(op.getOperation(), inputType, filterType, outputType, op.getStrides(),
                                            op.getDilations(), op.getSpatialOutputPadding(), origPads, logCb,
                                            checkLayout, checkChannelAlignment, supportsInputActCompression);
}

bool VPU::isSupportedSEPTransposedConv(IE::GroupTransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                       bool checkChannelAlignment, bool supportsInputActCompression) {
    if (!isFilterConst(op.getFilter())) {
        return false;
    }
    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getFilter().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    if (inputType.getShape().size() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
        return false;
    }
    if (filterType.getShape().size() != 5) {
        logCb(formatv("Only 5D filters are supported, got {0} dimensions", filterType.getShape().size()));
        return false;
    }
    if (outputType.getShape().size() != 4) {
        logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
        return false;
    }
    if (op.getPadsBegin().size() != 2 || op.getPadsEnd().size() != 2) {
        logCb(formatv("Pads begin and pads end should have a 2D shape, but got {0}D and {1}D", op.getPadsBegin().size(),
                      op.getPadsEnd().size()));
        return false;
    }

    auto origPads = PadInfo(op.getPadsBegin(), op.getPadsEnd());
    return isSupportedSEPTransposedConvImpl(op.getOperation(), inputType, filterType, outputType, op.getStrides(),
                                            op.getDilations(), op.getSpatialOutputPadding(), origPads, logCb,
                                            checkLayout, checkChannelAlignment, supportsInputActCompression);
}

bool VPU::isSupportedSEPTransposedConv(VPU::TransposedConvolutionOp op, LogCb logCb, bool checkLayout,
                                       bool checkChannelAlignment, bool supportsInputActCompression) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto filterType = mlir::cast<vpux::NDTypeInterface>(op.getFilter().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType());
    if (inputType.getShape().size() != 4) {
        logCb(formatv("Only 4D inputs are supported, got {0} dimensions", inputType.getShape().size()));
        return false;
    }
    if (filterType.getShape().size() != 4) {
        logCb(formatv("Only 4D filters are supported, got {0} dimensions", filterType.getShape().size()));
        return false;
    }
    if (outputType.getShape().size() != 4) {
        logCb(formatv("Only 4D outputs are supported, got {0} dimensions", outputType.getShape().size()));
        return false;
    }
    if (inputType.getShape()[Dims4D::Act::C] != filterType.getShape()[Dims4D::Filter::IC]) {
        logCb(formatv("The filter channels are inconsistent with activation channels"));
        return false;
    }
    if (op.getPadsBegin().size() != 2 || op.getPadsEnd().size() != 2) {
        logCb(formatv("Pads begin and pads end should have a 2D shape, but got {0}D and {1}D", op.getPadsBegin().size(),
                      op.getPadsEnd().size()));
        return false;
    }

    auto origPads = PadInfo(op.getPadsBegin(), op.getPadsEnd());
    return isSupportedSEPTransposedConvImpl(op.getOperation(), inputType, filterType, outputType, op.getStrides(),
                                            op.getDilations(), op.getSpatialOutputPadding(), origPads, logCb,
                                            checkLayout, checkChannelAlignment, supportsInputActCompression);
}

std::optional<bool> VPU::isSEPConvCompatibleWithClusterStrategy(VPU::NCEConvolutionOp nceConv,
                                                                VPU::MultiClusterStrategy strategy) {
    auto sparseInput = mlir::dyn_cast<vpux::VPU::SparseTensorType>(nceConv.getInput().getType());
    if (sparseInput == nullptr) {
        return std::nullopt;
    }

    auto seAttr = mlir::dyn_cast_or_null<vpux::VPU::SERollAttr>(sparseInput.getSeAttr());
    if (seAttr != nullptr) {
        return VPU::isRollSEPConvCompatibleWithClusterStrategy(seAttr, strategy);
    }
    return std::nullopt;
}

mlir::LogicalResult vpux::VPU::verifyConvUtil(mlir::Location loc, mlir::Operation* op, ShapeRef filterShape,
                                              ShapeRef kernelStrides, PaddingAttr padAttr,
                                              std::optional<ShapeRef> weightsTableShape, mlir::Value output) {
    const auto logCb = [loc](const formatv_object_base& msg) {
        std::ignore = errorAt(loc, "{0}", msg.str());
    };

    const auto outputShape = getShape(output);
    auto OC = outputShape[Dims4D::Act::C];

    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto SY = kernelStrides[Dims4D::Strides::Y];
    const auto SX = kernelStrides[Dims4D::Strides::X];

    const auto padTop = padAttr.getTop().getValue().getSExtValue();
    const auto padBottom = padAttr.getBottom().getValue().getSExtValue();
    const auto padLeft = padAttr.getLeft().getValue().getSExtValue();
    const auto padRight = padAttr.getRight().getValue().getSExtValue();

    if (!VPU::NCEInvariant::isAttrsSupported(op, KY, KX, SY, SX, padTop, padBottom, padLeft, padRight, logCb)) {
        return mlir::failure();
    }

    if (VPU::canAutopadOutput(op)) {
        OC = vpux::VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
    }

    const auto expectedWeightsTableShape = VPU::NCESparsity::inferWeightsTableShape(OC);

    if (weightsTableShape.has_value() && weightsTableShape.value() != expectedWeightsTableShape) {
        return errorAt(loc, "Got wrong shape for 'weightsTable' '{0}', expected '{1}'", weightsTableShape.value(),
                       expectedWeightsTableShape);
    }

    return mlir::success();
}

PadInfo vpux::VPU::shrinkPadsForDilatedConvolution(const PadInfo& pads, const ArrayRef<int64_t> dilations) {
    // SEP Dilated GroupConv will follow a different path than usual dilated Convolution
    // Current method for dilated convolution is done via kernel expansion
    // 3x3 kernel with dilation 4,4 padding 4,4 will be expanded to 9x9 kernel with dilation 1,1
    // Padding provided (4) makes sense and passes NCEInvariant checks ( 0 <= Pad <= K/2)
    // However with new SEP approach kernel will stay 3x3 so padding should be shrinked
    // to 1x1 in this example following calculation below.
    // For more information SEP Dilated Group Convolution E87313 could be checked.
    // If there is no dilation ( dilationY/X =1) this calculation does not change padding.

    // Below formula is only valid for below 3 cases, and padding/dilation is symmetrical
    // if dilation = padding then newPadding = 1
    // if padding = 0 then newPadding = 0
    // No dilation (dilation =1) then padding = originalPadding
    const auto dilationY = dilations[Dims4D::Dilation::Y.ind()];
    const auto dilationX = dilations[Dims4D::Dilation::X.ind()];
    PadInfo newPads = pads;
    newPads.top = std::max<int64_t>(newPads.top - dilationY + 1, 0l);
    newPads.left = std::max<int64_t>(newPads.left - dilationX + 1, 0l);
    newPads.bottom = std::max<int64_t>(newPads.bottom - dilationY + 1, 0l);
    newPads.right = std::max<int64_t>(newPads.right - dilationX + 1, 0l);
    return newPads;
}

namespace {
using ScaleVecType =
        std::variant<std::vector<vpux::type::float8_e4m3>, std::vector<vpux::type::float8_e5m2>, std::vector<float>>;

// The original scale is computed as:
// ((activation_scale * weights_scale) / output_scale) * static_scale

// For the Conv ops we keep the activation_scale * weights_scale.
// The static_scale / output_scale val will be added to the last op
// after the decomposition.
template <typename T,
          typename = std::enable_if_t<std::is_same_v<vpux::type::float8_e4m3, T> ||
                                      std::is_same_v<vpux::type::float8_e5m2, T> || std::is_same_v<float, T>>>
mlir::Value updateScaleTableForConvOps(mlir::PatternRewriter& rewriter, mlir::Value origScaleTable,
                                       mlir::Type scaleTableType, ArrayRef<double> inputRescale,
                                       VPU::NCESparsity::PPEConverterCb ppeConverter,
                                       VPU::NCESparsity::ScaleRetrieveCb scaleRetrieveConverter,
                                       SmallVector<double>& outQuantScales, int64_t oCh, mlir::Location loc,
                                       Logger log) {
    outQuantScales.reserve(oCh);
    assert(origScaleTable != nullptr && "Scale table does not exist.");

    auto scaleTableConst = origScaleTable.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_WHEN(scaleTableConst == nullptr, "Scale table must be const");

    auto scaleTableContent = scaleTableConst.getContent();
    auto scaleTableValues = scaleTableContent.getValues<T>();

    std::vector<T> scaleTableVec;
    scaleTableVec.reserve(oCh);
    std::copy(scaleTableValues.begin(), scaleTableValues.end(), std::back_inserter(scaleTableVec));

    for (int64_t i = 0; i < oCh; ++i) {
        double origScale = 1.0;
        const auto newScale = ppeConverter(0, 0, inputRescale[i], scaleTableType);

        origScale = scaleRetrieveConverter(scaleTableVec[i], scaleTableType);
        scaleTableVec[i] = std::get<T>(newScale);

        outQuantScales.push_back(origScale / inputRescale[i]);
    }

    const auto scaleTableShape = VPU::NCESparsity::inferWeightsTableShape(oCh, /*newFormat=*/true);
    auto convScaleTable =
            VPU::createNewWeightsTableTensor<T>(rewriter, loc, scaleTableVec, scaleTableShape, scaleTableType);

    log.trace("Created updated scale table for decomposed Conv ops.");
    return convScaleTable;
}

// Do conv scale update as described above, but replace the corresponding values inside weights table
void updateScaleInConvWeightTable(mlir::Value origWeightTable, std::vector<int32_t>& weightTableVec,
                                  ArrayRef<double> inputRescale, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                  VPU::NCESparsity::ScaleRetrieveCb scaleRetrieveConverter, mlir::Type convInElemType,
                                  SmallVector<double>& outQuantScales, int64_t oCh, Logger log) {
    assert(origWeightTable != nullptr && "Weights table does not exist.");

    auto weightsTableConst = origWeightTable.getDefiningOp<Const::DeclareOp>();
    assert(weightsTableConst != nullptr && "Weights table must be const");
    auto weightsTableContent = weightsTableConst.getContent();

    auto weightsTableValues = weightsTableContent.getValues<int32_t>();
    std::copy(weightsTableValues.begin(), weightsTableValues.end(), std::back_inserter(weightTableVec));

    constexpr size_t scaleOffset = 2;
    for (int64_t i = 0; i < oCh; ++i) {
        const auto index = VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * i;
        const auto origScale = scaleRetrieveConverter(weightTableVec[index + scaleOffset], convInElemType);

        outQuantScales.push_back(origScale / inputRescale[i]);

        const QuantizationApproximation scaleApproximation(inputRescale[i]);
        const auto multShift =
                ppeConverter(checked_cast<uint8_t>(scaleApproximation.shift()),
                             checked_cast<int16_t>(scaleApproximation.mult()), inputRescale[i], convInElemType);

        weightTableVec[index + scaleOffset] = std::get<int32_t>(multShift);
    }
    log.trace("Updated scale values in weights table for decomposed Conv ops.");
}

mlir::Value updateDataPtrSpPtrAndBiasInConvWeightTable(mlir::PatternRewriter& rewriter,
                                                       std::vector<int32_t>& weightTableVec, int64_t weightSetSize,
                                                       int64_t sparsitySetSize, int64_t oCh, size_t convTile,
                                                       mlir::Type inputElemType, config::ArchKind arch,
                                                       mlir::Location loc, Logger log) {
    // Adjust the weights table pointers to correspond to the new offsets of the slices
    constexpr size_t weightsOffset = 0;
    constexpr size_t sparsityOffset = 1;
    constexpr size_t biasOffset = 3;

    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(arch, false);
    const auto zeroBias = std::get<int32_t>(biasConverter(0.0, inputElemType));

    for (int64_t i = 0; i < oCh; ++i) {
        auto index = VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * i;
        // Apply bias for the first convolution only
        // originalConvOp+bias = (convOp0+convOp1+...+convOpN)+bias = (convOp0+bias)+convOp1+...+convOpN
        if (convTile != 0) {
            weightTableVec[index + biasOffset] = zeroBias;
        }
        weightTableVec[index + weightsOffset] = checked_cast<int32_t>(i * weightSetSize);
        weightTableVec[index + sparsityOffset] = checked_cast<int32_t>(i * sparsitySetSize);
    }

    log.trace("Update weights and sparsity pointers in weight table.");
    if (convTile != 0) {
        log.trace("Zero'ed out the bias for {0}th Conv.", convTile);
    }

    const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(static_cast<int64_t>(oCh));
    return VPU::createWeightsTableTensor(rewriter, loc, weightTableVec, wtShape);
}

SmallVector<int32_t> getScaleValuesTableForLegacyWTDepthwiseOp(VPU::NCESparsity::PPEConverterCb ppeConverter,
                                                               ArrayRef<double> outQuantScales,
                                                               mlir::Type scaleTableType, mlir::MLIRContext* ctx) {
    auto getF32Scale = [&](double val) -> float {
        if (mlir::isa<mlir::Float8E4M3FNType>(scaleTableType)) {
            return std::get<vpux::type::float8_e4m3>(ppeConverter(0, 0, val, scaleTableType));
        }

        if (mlir::isa<mlir::Float8E5M2Type>(scaleTableType)) {
            return std::get<vpux::type::float8_e5m2>(ppeConverter(0, 0, val, scaleTableType));
        }

        return std::get<float>(ppeConverter(0, 0, val, scaleTableType));
    };

    SmallVector<int32_t> dwScales;
    dwScales.reserve(outQuantScales.size());
    std::transform(outQuantScales.begin(), outQuantScales.end(), std::back_inserter(dwScales), [&](auto scale) {
        // When the original Conv has new weight table format, the scale may be of several types: fp8(e52m/e4m3)/f32.
        // However, if DW.Conv cannot use new weights format, then we must use pass a weights table in the old format,
        // which does not support different scale types. As such, we convert the original scale values to fp32,
        // then bit_cast to int32, in preparation to be added to the old weights table.
        if (scaleTableType != nullptr) {
            return llvm::bit_cast<int32_t>(getF32Scale(scale));
        }

        const QuantizationApproximation scaleApproximation(scale);
        // Inputs of Dw.Conv will be f16
        return std::get<int32_t>(ppeConverter(checked_cast<uint8_t>(scaleApproximation.shift()),
                                              checked_cast<int16_t>(scaleApproximation.mult()), scale,
                                              mlir::Float16Type::get(ctx)));
    });
    return dwScales;
}

template <typename T,
          typename = std::enable_if_t<std::is_same_v<vpux::type::float8_e4m3, T> ||
                                      std::is_same_v<vpux::type::float8_e5m2, T> || std::is_same_v<float, T>>>
mlir::Value getScaleTableForDepthwiseOp(mlir::PatternRewriter& rewriter, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                        ArrayRef<double> outQuantScales, mlir::Type scaleTableType, ShapeRef tableShape,
                                        mlir::Location loc) {
    std::vector<T> dwScales;
    dwScales.reserve(outQuantScales.size());
    std::transform(outQuantScales.begin(), outQuantScales.end(), std::back_inserter(dwScales), [&](auto scale) {
        return std::get<T>(ppeConverter(0, 0, scale, scaleTableType));
    });
    return VPU::createNewWeightsTableTensor<T>(rewriter, loc, dwScales, tableShape, scaleTableType);
}

SmallVector<double> getNewScaleValues(mlir::Type inElemType, mlir::Type filterElemType, int64_t oCh) {
    // if float input & per channel scale quant weights => scale values are put in weights table
    // if quant input & quant weights => scale values are put in weights table
    auto inQuantScales = extractScalesOrDefault(inElemType, 1.0);
    auto weightsQuantScales = extractScalesOrDefault(filterElemType, 1.0);

    broadcast(inQuantScales, oCh);
    broadcast(weightsQuantScales, oCh);

    SmallVector<double> inputRescale;
    inputRescale.reserve(oCh);
    std::transform(inQuantScales.begin(), inQuantScales.end(), weightsQuantScales.begin(),
                   std::back_inserter(inputRescale), std::multiplies<double>());
    return inputRescale;
}

}  // namespace

mlir::Value vpux::VPU::splitNCEConvolutionOverIC(VPU::NCEConvolutionOp origOp, mlir::Value weightInput,
                                                 SmallVector<VPU::NCEConvolutionOp>& convOps,
                                                 SmallVector<VPU::NCEEltwiseOp>& addOps,
                                                 SmallVector<VPU::DequantizeOp>& dequantizeOps,
                                                 const OutputTiling& tiles, VPU::DequantizeOp weightDequantizeOp,
                                                 mlir::PatternRewriter& rewriter, Logger log) {
    auto arch = config::getArch(origOp.getOperation());
    auto ctx = rewriter.getContext();

    // Get the NCEConvolutionOp's input and kernel sizes
    const auto inputShape = getBoundedShape(origOp.getInput());
    const auto inputW = inputShape[Dims4D::Act::W];
    const auto inputH = inputShape[Dims4D::Act::H];
    const auto inputN = inputShape[Dims4D::Act::N];

    const auto kernelShape = getBoundedShape(origOp.getFilter());
    const auto kernelW = kernelShape[Dims4D::Filter::KX];
    const auto kernelH = kernelShape[Dims4D::Filter::KY];
    const auto kernelN = kernelShape[Dims4D::Filter::OC];

    auto filterType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    auto filterElemType = filterType.getElementType();

    auto weightTable = origOp.getWeightsTable();
    auto biasTable = origOp.getWeightTableBias();
    auto scaleTable = origOp.getWeightTableScale();
    const auto hasLegacyWT = weightTable != nullptr;

    auto origInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto inElemType = origInType.getElementType();
    auto origOutType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto outElemType = origOutType.getElementType();

    // The outputs of ConvOps are used as intermediate data between ConvOps and AddOps.
    // Convert them to fp16 to make allowance for both precision and compatibility with Eltwise.
    auto f16Type = mlir::Float16Type::get(ctx);
    const auto f16TypeOutputs = origOutType.changeElemType(f16Type);

    const auto inputRescale = getNewScaleValues(inElemType, filterElemType, kernelN);
    SmallVector<double> outQuantScales;
    outQuantScales.reserve(kernelN);

    // A stripped PPE is generated, ignoring clamps, post-op's and per-tensor scale/bias
    // It will be used for all ops, except the last op in the decomposition.
    // There we will add the true ppe attr, comprising of clamp, post op and
    // per tensor output scales.
    // The last op in the chain is either an Eltwise op (per tensor output scale or no scale) or
    // a DepthwiseConv op (per channel output scale).
    auto strippedPpeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    if (const auto clampAdapter = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterClamp*>()) {
        strippedPpeAttr = clampAdapter->discardClamp(strippedPpeAttr, f16Type);
    }

    // The original PPE attribute of the convolution (containing post-op and per-tensor scale/bias info), ends up in the
    // final op (Add/DWConv)
    auto eltwisePpeAttr = strippedPpeAttr;
    auto finalPpeAttr = origOp.getPpeAttr();

    // First Conv ppe attr must not clear the per tensor bias
    auto firstConvPpeAttr = strippedPpeAttr;
    const auto scaleAdapter = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterScaleBias*>();
    if (scaleAdapter != nullptr) {
        const auto oldScale = scaleAdapter->getScale(finalPpeAttr);
        const auto oldBias = scaleAdapter->getBias(finalPpeAttr);
        auto isPerTensorScale = !inputRescale.empty() && llvm::all_equal(inputRescale);
        if (oldScale.has_value()) {
            eltwisePpeAttr = scaleAdapter->updateScale(eltwisePpeAttr, {1.0});
            finalPpeAttr = scaleAdapter->updateScale(finalPpeAttr, {1.0});

            if (oldBias.has_value()) {
                finalPpeAttr = scaleAdapter->updateBias(finalPpeAttr, 0.0);
                eltwisePpeAttr = scaleAdapter->updateBias(eltwisePpeAttr, 0.0);
            }

            if (isPerTensorScale) {
                const auto perTensorScale = inputRescale.front();
                firstConvPpeAttr = scaleAdapter->updateScale(firstConvPpeAttr, {perTensorScale});
                strippedPpeAttr = scaleAdapter->updateScale(strippedPpeAttr, {perTensorScale});

                if (oldBias.has_value()) {
                    strippedPpeAttr = scaleAdapter->updateBias(strippedPpeAttr, 0.0);
                    firstConvPpeAttr = scaleAdapter->updateBias(firstConvPpeAttr, oldBias.value());
                }
            } else {
                // if, for some reason, the scale in origOp is per tensor, but for the new Convs it will be per channel,
                // then we must clear the scale and bias from Conv ops so that they take the values
                // from weights/scale/bias table
                firstConvPpeAttr = scaleAdapter->discardScaleBias(firstConvPpeAttr);
                strippedPpeAttr = scaleAdapter->discardScaleBias(strippedPpeAttr);
            }
        }
    }

    if (auto preluAlphaAdapter = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterFpPreluAlpha*>()) {
        firstConvPpeAttr = preluAlphaAdapter->updateFpPreluAlpha(firstConvPpeAttr, SmallVector<double>{1.0});
        strippedPpeAttr = preluAlphaAdapter->updateFpPreluAlpha(strippedPpeAttr, SmallVector<double>{1.0});
        eltwisePpeAttr = preluAlphaAdapter->updateFpPreluAlpha(eltwisePpeAttr, SmallVector<double>{1.0});

        if (auto intPPE = mlir::dyn_cast_or_null<VPU::PPEIntAttr>(origOp.getPpeAttr())) {
            const auto isFloatInput = !mlir::isa<mlir::quant::QuantizedType>(inElemType);
            if (isFloatInput && mlir::isa<mlir::quant::UniformQuantizedType>(filterElemType)) {
                const auto weightsQuantScales = extractScalesOrDefault(filterElemType, 1.0);
                auto preluAlpha = preluAlphaAdapter->getFpPreluAlpha(finalPpeAttr);
                std::transform(preluAlpha.begin(), preluAlpha.end(), preluAlpha.begin(), [&](double preluScale) {
                    return preluScale / weightsQuantScales.front();
                });

                finalPpeAttr = preluAlphaAdapter->updateFpPreluAlpha(finalPpeAttr, preluAlpha);
            }
        }
    }

    if (auto fpPPE = mlir::dyn_cast_or_null<VPU::PPEFpAttr>(origOp.getPpeAttr())) {
        if (mlir::isa_and_nonnull<mlir::quant::UniformQuantizedPerAxisType>(outElemType) ||
            mlir::isa_and_nonnull<mlir::quant::UniformQuantizedType>(outElemType)) {
            auto zeroAdder = getFPAttr(ctx, 0.0);
            // clear out adder for all but last op
            auto fpStrippedPPE = mlir::cast<VPU::PPEFpAttr>(strippedPpeAttr);
            strippedPpeAttr = PPEFpAttr::get(
                    ctx, fpStrippedPPE.getMode(), fpStrippedPPE.getClampLow(), fpStrippedPPE.getClampHigh(),
                    fpStrippedPPE.getScale(), fpStrippedPPE.getPreluAlpha(), fpStrippedPPE.getBias(), zeroAdder,
                    fpStrippedPPE.getIn1Mult(), fpStrippedPPE.getIn2Mult(), fpStrippedPPE.getSprlut());

            auto fpFirstConvPPE = mlir::cast<VPU::PPEFpAttr>(firstConvPpeAttr);
            firstConvPpeAttr = PPEFpAttr::get(
                    ctx, fpFirstConvPPE.getMode(), fpFirstConvPPE.getClampLow(), fpFirstConvPPE.getClampHigh(),
                    fpFirstConvPPE.getScale(), fpFirstConvPPE.getPreluAlpha(), fpFirstConvPPE.getBias(), zeroAdder,
                    fpFirstConvPPE.getIn1Mult(), fpFirstConvPPE.getIn2Mult(), fpFirstConvPPE.getSprlut());

            auto fpEltiwsePPE = mlir::cast<VPU::PPEFpAttr>(eltwisePpeAttr);
            eltwisePpeAttr = PPEFpAttr::get(
                    ctx, fpEltiwsePPE.getMode(), fpEltiwsePPE.getClampLow(), fpEltiwsePPE.getClampHigh(),
                    fpEltiwsePPE.getScale(), fpEltiwsePPE.getPreluAlpha(), fpEltiwsePPE.getBias(), zeroAdder,
                    fpEltiwsePPE.getIn1Mult(), fpEltiwsePPE.getIn2Mult(), fpEltiwsePPE.getSprlut());
        }
    }

    auto convScaleTable = scaleTable;
    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(arch, !hasLegacyWT);
    const auto scaleRetrieveConverter = VPU::NCESparsity::getScaleRetrieveCb(arch, !hasLegacyWT);
    std::vector<int32_t> weightTableVec;

    if (hasLegacyWT) {
        updateScaleInConvWeightTable(weightTable, weightTableVec, inputRescale, ppeConverter, scaleRetrieveConverter,
                                     origInType.getElementType(), outQuantScales, kernelN, log.nest());
    } else if (scaleTable != nullptr) {
        auto scaleTableType = mlir::cast<NDTypeInterface>(scaleTable.getType()).getElementType();
        if (mlir::isa<mlir::Float8E4M3FNType>(scaleTableType)) {
            convScaleTable = updateScaleTableForConvOps<vpux::type::float8_e4m3>(
                    rewriter, scaleTable, scaleTableType, inputRescale, ppeConverter, scaleRetrieveConverter,
                    outQuantScales, kernelN, origOp.getLoc(), log.nest());
        } else if (mlir::isa<mlir::Float8E5M2Type>(scaleTableType)) {
            convScaleTable = updateScaleTableForConvOps<vpux::type::float8_e5m2>(
                    rewriter, scaleTable, scaleTableType, inputRescale, ppeConverter, scaleRetrieveConverter,
                    outQuantScales, kernelN, origOp.getLoc(), log.nest());
        } else {
            convScaleTable = updateScaleTableForConvOps<float>(rewriter, scaleTable, scaleTableType, inputRescale,
                                                               ppeConverter, scaleRetrieveConverter, outQuantScales,
                                                               kernelN, origOp.getLoc(), log.nest());
        }
    }

    mlir::Value zeroFilledBiasTable = nullptr;
    if (biasTable != nullptr) {
        const auto tablesShape = VPU::NCESparsity::inferWeightsTableShape(kernelN, /*newFormat=*/true);
        const auto zeroBias = std::vector<float>(tablesShape.totalSize(), 0.f);
        zeroFilledBiasTable = VPU::createNewWeightsTableTensor<float>(rewriter, origOp->getLoc(), zeroBias, tablesShape,
                                                                      rewriter.getF32Type());
    }

    for (size_t tile = 0; tile < tiles.size(); tile++) {
        auto offsetIC = tiles[tile].offsets[Dims4D::Act::C];
        auto sizeIC = tiles[tile].shape[Dims4D::Act::C];
        log.trace("Slicing channels {0} - {1}", offsetIC, sizeIC);

        // Slice inputs
        const auto sliceOffsets = SmallVector<int64_t>{0, offsetIC, 0, 0};
        const auto inSliceShape = SmallVector<int64_t>{inputN, sizeIC, inputH, inputW};
        auto convInput = rewriter.create<VPU::SliceOp>(appendLoc(origOp->getLoc(), "_in_slice_{0}", sliceOffsets),
                                                       origOp.getInput(), getIntArrayAttr(rewriter, sliceOffsets),
                                                       getIntArrayAttr(rewriter, inSliceShape));

        // Slice kernels
        const auto kernelSliceShape = SmallVector<int64_t>{kernelN, sizeIC, kernelH, kernelW};
        const auto rawKernelSliceShape = getIntArrayAttr(rewriter, kernelSliceShape);
        auto weightSlice = rewriter.create<VPU::SliceOp>(appendLoc(origOp->getLoc(), "_w_slice_{0}", sliceOffsets),
                                                         weightInput, getIntArrayAttr(rewriter, sliceOffsets),
                                                         getIntArrayAttr(rewriter, kernelSliceShape));
        auto weightSliceResult = weightSlice.getResult();
        if (weightDequantizeOp != nullptr) {
            // Slice over VPU.DequantizeOp
            // TODO: This logic may also be practicable to other element-wise operations, E#178703
            // input1            input2
            //                     |
            //    \             VPU.DequantizeOp
            //                     /
            //    VPU.NCE.Convolution

            // To:

            //  input1                input2
            //      |                   |
            // VPU.SlicexN          VPU.SlicexN
            //                          |
            //       \          VPU.DequantizeOpxN
            //                          /
            //         VPU.NCE.ConvolutionxN
            //                  |
            //          VPU.NCE.Eltwise.ADDx(N-1)

            auto dequantizeSlice = rewriter.create<VPU::DequantizeOp>(weightDequantizeOp->getLoc(), weightSliceResult,
                                                                      weightDequantizeOp.getDstElemTypeAttr(),
                                                                      weightDequantizeOp.getMultiClusterStrategyAttr());
            dequantizeOps.push_back(dequantizeSlice);
            weightSliceResult = dequantizeSlice.getResult();
        }

        const auto noOfBits = vpux::getElemTypeSize(filterElemType);
        const auto weightSetSize = alignMemSize(kernelH * kernelW * sizeIC * noOfBits,
                                                Byte(VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT))
                                           .to<Byte>()
                                           .count();
        const auto sparsitySetSize =
                alignValUp(divUp(kernelH * kernelW * sizeIC, CHAR_BIT * getValuesPerSparsityBit(filterElemType)),
                           static_cast<int64_t>(VPU::NCEInvariant::VPU_WEIGHT_SET_BYTE_ALIGNMENT));
        if (hasLegacyWT) {
            weightTable = updateDataPtrSpPtrAndBiasInConvWeightTable(rewriter, weightTableVec, weightSetSize,
                                                                     sparsitySetSize, kernelN, tile, inElemType, arch,
                                                                     origOp.getLoc(), log.nest());
        }

        // Apply bias for the first convolution only
        // Set the bias values to 0, if bias exists
        mlir::Value convBiasTable = (biasTable != nullptr && tile != 0) ? zeroFilledBiasTable : biasTable;
        auto convPpeAttr = tile == 0 ? firstConvPpeAttr : strippedPpeAttr;
        auto convOp = rewriter.create<VPU::NCEConvolutionOp>(
                appendLoc(origOp->getLoc(), "_ic_tile_{0}"), f16TypeOutputs, convInput.getResult(), weightSliceResult,
                weightTable, origOp.getWeightTableDataPtr(), origOp.getWeightTableSpPtr(), convScaleTable,
                convBiasTable, origOp.getWeightZeroPoints(), origOp.getStrides(), origOp.getPad(), convPpeAttr,
                origOp.getMpeEngineAttr(), rawKernelSliceShape, origOp.getMultiClusterStrategyAttr(),
                origOp.getOutputPaddingAttr(),
                /*inputPadding=*/nullptr);
        convOps.push_back(convOp);
        log.trace("Created new conv.");
    }

    // Add the outputs of the convolutions with NCEEltwise Add operations. This is needed because NCEConvolutionOp
    // accumulates all its input channels into 1 output channel. Splitting the Convolutions into smaller Convolutions,
    // the outputs have to be added together.

    const auto opType = VPU::EltwiseType::ADD;
    VPU::NCEEltwiseOp addResult = nullptr;

    // Assumption: Unless the output type has per channel quantization scales, there is no way for the output scale
    // to be per channel. The scale is computed as:
    // input_quant_scale * weights_quant_scale  * static_scale / output_quant_scale
    // - input_quant_scale * weights_quant_scale is added to the decomposed Conv's weights table
    // - static_scale is per tensor
    // - output_quant_scale is the only one that can be per channel and that happens when outElemType is
    //   mlir::quant::UniformQuantizedPerAxisType
    const bool hasPerTensorOrNoOutputScales = !mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(outElemType);
    for (size_t index = 0; index < convOps.size() - 1; index++) {
        const auto addOperand = index == 0 ? convOps[index].getOutput() : addResult.getOutput();

        // NCEEltwise inType and outType are always same with convOp outType
        // TODO: check how in-place works here, E#178731
        auto eltwiseOutputType =
                (index == convOps.size() - 2 && hasPerTensorOrNoOutputScales) ? origOutType : f16TypeOutputs;
        auto ppeAttr = (index == convOps.size() - 2 && hasPerTensorOrNoOutputScales) ? finalPpeAttr : eltwisePpeAttr;
        auto outputPadding = origOp.getOutputPaddingAttr();
        addResult = rewriter.create<VPU::NCEEltwiseOp>(appendLoc(origOp->getLoc(), "_accumulator_{0}", index),
                                                       eltwiseOutputType, addOperand, convOps[index + 1].getOutput(),
                                                       opType, ppeAttr, /*multicluster_strategy_attr=*/nullptr,
                                                       /*in_place=*/nullptr, outputPadding, outputPadding);

        // change NCEConv's output layout to supported NCEEltwise input layout
        // Eg: if NCEConv (inL=NHWC,outL=NCHW) splits into 3 small NCEConv:
        //   NCEConv (inL=NHWC,out=NHWC)    NCEConv (inL=NHWC,out=NHWC)     NCEConv (inL=NHWC,out=NHWC)
        //              \                         /                                     /
        //               NCEElt (inL=NHWC,out=NHWC)                                    /
        //                             \                                              /
        //                                         NCEElt (inL=NHWC,out=NCHW)
        if (auto iface = mlir::dyn_cast<IE::LayoutInfoOpInterface>(addResult.getOperation())) {
            auto orderInfo = iface.getLayoutInfo();
            iface.inferLayoutInfo(orderInfo, /*seOpsEnabled=*/false, /*seExperimentalOpsEnabled=*/false);
            const auto supportOrder1 = orderInfo.getInput(0);
            const auto supportOrder2 = orderInfo.getInput(1);
            const auto inputOrder1 = DimsOrder::fromValue(addResult.getInput1());
            const auto inputOrder2 = DimsOrder::fromValue(addResult.getInput2());

            if (supportOrder1 != inputOrder1 && supportOrder2 != inputOrder2) {
                const auto newInput1Type = mlir::dyn_cast<vpux::NDTypeInterface>(addResult.getInput1().getType())
                                                   .changeDimsOrder(supportOrder1);
                const auto newInput2Type = mlir::dyn_cast<vpux::NDTypeInterface>(addResult.getInput2().getType())
                                                   .changeDimsOrder(supportOrder2);

                auto input1Op = addResult.getInput1().getDefiningOp();
                auto input2Op = addResult.getInput2().getDefiningOp();
                input1Op->getResult(0).setType(newInput1Type);
                input2Op->getResult(0).setType(newInput2Type);

                addResult.getOperation()->setOperands({input1Op->getResult(0), input2Op->getResult(0)});
            }
        }
        addOps.push_back(addResult);
    }

    // No final output scales to apply. Return result of last Eltwise op
    if (outQuantScales.empty() || llvm::all_of(outQuantScales, [](double val) {
            return isDoubleEqual(val, 1.0);
        })) {
        return addResult.getOutput();
    }

    // If we have a per channel output quantization scale, we cannot add it to the last eltwise op due to the lack
    // of weights table support We need a DW.Conv task to apply it.
    if (!hasPerTensorOrNoOutputScales) {
        // Construct dummy weights for DW.Conv, all filled with 1s
        const auto weightsShape = Shape{kernelN, 1, 1, 1};
        auto dwWeights = vpux::buildDwWeights(appendLoc(origOp->getLoc(), "_dummy_weights"),
                                              weightsShape[Dims4D::Filter::OC], f16Type, rewriter);
        auto alignedWeights = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), dwWeights);

        mlir::Value dwWeightTable = nullptr;
        mlir::Value dwScaleTable = nullptr;
        mlir::Value dwBiasTable = nullptr;

        const auto dwSupportsNewWeightTableFormat =
                VPU::MPEEngineConfig::isNewWeightTableFormatSupportedWithDwOps(arch);
        if (!dwSupportsNewWeightTableFormat) {
            const auto alignedShapeOfWeights = getShape(alignedWeights);
            const auto noOfBits = vpux::getElemTypeSize(dwWeights.getType());
            const auto weightSetNumElems = alignedShapeOfWeights[Dims4D::Filter::IC] *
                                           alignedShapeOfWeights[Dims4D::Filter::KY] *
                                           alignedShapeOfWeights[Dims4D::Filter::KX];
            const auto weightSetSize = (weightSetNumElems * noOfBits).to<Byte>().count();

            const auto outputScaleVals = getScaleValuesTableForLegacyWTDepthwiseOp(
                    ppeConverter, outQuantScales, scaleTable != nullptr ? scaleTable.getType() : nullptr, ctx);

            std::vector<int32_t> weightTableContent(
                    VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * alignedShapeOfWeights[Dims4D::Filter::OC], 0);
            const auto wtShape = VPU::NCESparsity::inferWeightsTableShape(alignedShapeOfWeights[Dims4D::Filter::OC]);

            constexpr size_t weightsOffset = 0;
            constexpr size_t scaleOffset = 2;

            for (int64_t i = 0; i < alignedShapeOfWeights[Dims4D::Filter::OC]; ++i) {
                auto index = VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * i;

                weightTableContent[index + weightsOffset] = static_cast<int32_t>(i * weightSetSize);
                weightTableContent[index + scaleOffset] = outputScaleVals[i];
            }

            dwWeightTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightTableContent, wtShape);
        } else {
            const auto tablesShape = VPU::NCESparsity::inferWeightsTableShape(kernelN, /*newFormat=*/true);

            auto scaleTableType = scaleTable != nullptr
                                          ? mlir::cast<NDTypeInterface>(scaleTable.getType()).getElementType()
                                          : rewriter.getF32Type();

            if (mlir::isa<mlir::Float8E4M3FNType>(scaleTableType)) {
                dwScaleTable = getScaleTableForDepthwiseOp<vpux::type::float8_e4m3>(
                        rewriter, ppeConverter, outQuantScales, scaleTableType, tablesShape, origOp->getLoc());
            } else if (mlir::isa<mlir::Float8E5M2Type>(scaleTableType)) {
                dwScaleTable = getScaleTableForDepthwiseOp<vpux::type::float8_e5m2>(
                        rewriter, ppeConverter, outQuantScales, scaleTableType, tablesShape, origOp->getLoc());
            } else {
                dwScaleTable = getScaleTableForDepthwiseOp<float>(rewriter, ppeConverter, outQuantScales,
                                                                  scaleTableType, tablesShape, origOp->getLoc());
            }

            if (zeroFilledBiasTable == nullptr) {
                const auto zeroBias = std::vector<float>(tablesShape.totalSize(), 0.f);
                zeroFilledBiasTable = VPU::createNewWeightsTableTensor<float>(rewriter, origOp->getLoc(), zeroBias,
                                                                              tablesShape, rewriter.getF32Type());
            }

            dwBiasTable = zeroFilledBiasTable;
        }

        if (scaleAdapter != nullptr) {
            // since we need per channel scale values, ensure that the op will pick up the
            // values in scale table as oposed to reading it from PPE
            finalPpeAttr = scaleAdapter->discardScaleBias(finalPpeAttr);
        }

        auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        auto padding = VPU::getPaddingAttr(ctx, PadInfo(0, 0, 0, 0));
        auto nceDepthConvolutionOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
                appendLoc(origOp->getLoc(), "_dw_conv_out_scale"), origOutType, addResult.getOutput(), alignedWeights,
                dwWeightTable, /*data_ptr_table=*/nullptr, /*sparsity_ptr_table=*/nullptr, dwScaleTable, dwBiasTable,
                /*zp_table=*/nullptr, strides, padding, finalPpeAttr, getIntArrayAttr(rewriter, weightsShape.raw()),
                /*multiClusterStrategyAttr=*/nullptr, origOp.getOutputPaddingAttr(), nullptr);

        return nceDepthConvolutionOp.getOutput();
    }

    // If we have output quantization scale and it is per tensor, adapt the PPE scale of the last Eltwise with
    // (output_quant_scale * static_scale) value
    if (const auto ppeAdapter = VPU::PpeVersionConfig::getFactoryAs<VPU::IPpeAdapterScaleBias*>()) {
        const auto outputScale = outQuantScales.front();
        if (mlir::isa<vpux::VPU::PPEFpAttr>(finalPpeAttr)) {
            finalPpeAttr = ppeAdapter->updateScale(finalPpeAttr, SmallVector<double>{outputScale});
        } else if (auto finalIntPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(finalPpeAttr)) {
            auto quantScale = finalIntPpeAttr.getQuantScale() == nullptr
                                      ? mlir::SmallVector<double>{1.0}
                                      : parseFPArrayAttr<double>(finalIntPpeAttr.getQuantScale());

            std::transform(quantScale.begin(), quantScale.end(), quantScale.begin(), [&](auto val) {
                return val * outputScale;
            });

            finalPpeAttr = PPEIntAttr::get(ctx, finalIntPpeAttr.getMode(), finalIntPpeAttr.getClampLow(),
                                           finalIntPpeAttr.getClampHigh(), finalIntPpeAttr.getLreluMult(),
                                           finalIntPpeAttr.getLreluShift(), getFPArrayAttr(ctx, quantScale),
                                           finalIntPpeAttr.getQuantMult(), finalIntPpeAttr.getQuantShift(),
                                           finalIntPpeAttr.getQuantPostShift(), finalIntPpeAttr.getIn1QuantMult(),
                                           finalIntPpeAttr.getIn2QuantMult(), finalIntPpeAttr.getFpPreluAlpha());
        }

        addResult.setPpeAttr(finalPpeAttr);
    }

    return addResult.getOutput();
}
