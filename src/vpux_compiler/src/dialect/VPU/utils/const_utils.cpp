//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include <vpux/utils/core/numeric.hpp>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights,
                                            const Const::ContentAttr& bias, int64_t OC,
                                            VPU::NCESparsity::PPEConverterCb ppeConverter,
                                            VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale,
                                            bool hasAutopad) {
    bool is5DShape = vpux::getShape(weights).size() == 5;

    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep =
            is5DShape ? VPU::NCESparsity::get5DWeightPtrStep(weights) : VPU::NCESparsity::getWeightPtrStep(weights);
    const auto sparsityPtrStep = 0;

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(opInput.getType()).getElementType();
    const auto weightsElemType =
            weights ? mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType() : nullptr;

    const auto wtVec = VPU::NCESparsity::getWeightsTable(inElemType, opOutputElemType, weightPtrOffset, weightPtrStep,
                                                         sparsityPtrOffset, sparsityPtrStep, ppeConverter,
                                                         biasConverter, OC, weightsElemType, bias, constScale);

    if (hasAutopad) {
        return VPU::NCESparsity::getExpandedWeightsTable(wtVec, OC);
    }

    return wtVec;
}

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                            const Const::ContentAttr& bias, int64_t OC,
                                            VPU::NCESparsity::PPEConverterCb ppeConverter,
                                            VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale,
                                            bool hasAutopad) {
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();
    return createWeightsTableData(opInput, outElemType, weights, bias, OC, ppeConverter, biasConverter, constScale,
                                  hasAutopad);
}

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable,
                                     vpux::ShapeRef weightsTableShape) {
    const auto elemType = getSInt32Type(builder.getContext());

    const auto dataStorageType = mlir::RankedTensorType::get(weightsTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, weightsTable);
}

std::vector<float> createBiasTableData(mlir::Value opInput, mlir::Type outElemType, mlir::Value weights,
                                       const Const::ContentAttr& bias, int64_t OC,
                                       VPU::NCESparsity::BiasConverterCb biasConverter) {
    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(opInput.getType()).getElementType();
    const auto weightsElemType =
            weights ? mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType() : nullptr;

    return VPU::NCESparsity::getBiasTable(inElemType, outElemType, biasConverter, OC, weightsElemType, bias);
}

std::vector<float> createBiasTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                       const Const::ContentAttr& bias, int64_t OC,
                                       VPU::NCESparsity::BiasConverterCb biasConverter) {
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();

    return createBiasTableData(opInput, outElemType, weights, bias, OC, biasConverter);
}

NewWeightsTableData::NewWeightsTableData(bool useNewWeightTableFormat, mlir::Value opInput, mlir::Type opOutputElemType,
                                         mlir::Value weights, const Const::ContentAttr& bias, int64_t OC,
                                         VPU::NCESparsity::PPEConverterCb ppeConverter,
                                         VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale) {
    initializeData(useNewWeightTableFormat, opInput, opOutputElemType, weights, bias, OC, ppeConverter, biasConverter,
                   constScale);
}

NewWeightsTableData::NewWeightsTableData(bool useNewWeightTableFormat, mlir::Value opInput, mlir::Value opOutput,
                                         mlir::Value weights, const Const::ContentAttr& bias, int64_t OC,
                                         VPU::NCESparsity::PPEConverterCb ppeConverter,
                                         VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale) {
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();

    initializeData(useNewWeightTableFormat, opInput, outElemType, weights, bias, OC, ppeConverter, biasConverter,
                   constScale);
}

void NewWeightsTableData::initializeData(bool useNewWeightTableFormat, mlir::Value opInput, mlir::Type opOutputElemType,
                                         mlir::Value weights, const Const::ContentAttr& bias, int64_t OC,
                                         VPU::NCESparsity::PPEConverterCb ppeConverter,
                                         VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale) {
    // leave vectors empty for archs using the legacy weights table format
    if (useNewWeightTableFormat) {
        scaleData = createScaleTableData<float>(opInput, opOutputElemType, weights, OC, ppeConverter, constScale);
        if (bias != nullptr) {
            biasData = createBiasTableData(opInput, opOutputElemType, weights, bias, OC, biasConverter);
        } else {
            biasData = std::vector<float>(OC, 0.0);
        }
    }
}

NewWeightsTableTensors::NewWeightsTableTensors(bool useNewWeightTableFormat, mlir::OpBuilder& builder,
                                               mlir::Location loc, mlir::Value opInput, mlir::Type opOutputElemType,
                                               mlir::Value weights, const Const::ContentAttr& bias,
                                               ShapeRef weightTableShape, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                               VPU::NCESparsity::BiasConverterCb biasConverter,
                                               mlir::FloatAttr constScale) {
    initializeTensors(useNewWeightTableFormat, builder, loc, opInput, opOutputElemType, weights, bias, weightTableShape,
                      ppeConverter, biasConverter, constScale);
}

NewWeightsTableTensors::NewWeightsTableTensors(bool useNewWeightTableFormat, mlir::OpBuilder& builder,
                                               mlir::Location loc, mlir::Value opInput, mlir::Value opOutput,
                                               mlir::Value weights, const Const::ContentAttr& bias,
                                               ShapeRef weightTableShape, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                               VPU::NCESparsity::BiasConverterCb biasConverter,
                                               mlir::FloatAttr constScale) {
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();

    initializeTensors(useNewWeightTableFormat, builder, loc, opInput, outElemType, weights, bias, weightTableShape,
                      ppeConverter, biasConverter, constScale);
}

void NewWeightsTableTensors::initializeTensors(bool useNewWeightTableFormat, mlir::OpBuilder& builder,
                                               mlir::Location loc, mlir::Value opInput, mlir::Type opOutputElemType,
                                               mlir::Value weights, const Const::ContentAttr& bias,
                                               ShapeRef weightTableShape, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                               VPU::NCESparsity::BiasConverterCb biasConverter,
                                               mlir::FloatAttr constScale) {
    const auto newWeightsTableData =
            NewWeightsTableData(useNewWeightTableFormat, opInput, opOutputElemType, weights, bias,
                                weightTableShape.totalSize(), ppeConverter, biasConverter, constScale);

    scaleTensor = initializeScaleBiasTensor(builder, loc, newWeightsTableData.scaleData, weightTableShape);
    biasTensor = initializeScaleBiasTensor(builder, loc, newWeightsTableData.biasData, weightTableShape);
}

mlir::Value NewWeightsTableTensors::initializeScaleBiasTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                              ArrayRef<float> tableData, ShapeRef weightTableShape) {
    return tableData.empty() ? nullptr
                             : createNewWeightsTableTensor<float>(builder, loc, tableData, weightTableShape,
                                                                  builder.getF32Type());
}

namespace {

mlir::Value getAlignedConstWeights(mlir::OpBuilder& builder, mlir::Location loc, Const::DeclareOp weightsConst,
                                   ShapeRef flatWeightShape, int64_t padding) {
    const auto& weightsContentAttr = weightsConst.getContentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.transform().reorder(DimsOrder::NCHW).get();

    auto flatWeightsContentAttr = nchwWeightsContentAttr.transform().reshape(flatWeightShape).get();
    auto alignedWeightsContentAttr =
            flatWeightsContentAttr.transform().padWithZero({0, 0, 0, 0}, {0, padding, 0, 0}).get();
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.transform().reorder(DimsOrder::NHWC).get();

    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(weightsConst.getOutput().getType());
    const auto outAllocType = mlir::cast<vpux::NDTypeInterface>(
            mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType()));
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, std::move(nhwcWeightsContentAttr));

    return alignedWeightsOp.getOutput();
}

Const::ContentAttr buildPadData(const mlir::Type type, ArrayRef<int64_t> shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "Unsupported shape size {0}", shape.size());
    const auto OC = shape[Dims4D::Filter::OC.ind()];

    if (const auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        const auto padType = mlir::RankedTensorType::get(shape, normalizeQuantStorageType(quantizedType));
        uint8_t padValueUint8 = 0;

        if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(quantizedType)) {
            padValueUint8 = static_cast<uint8_t>(uniformType.getZeroPoint());
        } else if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(quantizedType)) {
            const auto zeroPoints = perAxisType.getZeroPoints();
            VPUX_THROW_UNLESS(checked_cast<size_t>(OC) == zeroPoints.size(),
                              "Number of zero points {0} and channels {1} don't match", zeroPoints.size(), OC);

            // assuming all zero points are equal to broadcast
            VPUX_THROW_UNLESS(
                    zeroPoints.size() == 1 || std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin()),
                    "All zero points should be equal");
            padValueUint8 = static_cast<uint8_t>(zeroPoints.front());
        } else {
            VPUX_THROW("Unsupported Quantized Type '{0}'", quantizedType);
        }
        const auto padAttr = Const::createConstContent(padType, ArrayRef(padValueUint8));

        return Const::ContentAttr::get(padAttr, Const::ContentSetup(padType).castElemType(quantizedType));
    } else {
        const auto ndType = mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(shape, type));
        const auto padType = mlir::cast<mlir::RankedTensorType>(ndType.changeDimsOrder(DimsOrder::NCHW));
        const auto padAttr = Const::createConstContent(padType, ArrayRef(vpux::type::float16(0.f)));

        return Const::ContentAttr::get(padAttr);
    }
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      ShapeRef flatWeightShape, int64_t padding) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(origFilter.getType());
    const auto origOrder = origFilterType.getDimsOrder();
    const auto flatWeightType = origFilterType.changeShape(flatWeightShape).changeDimsOrder(origOrder);
    auto flatWeightsOp =
            builder.create<IE::ShapeCastOp>(loc, flatWeightType, origFilter, getIntArrayAttr(ctx, flatWeightShape));

    // Step 2: Permute flat input to NCHW.
    auto flatWeightTypeNCHWType = flatWeightType.changeDimsOrder(DimsOrder::NCHW);
    const auto nchwAttr = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    const auto flatWeightsDimsAttr =
            mlir::AffineMapAttr::get(getPermutationFromOrders(origOrder, DimsOrder::NCHW, ctx));
    auto flatWeightsNCHW = builder.create<IE::PermuteCastOp>(loc, flatWeightTypeNCHWType, flatWeightsOp->getResult(0),
                                                             nchwAttr, flatWeightsDimsAttr);

    // Step 3: Create padding for flat NCHW input. IC must be a multiple of 16.
    const auto OC = flatWeightShape[Dims4D::Filter::OC];
    const auto flatWeightChannelsCount = flatWeightShape[Dims4D::Filter::IC];
    const auto alignedWeightShape = SmallVector<int64_t>{OC, flatWeightChannelsCount + padding, 1, 1};
    const auto outShapedType = mlir::cast<vpux::NDTypeInterface>(
            mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType()));
    const auto outAllocType = outShapedType.changeDimsOrder(DimsOrder::NHWC);

    const auto padShape = SmallVector<int64_t>{OC, padding, 1, 1};
    auto padContentAttr = buildPadData(origFilterType.getElementType(), padShape);

    const auto padAllocType =
            mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(padShape, origFilterType.getElementType()));
    const auto padAllocTypeNHWC = padAllocType.changeDimsOrder(DimsOrder::NCHW);
    auto paddedTensor = builder.create<Const::DeclareOp>(loc, padAllocTypeNHWC, std::move(padContentAttr));

    // Step 4: Concatenate flat NCHW input with padding.

    auto concatViewOp =
            builder.create<IE::ConcatOp>(loc, SmallVector<mlir::Value>{flatWeightsNCHW, paddedTensor}, Dims4D::Act::C);

    // Step 5: Permute the result to NHWC.
    const auto nhwcAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(DimsOrder::NCHW, DimsOrder::NHWC, ctx));

    auto outOpNCHW =
            builder.create<IE::PermuteCastOp>(loc, outAllocType, concatViewOp.getOutput(), nhwcAttr, memPermAttr);

    return outOpNCHW.getOutput();
}

}  // namespace

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(origFilter.getType());
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (filtersPerInChan * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto padding = alignment - remainder;

    const auto flatWeightChannelsCount = filtersPerInChan * KY * KX;
    const auto flatWeightShape = Shape{OC, flatWeightChannelsCount, 1, 1};

    if (auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>()) {
        return getAlignedConstWeights(builder, loc, weightsConst, flatWeightShape, padding);
    } else {
        return getAlignedNonConstWeights(builder, loc, origFilter, flatWeightShape, padding);
    }
}

mlir::Value alignConvWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    const auto origFilterType = mlir::cast<vpux::NDTypeInterface>(origFilter.getType());
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (IC * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto flatWeightShape = Shape{OC, 1, 1, IC * KY * KX};
    const auto padding = alignment - remainder;

    if (mlir::isa<mlir::BlockArgument>(origFilter)) {
        auto reshape =
                builder.create<VPU::ReshapeOp>(loc, origFilter,
                                               /*shape=*/nullptr,
                                               /*special_zero=*/false, getIntArrayAttr(builder, flatWeightShape));

        auto padBeginAttr = getIntArrayAttr(builder, Shape{{0, 0, 0, 0}});
        auto padEndAttr = getIntArrayAttr(builder, Shape{{0, 0, 0, padding}});
        auto expandOp = builder.create<VPU::ExpandOp>(loc, reshape.getOutput(), padBeginAttr, padEndAttr);
        auto layoutCast = builder.create<VPU::LayoutCastOp>(loc, expandOp.getOutput(),
                                                            DimsOrder::NHWC.toAffineMap(origFilter.getContext()));
        return layoutCast.getOutput();
    }

    auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(weightsConst != nullptr, "Convolution does not provide constant weights");

    auto alignedWeightsContentAttr = weightsConst.getContentAttr()
                                             .transform()
                                             .reshape(flatWeightShape)
                                             .padWithZero({0, 0, 0, 0}, {0, 0, 0, padding})
                                             .get();

    const auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, IC * KY * KX + padding};
    const auto outAllocType = mlir::cast<vpux::NDTypeInterface>(
            mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType()));
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);

    auto alignedWeightsOp =
            builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, std::move(alignedWeightsContentAttr));
    return alignedWeightsOp.getOutput();
}

Byte calculateAlignedBuffersMemoryRequirement(config::ArchKind arch, SmallVector<Byte>& bufferSizes) {
    Byte offsetAlignment = Byte(vpux::DEFAULT_CMX_ALIGNMENT);
    Byte sizeAlignment = Byte(1);
    if (arch == config::ArchKind::NPU37XX || arch == config::ArchKind::NPU40XX) {
        offsetAlignment = Byte(getAddressAlignmentForSwizzling(SWIZZLING_KEY_5, arch));
        sizeAlignment = Byte(vpux::getSizeAlignmentForSwizzling(arch));
    }
    return vpux::calculateAlignedBuffersMemoryRequirement(bufferSizes, offsetAlignment, sizeAlignment);
}

bool isNullOrConstWithSingleValue(mlir::Value value) {
    if (value == nullptr) {
        return true;
    }

    auto declareOp = mlir::dyn_cast_or_null<Const::DeclareOp>(value.getDefiningOp());
    if (declareOp == nullptr) {
        return false;
    }

    return declareOp.getContentAttr().isSplat();
}

vpux::TensorAttr createTensorAttrFromType(vpux::NDTypeInterface inType, mlir::MLIRContext* ctx) {
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(inType)) {
        return getTensorAttr(inType.getContext(), inType.getDimsOrder().toAffineMap(ctx), inType.getMemSpace(),
                             boundedType.getBounds());
    }

    return getTensorAttr(inType.getContext(), inType.getDimsOrder().toAffineMap(ctx), inType.getMemSpace());
}

mlir::FailureOr<SmallVector<int64_t>> extractConstData(mlir::Location loc, mlir::Value value) {
    if (value == nullptr) {
        return errorAt(loc, "Target shape was not provided");
    }

    while (auto parentOp = value.getDefiningOp<VPU::CopyOp>()) {
        value = parentOp->getOperand(0);
    }

    auto valueConst = value.getDefiningOp<Const::DeclareOp>();
    if (valueConst == nullptr) {
        return mlir::failure();
    }

    const auto valueContent = valueConst.getContent();
    return to_small_vector(valueContent.getValues<int64_t>());
}

}  // namespace VPU
}  // namespace vpux
