//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include <mlir/IR/Value.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/custom_pwl_table.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <vpux/utils/core/numeric.hpp>

namespace vpux {
namespace VPU {

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights,
                                            const Const::ContentAttr& bias, int64_t OC,
                                            VPU::NCESparsity::PPEConverterCb ppeConverter,
                                            VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale,
                                            bool hasAutopad) {
    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = VPU::NCESparsity::getWeightPtrStep(weights);
    const auto sparsityPtrStep = 0;

    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

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
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    return createWeightsTableData(opInput, outElemType, weights, bias, OC, ppeConverter, biasConverter, constScale,
                                  hasAutopad);
}

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable) {
    const int64_t OC = weightsTable.size() / VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;

    const auto elemType = getSInt32Type(builder.getContext());
    const auto weightTableShape = NCESparsity::inferWeightsTableShape(OC);

    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, weightsTable);
}

std::vector<int32_t> createDataPointerTableData(mlir::Value opInput, mlir::Value opOutput,
                                                ArrayRef<int32_t> workloadsSizes, mlir::Value weights,
                                                int32_t weightPtrOffset, int64_t OC, ArrayRef<uint8_t> zeroPoints) {
    const auto weightPtrStep = VPU::NCESparsity::getWeightPtrStep(weights);

    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();

    return VPU::NCESparsity::getDataPointerTable(inElemType, outElemType, workloadsSizes, weightPtrOffset,
                                                 weightPtrStep, OC, zeroPoints);
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> createSparseDataPointerTableDataPair(
        mlir::Value opInput, mlir::Value opOutput, ArrayRef<int32_t> workloadsSizes, mlir::Value weights,
        int32_t weightPtrOffset, int32_t sparsityPtrOffset, ArrayRef<uint8_t> sparsityArray, int64_t OC,
        ArrayRef<uint8_t> zeroPoints) {
    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    const auto weightsShape = getShape(weights);

    return VPU::NCESparsity::getSparseDataPointerTablePair(inElemType, outElemType, workloadsSizes, weightPtrOffset,
                                                           weightsShape, sparsityPtrOffset, sparsityArray, OC,
                                                           weightsElemType, zeroPoints);
}

std::vector<float> createBiasTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                       const Const::ContentAttr& bias, int64_t OC,
                                       VPU::NCESparsity::BiasConverterCb biasConverter) {
    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    return VPU::NCESparsity::getBiasTable(inElemType, outElemType, biasConverter, OC, weightsElemType, bias);
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
    const auto origFilterType = weightsConst.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto outAllocType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                      .cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, std::move(nhwcWeightsContentAttr));

    return alignedWeightsOp.getOutput();
}

Const::ContentAttr buildPadData(const mlir::Type type, ArrayRef<int64_t> shape) {
    VPUX_THROW_UNLESS(shape.size() == 4, "Unsupported shape size {0}", shape.size());
    const auto OC = shape[Dims4D::Filter::OC.ind()];

    if (const auto quantizedType = type.dyn_cast<mlir::quant::QuantizedType>()) {
        const auto padType = mlir::RankedTensorType::get(shape, normalizeQuantStorageType(quantizedType));
        uint8_t padValueUint8 = 0;

        if (const auto uniformType = quantizedType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
            padValueUint8 = static_cast<uint8_t>(uniformType.getZeroPoint());
        } else if (const auto perAxisType = quantizedType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
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
        const auto ndType = mlir::RankedTensorType::get(shape, type).cast<vpux::NDTypeInterface>();
        const auto padType = ndType.changeDimsOrder(DimsOrder::NCHW).cast<mlir::RankedTensorType>();
        const auto padAttr = Const::createConstContent(padType, ArrayRef(vpux::type::float16(0.f)));

        return Const::ContentAttr::get(padAttr);
    }
}

mlir::Value getAlignedNonConstWeights(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter,
                                      ShapeRef flatWeightShape, int64_t padding) {
    auto ctx = builder.getContext();
    // Step 1: Flatten input to OCxICx1x1, where IC = filters * KY * KX.
    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
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
    const auto outShapedType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                       .cast<vpux::NDTypeInterface>();
    const auto outAllocType = outShapedType.changeDimsOrder(DimsOrder::NHWC);

    const auto padShape = SmallVector<int64_t>{OC, padding, 1, 1};
    auto padContentAttr = buildPadData(origFilterType.getElementType(), padShape);

    const auto padAllocType =
            mlir::RankedTensorType::get(padShape, origFilterType.getElementType()).cast<vpux::NDTypeInterface>();
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

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
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

    const auto origFilterType = origFilter.getType().cast<vpux::NDTypeInterface>();
    const auto alignment = VPU::NCEInvariant::getAlignment(origFilterType.getElementType());

    const auto remainder = (IC * KY * KX) % alignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);

    if (remainder == 0) {
        return origFilter;
    }

    const auto flatWeightShape = Shape{OC, 1, 1, IC * KY * KX};
    const auto padding = alignment - remainder;

    if (origFilter.isa<mlir::BlockArgument>()) {
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
    const auto outAllocType = mlir::RankedTensorType::get(alignedWeightShape, origFilterType.getElementType())
                                      .cast<vpux::NDTypeInterface>();
    const auto outAllocTypeNHWC = outAllocType.changeDimsOrder(DimsOrder::NHWC);

    auto alignedWeightsOp =
            builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, std::move(alignedWeightsContentAttr));
    return alignedWeightsOp.getOutput();
}

Byte calculateAlignedBuffersMemoryRequirement(VPU::ArchKind arch, SmallVector<Byte>& bufferSizes) {
    Byte offsetAlignment = Byte(vpux::DEFAULT_CMX_ALIGNMENT);
    Byte sizeAlignment = Byte(1);
    if (arch == VPU::ArchKind::NPU37XX || arch == VPU::ArchKind::NPU40XX) {
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
    const auto bounds =
            mlir::isa<BoundedTypeInterface>(inType) ? mlir::cast<BoundedTypeInterface>(inType).getBounds() : nullptr;
    return vpux::getTensorAttr(inType.getDimsOrder().toAffineMap(ctx), inType.getMemSpace(), bounds);
}

}  // namespace VPU
}  // namespace vpux
