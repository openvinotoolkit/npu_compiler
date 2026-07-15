//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include <vpux/utils/core/numeric.hpp>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

template <class T>
void replicateZeroPointsInPlace(SmallVector<T>& zeroPoints, size_t size) {
    VPUX_THROW_UNLESS(!zeroPoints.empty() && zeroPoints.size() <= size && size % zeroPoints.size() == 0,
                      "Cannot replicate {0} zero-points to size {1}", zeroPoints.size(), size);

    if (zeroPoints.size() == size) {
        return;
    }

    const size_t replicationFactor = size / zeroPoints.size();
    const SmallVector<T> originalValues(zeroPoints);
    zeroPoints.reserve(size);
    for (size_t i = 1; i < replicationFactor; ++i) {
        zeroPoints.append(originalValues.begin(), originalValues.end());
    }
}

//
// Legacy weights table format
//

std::vector<int32_t> createWeightsTableData(const WeightsTableParams& params, bool hasAutopad) {
    bool is5DShape = vpux::getShape(params.weights).size() == 5;

    const auto weightPtrOffset = 0;
    const auto sparsityPtrOffset = 0;
    const auto weightPtrStep = is5DShape ? VPU::NCESparsity::get5DWeightPtrStep(params.weights)
                                         : VPU::NCESparsity::getWeightPtrStep(params.weights);
    const auto sparsityPtrStep = 0;

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(params.opInput.getType()).getElementType();
    const auto weightsElemType =
            params.weights ? mlir::cast<vpux::NDTypeInterface>(params.weights.getType()).getElementType() : nullptr;

    auto wtVec = VPU::NCESparsity::getWeightsTable(
            inElemType, params.opOutputElemType, weightPtrOffset, weightPtrStep, sparsityPtrOffset, sparsityPtrStep,
            params.ppeConverter, params.biasConverter, params.OC, weightsElemType, params.bias, params.constScale);

    if (hasAutopad) {
        return VPU::NCESparsity::getExpandedWeightsTable(wtVec, params.OC);
    }

    return wtVec;
}

//
// New weights table format
//

NewWeightsTableKind getNewWeightsTableKind(mlir::Operation* op) {
    if (op == nullptr) {
        return NewWeightsTableKind::None;
    }
    if (mlir::isa<IE::GroupConvolutionOp, VPU::GroupConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCEInterpolateOp>(
                op)) {
        return NewWeightsTableKind::DataPointer;
    }
    if (mlir::isa<IE::ConvolutionOp, VPU::NCEConvolutionOp, VPU::TransposedConvolutionOp, IE::MatMulOp,
                  VPU::NCEMatMulOp>(op)) {
        return NewWeightsTableKind::ZeroPoint;
    }
    return NewWeightsTableKind::None;
}

SmallVector<int32_t> materializeDataPointerTable(mlir::MLIRContext* context,
                                                 ArrayRef<SmallVector<int32_t>> workloadSizes, mlir::Value weights,
                                                 int32_t weightPtrOffset, int64_t OC, mlir::Value zeroPoints) {
    if (zeroPoints != nullptr) {
        const auto zpConst = zeroPoints.getDefiningOp<Const::DeclareOp>();
        const auto zpElemType = mlir::cast<vpux::NDTypeInterface>(zeroPoints.getType()).getElementType();
        VPUX_THROW_UNLESS(zpElemType.isInteger(),
                          "Only zero-points of integer type supported for data-pointer table, got {0}", zpElemType);

        auto buildTable = [&](auto storageType) {
            using StorageT = decltype(storageType);
            const auto content = zpConst.getContent();
            auto zeroPointsTyped = to_small_vector(content.getValues<StorageT>());

            if (OC > static_cast<int64_t>(zeroPointsTyped.size())) {
                replicateZeroPointsInPlace(zeroPointsTyped, OC);
            }
            return to_small_vector(createDataPointerTableData<StorageT>(context, workloadSizes, weights,
                                                                        weightPtrOffset, OC, zeroPointsTyped));
        };

        return zpElemType.isSignedInteger() ? buildTable(int8_t{}) : buildTable(uint8_t{});
    }

    return to_small_vector(createDataPointerTableData<int8_t>(context, workloadSizes, weights, weightPtrOffset, OC));
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

SmallVector<int32_t> materializeZeroPointTable(mlir::Type weightsElemType, int64_t OC, ArrayRef<int32_t> workloadSizes,
                                               mlir::Value zeroPoints) {
    const auto zpConst = zeroPoints.getDefiningOp<Const::DeclareOp>();
    const auto zpElemType = mlir::cast<vpux::NDTypeInterface>(zeroPoints.getType()).getElementType();
    VPUX_THROW_UNLESS(zpElemType.isInteger(),
                      "Only zero-points of integer type supported for zero-point table, got {0}", zpElemType);
    const auto isZeroPointSubByte = zpElemType.isInteger(4) || zpElemType.isInteger(2);

    auto buildTable = [&](auto storageType) {
        using StorageT = decltype(storageType);

        const auto content = zpConst.getContent();
        auto tableData = createZeroPointTableData<StorageT>(workloadSizes, weightsElemType, OC, isZeroPointSubByte,
                                                            to_small_vector(content.getValues<StorageT>()));

        return to_small_vector(llvm::map_range(tableData, [](StorageT zp) {
            return static_cast<int32_t>(zp);
        }));
    };

    return zpElemType.isSignedInteger() ? buildTable(int8_t{}) : buildTable(uint8_t{});
}

NewWeightsTableData::NewWeightsTableData(bool useNewWeightTableFormat, const WeightsTableParams& params) {
    // leave vectors empty for archs using the legacy weights table format
    if (!useNewWeightTableFormat) {
        return;
    }

    scaleData = createScaleTableData<float>(params.opInput, params.opOutputElemType, params.weights, params.OC,
                                            params.ppeConverter, params.constScale);
    if (params.bias != nullptr) {
        biasData = createBiasTableData(params.opInput, params.opOutputElemType, params.weights, params.bias, params.OC,
                                       params.biasConverter);
    } else {
        biasData = std::vector<float>(params.OC, 0.0);
    }

    // The workload-dependent table (data-pointer or zero-point) is materialized later in create-new-weight-tables-data
    // pass, after workloads are known. Pre-fill only the table relevant to the op kind with dummy values.
    switch (getNewWeightsTableKind(params.op)) {
    case NewWeightsTableKind::DataPointer:
        dataPointerData = std::vector<int32_t>(params.OC, 0);
        break;
    case NewWeightsTableKind::ZeroPoint:
        zeroPointData = std::vector<int8_t>(params.OC, 0);
        break;
    case NewWeightsTableKind::None:
        break;
    }
}

NewWeightsTableTensors::NewWeightsTableTensors(bool useNewWeightTableFormat, const WeightsTableParams& params,
                                               mlir::OpBuilder& builder, mlir::Location loc,
                                               ShapeRef weightTableShape) {
    const auto newWeightsTableData = NewWeightsTableData(useNewWeightTableFormat, params);

    scaleTensor = initializeScaleBiasTensor(builder, loc, newWeightsTableData.scaleData, weightTableShape);
    biasTensor = initializeScaleBiasTensor(builder, loc, newWeightsTableData.biasData, weightTableShape);

    // The workload-dependent table (data-pointer or zero-point) is materialized later in create-new-weight-tables-data
    // pass, after workloads are known. Initialize the relevant tensor with dummy values now.
    switch (getNewWeightsTableKind(params.op)) {
    case NewWeightsTableKind::DataPointer:
        dataPointerTensor = initializeDataPointerTensorWithDummyValues(
                builder, loc, newWeightsTableData.dataPointerData, weightTableShape, params.zeroPoints);
        break;
    case NewWeightsTableKind::ZeroPoint:
        zeroPointTensor = initializeZeroPointsTensorWithDummyValues(builder, loc, newWeightsTableData.zeroPointData,
                                                                    weightTableShape, params.zeroPoints);
        break;
    case NewWeightsTableKind::None:
        break;
    }
}

mlir::Value NewWeightsTableTensors::initializeDataPointerTensorWithDummyValues(mlir::OpBuilder& builder,
                                                                               mlir::Location loc,
                                                                               ArrayRef<int32_t> tableData,
                                                                               ShapeRef weightTableShape,
                                                                               mlir::Value zeroPoints) {
    if (tableData.empty()) {
        return nullptr;
    }

    // Create attributes
    // The shape of the data-pointer table will be expanded based on workloads later
    auto dummyOutputType = mlir::RankedTensorType::get(weightTableShape, getSInt32Type(builder.getContext()));
    auto dataPointerTableOp = builder.create<VPU::DataPointerTableOp>(loc, dummyOutputType, zeroPoints,
                                                                      /*workloadSizes=*/nullptr,
                                                                      /*dataPointerTableData=*/nullptr);

    return dataPointerTableOp.getResult();
}

mlir::Value NewWeightsTableTensors::initializeScaleBiasTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                              ArrayRef<float> tableData, ShapeRef weightTableShape) {
    return tableData.empty()
                   ? nullptr
                   : createTensorFromTableData<float>(builder, loc, tableData, weightTableShape, builder.getF32Type());
}

mlir::Value NewWeightsTableTensors::initializeZeroPointsTensorWithDummyValues(mlir::OpBuilder& builder,
                                                                              mlir::Location loc,
                                                                              ArrayRef<int8_t> tableData,
                                                                              ShapeRef weightTableShape,
                                                                              mlir::Value zeroPoints) {
    if (tableData.empty() || zeroPoints == nullptr) {
        return nullptr;
    }

    const auto zps = mlir::cast<vpux::NDTypeInterface>(zeroPoints.getType());
    // Skip when all channels share a single zero-point (e.g. tensor<1x1x1x1xsi8>).
    if (zps.getNumElements() > 1) {
        // Create attributes
        // The shape of the zero-point table will be expanded based on workloads later.
        auto dummyOutputType = mlir::RankedTensorType::get(weightTableShape, builder.getI8Type());
        auto createZpTableOp = builder.create<VPU::ZeroPointTableOp>(loc, dummyOutputType, zeroPoints,
                                                                     /*workloadSizes=*/nullptr,
                                                                     /*zeroPointTableData=*/nullptr);

        return createZpTableOp.getResult();
    }

    return nullptr;
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
                              "Number of zero-points {0} and channels {1} don't match", zeroPoints.size(), OC);

            // assuming all zero-points are equal to broadcast
            VPUX_THROW_UNLESS(
                    zeroPoints.size() == 1 || std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin()),
                    "All zero-points should be equal");
            padValueUint8 = static_cast<uint8_t>(zeroPoints.front());
        } else {
            VPUX_THROW("Unsupported Quantized Type '{0}'", quantizedType);
        }
        const auto padAttr = Const::createConstContent(padType, ArrayRef(padValueUint8));

        return Const::ContentAttr::get(padAttr, Const::ContentSetup(padAttr, padType).castElemType(quantizedType));
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
        auto reshape = builder.create<VPU::ReshapeOp>(loc, origFilter, getIntArrayAttr(builder, flatWeightShape));

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
