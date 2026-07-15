//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

namespace vpux {
namespace VPU {

template <typename Type>
mlir::Value createTensorFromTableData(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<Type> tableData,
                                      vpux::ShapeRef weightsTableShape, mlir::Type elemType) {
    const auto dataStorageType = mlir::RankedTensorType::get(weightsTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, tableData);
}

// Indicates which workload-dependent table (if any) the new weights-table format must materialize for a given op.
// Conv-like ops use a zero-point table, while DepthWise Conv and Interpolate ops use a data-pointer table.
enum class NewWeightsTableKind { None, DataPointer, ZeroPoint };
NewWeightsTableKind getNewWeightsTableKind(mlir::Operation* op);

struct WeightsTableParams {
    mlir::Operation* op;
    mlir::Value opInput;
    mlir::Type opOutputElemType;
    mlir::Value weights;
    Const::ContentAttr bias;
    int64_t OC;
    VPU::NCESparsity::PPEConverterCb ppeConverter;
    VPU::NCESparsity::BiasConverterCb biasConverter;
    mlir::FloatAttr constScale;
    mlir::Value zeroPoints;

    WeightsTableParams(mlir::Operation* op, mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights,
                       const Const::ContentAttr& bias, int64_t OC, VPU::NCESparsity::PPEConverterCb ppeConverter,
                       VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale,
                       mlir::Value zeroPoints)
            : op(op),
              opInput(opInput),
              opOutputElemType(opOutputElemType),
              weights(weights),
              bias(bias),
              OC(OC),
              ppeConverter(ppeConverter),
              biasConverter(biasConverter),
              constScale(constScale),
              zeroPoints(zeroPoints) {
    }

    WeightsTableParams(mlir::Operation* op, mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                       const Const::ContentAttr& bias, int64_t OC, VPU::NCESparsity::PPEConverterCb ppeConverter,
                       VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale,
                       mlir::Value zeroPoints)
            : op(op),
              opInput(opInput),
              opOutputElemType(mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType()),
              weights(weights),
              bias(bias),
              OC(OC),
              ppeConverter(ppeConverter),
              biasConverter(biasConverter),
              constScale(constScale),
              zeroPoints(zeroPoints) {
    }
};

//
// Legacy weights table format
//

std::vector<int32_t> createWeightsTableData(const WeightsTableParams& params, bool hasAutopad);

//
// New weights table format
//

SmallVector<int32_t> materializeDataPointerTable(mlir::MLIRContext* context,
                                                 ArrayRef<SmallVector<int32_t>> workloadSizes, mlir::Value weights,
                                                 int32_t weightPtrOffset, int64_t OC, mlir::Value zeroPoints = nullptr);
template <typename T>
std::vector<int32_t> createDataPointerTableData(mlir::MLIRContext* context,
                                                ArrayRef<SmallVector<int32_t>> workloadSizes, mlir::Value weights,
                                                int32_t weightPtrOffset, int64_t OC,
                                                ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    const auto weightPtrStep = VPU::NCESparsity::getWeightPtrStep(weights);

    return VPU::NCESparsity::getDataPointerTable(context, workloadSizes, weightPtrOffset, weightPtrStep, OC,
                                                 zeroPoints);
}

template <typename T>
std::pair<std::vector<int32_t>, std::vector<int32_t>> createSparseDataPointerTableDataPair(
        mlir::Value opInput, mlir::Value opOutput, ArrayRef<SmallVector<int32_t>> workloadSizes, mlir::Value weights,
        int32_t weightPtrOffset, int32_t sparsityPtrOffset, ArrayRef<uint8_t> sparsityArray, int64_t OC,
        ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(opInput.getType()).getElementType();
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();
    const auto weightsElemType =
            weights ? mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType() : nullptr;

    const auto weightsShape = getShape(weights);

    return VPU::NCESparsity::getSparseDataPointerTablePair(inElemType, outElemType, workloadSizes, weightPtrOffset,
                                                           weightsShape, sparsityPtrOffset, sparsityArray, OC,
                                                           weightsElemType, zeroPoints);
}

template <typename T>
std::vector<T> createScaleTableData(mlir::Value opInput, mlir::Type outElemType, mlir::Value weights, int64_t OC,
                                    VPU::NCESparsity::PPEConverterCb ppeConverter, mlir::FloatAttr constScale) {
    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(opInput.getType()).getElementType();
    const auto weightsElemType =
            weights ? mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType() : nullptr;

    return VPU::NCESparsity::getScaleTable<T>(inElemType, outElemType, ppeConverter, OC, weightsElemType, constScale);
}

template <typename T>
std::vector<T> createScaleTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights, int64_t OC,
                                    VPU::NCESparsity::PPEConverterCb ppeConverter, mlir::FloatAttr constScale) {
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();

    return createScaleTableData<T>(opInput, outElemType, weights, OC, ppeConverter, constScale);
}

std::vector<float> createBiasTableData(mlir::Value opInput, mlir::Type outElemType, mlir::Value weights,
                                       const Const::ContentAttr& bias, int64_t OC,
                                       VPU::NCESparsity::BiasConverterCb biasConverter);

std::vector<float> createBiasTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                       const Const::ContentAttr& bias, int64_t OC,
                                       VPU::NCESparsity::BiasConverterCb biasConverter);

SmallVector<int32_t> materializeZeroPointTable(mlir::Type weightsElemType, int64_t OC, ArrayRef<int32_t> workloadSizes,
                                               mlir::Value zeroPoints);

template <typename T>
std::vector<T> createZeroPointTableData(ArrayRef<int32_t> workloadSizes, mlir::Type weightsElemType, int64_t OC,
                                        bool isZeroPoint4Bit, ArrayRef<T> zeroPoints) {
    return VPU::NCESparsity::getZeroPointTable(workloadSizes, OC, weightsElemType, isZeroPoint4Bit, zeroPoints);
}

struct NewWeightsTableData {
    NewWeightsTableData(bool useNewWeightTableFormat, const WeightsTableParams& params);

    std::vector<int32_t> dataPointerData{}, sparsityPointerData{};
    std::vector<float> scaleData{}, biasData{};
    std::vector<int8_t> zeroPointData{};
};

struct NewWeightsTableTensors {
    NewWeightsTableTensors(bool useNewWeightTableFormat, const WeightsTableParams& params, mlir::OpBuilder& builder,
                           mlir::Location loc, ShapeRef weightTableShape);

    mlir::Value dataPointerTensor = nullptr, sparsityPointerTensor = nullptr, scaleTensor = nullptr,
                biasTensor = nullptr, zeroPointTensor = nullptr;

private:
    mlir::Value initializeDataPointerTensorWithDummyValues(mlir::OpBuilder& builder, mlir::Location loc,
                                                           ArrayRef<int32_t> tableData, ShapeRef weightTableShape,
                                                           mlir::Value zeroPoints);
    mlir::Value initializeScaleBiasTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<float> tableData,
                                          ShapeRef weightTableShape);
    mlir::Value initializeZeroPointsTensorWithDummyValues(mlir::OpBuilder& builder, mlir::Location loc,
                                                          ArrayRef<int8_t> tableData, ShapeRef weightTableShape,
                                                          mlir::Value zeroPoints);
};

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);
mlir::Value alignConvWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);
bool isNullOrConstWithSingleValue(mlir::Value value);

/**
 * @brief calculate memory requirement for given buffer sizes and architecture-dependent allocation requirements
 *
 * @param arch - architecture type
 * @param bufferSizes - vector containing sizes [bytes] of buffers to be allocated
 *
 * @return required memory taking into account the allocation requirements for swizzled buffers [bytes].
 *
 * Starting with NPU37XX the required memory size is
 * calculated according to requirements for CMX allocation for swizzled buffers.
 *
 * NOTE: see also vpux::calculateAlignedBuffersMemoryRequirement
 */
Byte calculateAlignedBuffersMemoryRequirement(config::ArchKind arch, mlir::SmallVector<Byte>& bufferSizes);

mlir::FailureOr<SmallVector<int64_t>> extractConstData(mlir::Location loc, mlir::Value value);

}  // namespace VPU
}  // namespace vpux
