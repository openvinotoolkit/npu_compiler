//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"

namespace vpux {
namespace VPU {

std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights,
                                            const Const::ContentAttr& bias, int64_t OC,
                                            VPU::NCESparsity::PPEConverterCb ppeConverter,
                                            VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale,
                                            bool hasAutopad);
std::vector<int32_t> createWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                            const Const::ContentAttr& bias, int64_t OC,
                                            VPU::NCESparsity::PPEConverterCb ppeConverter,
                                            VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale,
                                            bool hasAutopad);

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable,
                                     vpux::ShapeRef weightsTableShape);

template <typename T>
std::vector<int32_t> createDataPointerTableData(mlir::Value opInput, mlir::Value opOutput,
                                                ArrayRef<int32_t> workloadSizes, mlir::Value weights,
                                                int32_t weightPtrOffset, int64_t OC,
                                                ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    const auto weightPtrStep = VPU::NCESparsity::getWeightPtrStep(weights);

    const auto inElemType = mlir::cast<vpux::NDTypeInterface>(opInput.getType()).getElementType();
    const auto outElemType = mlir::cast<vpux::NDTypeInterface>(opOutput.getType()).getElementType();

    return VPU::NCESparsity::getDataPointerTable(inElemType, outElemType, workloadSizes, weightPtrOffset, weightPtrStep,
                                                 OC, zeroPoints);
}

template <typename T>
std::pair<std::vector<int32_t>, std::vector<int32_t>> createSparseDataPointerTableDataPair(
        mlir::Value opInput, mlir::Value opOutput, ArrayRef<int32_t> workloadSizes, mlir::Value weights,
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

template <typename T>
std::vector<T> createZeroPointOnlyTableData(ArrayRef<int32_t> workloadSizes, mlir::Value weights, int64_t OC,
                                            bool isZeroPoint4Bit, ArrayRef<T> zeroPoints) {
    const auto weightsElemType =
            weights ? mlir::cast<vpux::NDTypeInterface>(weights.getType()).getElementType() : nullptr;

    return VPU::NCESparsity::getZeroPointOnlyTable(workloadSizes, OC, weightsElemType, isZeroPoint4Bit, zeroPoints);
}

template <typename Type>
mlir::Value createNewWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<Type> table,
                                        vpux::ShapeRef weightsTableShape, mlir::Type elemType) {
    const auto dataStorageType = mlir::RankedTensorType::get(weightsTableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, table);
}

struct NewWeightsTableData {
    NewWeightsTableData(bool useNewWeightTableFormat, mlir::Value opInput, mlir::Type opOutputElemType,
                        mlir::Value weights, const Const::ContentAttr& bias, int64_t OC,
                        VPU::NCESparsity::PPEConverterCb ppeConverter, VPU::NCESparsity::BiasConverterCb biasConverter,
                        mlir::FloatAttr constScale);
    NewWeightsTableData(bool useNewWeightTableFormat, mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                        const Const::ContentAttr& bias, int64_t OC, VPU::NCESparsity::PPEConverterCb ppeConverter,
                        VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale);

    std::vector<int32_t> dataPointerData{}, sparsityPointerData{};
    std::vector<float> scaleData{}, biasData{};
    std::vector<int8_t> zeroPointData{};

private:
    void initializeData(bool useNewWeightTableFormat, mlir::Value opInput, mlir::Type opOutputElemType,
                        mlir::Value weights, const Const::ContentAttr& bias, int64_t OC,
                        VPU::NCESparsity::PPEConverterCb ppeConverter, VPU::NCESparsity::BiasConverterCb biasConverter,
                        mlir::FloatAttr constScale);
};

struct NewWeightsTableTensors {
    NewWeightsTableTensors(bool useNewWeightTableFormat, mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights,
                           const Const::ContentAttr& bias, ShapeRef weightTableShape,
                           VPU::NCESparsity::PPEConverterCb ppeConverter,
                           VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale = nullptr);
    NewWeightsTableTensors(bool useNewWeightTableFormat, mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                           const Const::ContentAttr& bias, ShapeRef weightTableShape,
                           VPU::NCESparsity::PPEConverterCb ppeConverter,
                           VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale = nullptr);

    mlir::Value dataPointerTensor = nullptr, sparsityPointerTensor = nullptr, scaleTensor = nullptr,
                biasTensor = nullptr, zeroPointTensor = nullptr;

private:
    void initializeTensors(bool useNewWeightTableFormat, mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights,
                           const Const::ContentAttr& bias, ShapeRef weightTableShape,
                           VPU::NCESparsity::PPEConverterCb ppeConverter,
                           VPU::NCESparsity::BiasConverterCb biasConverter, mlir::FloatAttr constScale);
    mlir::Value initializeScaleBiasTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<float> tableData,
                                          ShapeRef weightTableShape);
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

}  // namespace VPU
}  // namespace vpux
