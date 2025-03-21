//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
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

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable);

std::vector<int32_t> createDataPointerTableData(mlir::Value opInput, mlir::Value opOutput,
                                                ArrayRef<int32_t> workloadsSizes, mlir::Value weights,
                                                int32_t weightPtrOffset, int64_t OC, ArrayRef<uint8_t> zeroPoints = {});

std::pair<std::vector<int32_t>, std::vector<int32_t>> createSparseDataPointerTableDataPair(
        mlir::Value opInput, mlir::Value opOutput, ArrayRef<int32_t> workloadsSizes, mlir::Value weights,
        int32_t weightPtrOffset, int32_t sparsityPtrOffset, ArrayRef<uint8_t> sparsityArray, int64_t OC,
        ArrayRef<uint8_t> zeroPoints = {});

template <typename T>
std::vector<T> createScaleTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights, int64_t OC,
                                    VPU::NCESparsity::PPEConverterCb ppeConverter, mlir::FloatAttr constScale) {
    const auto inElemType = opInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto outElemType = opOutput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto weightsElemType = weights ? weights.getType().cast<vpux::NDTypeInterface>().getElementType() : nullptr;

    return VPU::NCESparsity::getScaleTable<T>(inElemType, outElemType, ppeConverter, OC, weightsElemType, constScale);
}

std::vector<float> createBiasTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                       const Const::ContentAttr& bias, int64_t OC,
                                       VPU::NCESparsity::BiasConverterCb biasConverter);

template <typename Type>
mlir::Value createNewWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<Type> table,
                                        mlir::Type elemType) {
    const int64_t numberOfElements = table.size();

    const auto tableShape = NCESparsity::inferWeightsTablesNewFormatShape(numberOfElements);

    const auto dataStorageType = mlir::RankedTensorType::get(tableShape.raw(), elemType);
    return Const::createConst(builder, loc, dataStorageType, table);
}

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
Byte calculateAlignedBuffersMemoryRequirement(VPU::ArchKind arch, mlir::SmallVector<Byte>& bufferSizes);

}  // namespace VPU
}  // namespace vpux
