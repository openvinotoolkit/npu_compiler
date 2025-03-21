//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

//
// build
//

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       bool isWeights, VPUIP::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, nullptr, nullptr, isWeights, sparsityCompression);
}

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       mlir::Value sparsityMap, bool isWeights,
                                       VPUIP::SparsityCompressionAttr sparsityCompression) {
    build(builder, state, data, sparsityMap, nullptr, isWeights, sparsityCompression);
}

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       mlir::Value sparsityMap, mlir::Value storageElementTable, bool isWeights,
                                       VPUIP::SparsityCompressionAttr sparsityCompression) {
    const auto isWeightsAttr = isWeights ? mlir::UnitAttr::get(builder.getContext()) : nullptr;
    build(builder, state, data, sparsityMap, storageElementTable, isWeightsAttr, sparsityCompression, nullptr);
}

void VPUIP::GroupSparseBufferOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value data,
                                       mlir::Value sparsityMap, mlir::Value storageElementTable, VPU::SEAttr seAttr) {
    build(builder, state, data, sparsityMap, storageElementTable, nullptr, nullptr, seAttr);
}

//
// getViewSources
//

mlir::ValueRange VPUIP::GroupSparseBufferOp::getViewSources() {
    return getOperands();
}

//
// inferReturnTypes
//

mlir::LogicalResult VPUIP::GroupSparseBufferOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                                 std::optional<mlir::Location> optLoc,
                                                                 mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                                 mlir::OpaqueProperties props,
                                                                 mlir::RegionRange /*ranges*/,
                                                                 SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::GroupSparseBufferOpAdaptor groupOp(operands, attrs, props);
    if (mlir::failed(groupOp.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = groupOp.getData().getType();
    const auto sparsityMapType = groupOp.getSparsityMap() != nullptr ? groupOp.getSparsityMap().getType() : nullptr;
    const auto storageElementTableType =
            groupOp.getStorageElementTable() != nullptr ? groupOp.getStorageElementTable().getType() : nullptr;

    inferredReturnTypes.push_back(
            VPUIP::SparseBufferType::get(dataType, sparsityMapType, storageElementTableType, groupOp.getIsWeightsAttr(),
                                         groupOp.getSparsityCompressionAttr(), groupOp.getSeAttrAttr()));

    return mlir::success();
}
