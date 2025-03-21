//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"

#include <mlir/IR/Builders.h>

namespace vpux {
namespace VPUIPDPU {

enum class IOType { INT, FP, IOTypeNum };

enum class BlockArg {
    ACT_IN,
    ACT_SE_IN,
    ACT_SPARSE_MAP_IN,
    WEIGHTS_TABLE,
    WEIGHTS_TABLE_DATA_PTR,
    WEIGHTS_TABLE_SP_PTR,
    WEIGHTS_TABLE_SCALE,
    WEIGHTS_TABLE_BIAS,
    WEIGHTS_ZERO_POINTS,
    WEIGHTS,
    WEIGHTS_SPARSE_MAP,
    SPR_LOOKUP_TABLE,
    PALLET_LOOKUP_TABLE,
    ACT_OUT,
    ACT_SPARSE_MAP_OUT,
    Count
};

IOType getIOType(mlir::Type type);

template <typename AttrType, typename ValueType>
AttrType getEnumAttrOrNull(mlir::OpBuilder& builder, const std::optional<ValueType>& attr) {
    if (attr.has_value()) {
        return AttrType::get(builder.getContext(), attr.value());
    }

    return nullptr;
}
mlir::FloatAttr getF32FloatAttrOrNull(mlir::OpBuilder& builder, const std::optional<float>& attr);
mlir::ArrayAttr getF64ArrayAttrOrNull(mlir::OpBuilder& builder, const std::optional<llvm::ArrayRef<double>>& attr);

mlir::MemRefType getBufferType(mlir::Operation* bufferRef);

uint64_t getSwizzlingKey(mlir::Operation* bufferRef);

mlir::BlockArgument getInvBlockArg(BlockArg invBlockArg, mlir::Block* invBlock,
                                   const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

mlir::Type getBaseType(mlir::Type type, bool isPalletModeEnabled = false);

mlir::LogicalResult getQuantConfig(const Logger&, mlir::Type type, SmallVector<int64_t>& quantMult,
                                   SmallVector<int64_t>& quantShift, SmallVector<uint8_t>& quantZero);

mlir::IntegerAttr getI64IntegerAttrOrNull(mlir::OpBuilder& builder, const std::optional<int64_t>& attr);

VPUIPDPU::ODUDataBitWidth getDataBitWidth(mlir::Type outActType);

template <typename TRegField_target_width_lsbType, typename TRegField_target_width_msbType>
void computeLsbAndMsbFromTargetWidth(int64_t targetWidth, uint64_t& msbWidth, uint64_t& lsbWidth) {
    auto lsbBitWidth = TRegField_target_width_lsbType::getRegFieldWidth();
    auto msbBitWidth = TRegField_target_width_msbType::getRegFieldWidth();

    auto bitMask = (1 << (lsbBitWidth + msbBitWidth)) - 1;
    VPUX_THROW_WHEN(targetWidth & ~bitMask, "target_width value {0} is too big for {1} bits", targetWidth,
                    lsbBitWidth + msbBitWidth);

    auto bitMaskLsb = (1 << lsbBitWidth) - 1;
    lsbWidth = targetWidth & bitMaskLsb;

    auto bitMaskMsb = ((1 << msbBitWidth) - 1) << lsbBitWidth;
    msbWidth = (targetWidth & bitMaskMsb) >> lsbBitWidth;
}
}  // namespace VPUIPDPU
}  // namespace vpux
