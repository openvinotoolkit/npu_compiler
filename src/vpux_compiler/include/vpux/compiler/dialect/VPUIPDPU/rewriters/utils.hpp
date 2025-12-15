//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Builders.h>

namespace vpux::VPUIPDPU {
enum class ODUDataBitWidth : uint32_t;
}  // namespace vpux::VPUIPDPU

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
    DYNAMIC_SEQUENCE_LENGTH,
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

std::optional<ODUDataBitWidth> getOutDataWidth(mlir::Type outDataType);

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

template <typename DataType>
SmallVector<DataType> getZeroPoints(mlir::Type type) {
    static_assert(std::is_integral<DataType>::value, "DataType must be an integer type");
    SmallVector<DataType> quantZeroPoints;

    if (const auto uniformQuantType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type)) {
        quantZeroPoints.push_back(checked_cast<DataType>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(type)) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        quantZeroPoints.resize(zp.size());
        std::transform(zp.begin(), zp.end(), quantZeroPoints.begin(), [](int64_t a) {
            return checked_cast<DataType>(a);
        });
    } else {
        quantZeroPoints.push_back(0);
    }

    return quantZeroPoints;
}

// Helper function to calculate zero point offset for input/output activations and weights
int64_t getZeroPoint(vpux::NDTypeInterface type);

int64_t getRangeSize(int64_t start, int64_t end);

}  // namespace VPUIPDPU
}  // namespace vpux
