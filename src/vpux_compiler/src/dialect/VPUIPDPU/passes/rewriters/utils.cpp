//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {
namespace VPUIPDPU {

IOType getIOType(mlir::Type type) {
    auto baseType = getBaseType(type);
    if (mlir::isa<mlir::IntegerType>(baseType)) {
        return IOType::INT;
    } else if (mlir::isa<mlir::FloatType>(baseType)) {
        return IOType::FP;
    }

    return IOType::IOTypeNum;
}

mlir::FloatAttr getF32FloatAttrOrNull(mlir::OpBuilder& builder, const std::optional<float>& attr) {
    if (attr.has_value()) {
        return builder.getF32FloatAttr(attr.value());
    }

    return nullptr;
}

mlir::ArrayAttr getF64ArrayAttrOrNull(mlir::OpBuilder& builder, const std::optional<llvm::ArrayRef<double>>& attr) {
    if (attr.has_value()) {
        return builder.getF64ArrayAttr(attr.value());
    }

    return nullptr;
}

mlir::MemRefType getBufferType(mlir::Operation* bufferRef) {
    mlir::MemRefType bufferType;

    if (mlir::isa<VPUASM::DeclareBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(bufferRef);
        bufferType = buffer.getBufferType().getMemref();
    } else if (mlir::isa<VPUASM::ConstBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(bufferRef);
        bufferType = buffer.getBufferType().getMemref();
    } else {
        VPUX_THROW("Not a buffer: {0}", bufferRef);
    }

    return bufferType;
}

uint64_t getSwizzlingKey(mlir::Operation* bufferRef) {
    uint64_t swizzlingKey = 0;

    if (mlir::isa<VPUASM::DeclareBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::DeclareBufferOp>(bufferRef);
        swizzlingKey = buffer.getBufferType().getTraits().getSwizzlingKey();
    } else if (mlir::isa<VPUASM::ConstBufferOp>(bufferRef)) {
        auto buffer = mlir::cast<VPUASM::ConstBufferOp>(bufferRef);
        swizzlingKey = buffer.getBufferType().getTraits().getSwizzlingKey();
    } else {
        VPUX_THROW("Not a buffer: {0}", bufferRef);
    }

    return swizzlingKey;
}

mlir::BlockArgument getInvBlockArg(BlockArg invBlockArg, mlir::Block* invBlock,
                                   const std::unordered_map<BlockArg, size_t>& invBlockArgsPos) {
    auto arg = invBlockArgsPos.find(invBlockArg);
    if (arg == invBlockArgsPos.end()) {
        return nullptr;
    }

    return invBlock->getArgument(arg->second);
}

mlir::Type getBaseType(mlir::Type type, bool isPalletModeEnabled) {
    if (!mlir::isa<mlir::quant::QuantizedType>(type)) {
        return type;
    }

    if (isPalletModeEnabled) {
        VPUX_THROW_UNLESS(
                mlir::isa<mlir::quant::QuantileQuantizedType>(type) ||
                        mlir::isa<mlir::quant::QuantileQuantizedPerAxisType>(type),
                "Pallet mode requires weights to be of QuantileQuantizedType or QuantileQuantizedPerAxisType, "
                "containing the quantile look up table");

        // In case of palletization the actual de-palletized weight type is contained in the quantileType field
        if (auto quantType = mlir::dyn_cast<mlir::quant::QuantileQuantizedType>(type)) {
            return quantType.getQuantileType();
        }
        if (auto quantType = mlir::dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>(type)) {
            return quantType.getQuantileType();
        }
    }

    auto quantType = mlir::cast<mlir::quant::QuantizedType>(type);
    auto quantStorageType = quantType.getStorageType();
    if (mlir::isa<mlir::Float8E5M2Type>(quantStorageType)) {
        return mlir::Float8E5M2Type::get(type.getContext());
    }

    if (mlir::isa<mlir::Float8E4M3FNType>(quantStorageType)) {
        return mlir::Float8E4M3FNType::get(type.getContext());
    }

    auto signedness = quantType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
    auto bitWidth = quantType.getStorageTypeIntegralWidth();
    return mlir::IntegerType::get(type.getContext(), bitWidth, signedness);
}

mlir::LogicalResult getQuantConfig(const Logger&, mlir::Type type, SmallVector<int64_t>& quantMult,
                                   SmallVector<int64_t>& quantShift, SmallVector<uint8_t>& quantZero) {
    if (const auto qType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(type)) {
        quantZero.push_back(checked_cast<uint8_t>(qType.getZeroPoint()));
        const auto scaleApproximation = QuantizationApproximation(qType.getScale());
        quantMult.push_back(scaleApproximation.mult());
        quantShift.push_back(scaleApproximation.shift());
    } else if (const auto qPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(type)) {
        auto qtypeQuantZp = qPerAxisType.getZeroPoints();
        auto qtypeQuantScale = qPerAxisType.getScales();

        quantZero.resize(qtypeQuantZp.size());
        std::transform(qtypeQuantZp.begin(), qtypeQuantZp.end(), quantZero.begin(), [](int64_t val) {
            return checked_cast<uint8_t>(val);
        });

        quantMult.resize(qtypeQuantScale.size());
        quantShift.resize(qtypeQuantScale.size());
        for (std::size_t i = 0; i < qtypeQuantScale.size(); ++i) {
            const auto scaleApproximation = QuantizationApproximation(qtypeQuantScale[i]);
            quantMult[i] = scaleApproximation.mult();
            quantShift[i] = scaleApproximation.shift();
        }
    } else {
        quantMult.push_back(1);
        quantShift.push_back(0);
        quantZero.push_back(0);
    }

    return mlir::success();
}

mlir::IntegerAttr getI64IntegerAttrOrNull(mlir::OpBuilder& builder, const std::optional<int64_t>& attr) {
    if (attr.has_value()) {
        return builder.getI64IntegerAttr(attr.value());
    }

    return nullptr;
}

VPUIPDPU::ODUDataBitWidth getDataBitWidth(mlir::Type outActType) {
    auto asIntegerType = mlir::dyn_cast<mlir::IntegerType>(outActType);
    auto asFloatType = mlir::dyn_cast<mlir::FloatType>(outActType);
    VPUX_THROW_UNLESS(asIntegerType || asFloatType, "Not a Float or Integer Type");

    const auto width = asIntegerType ? asIntegerType.getWidth() : asFloatType.getWidth();
    const auto oduBitWidth = VPUIPDPU::symbolizeODUDataBitWidth(log2(width));
    VPUX_THROW_UNLESS(oduBitWidth.has_value(), "Unable to determine data bit width from out_activations {0}",
                      outActType);

    return oduBitWidth.value();
}

std::optional<ODUDataBitWidth> getOutDataWidth(mlir::Type outDataType) {
    std::optional<ODUDataBitWidth> outDataWidth;

    if (outDataType.isF32()) {
        return ODUDataBitWidth::ODU_DTYPE_32BIT;
    } else if (outDataType.isF16()) {
        return ODUDataBitWidth::ODU_DTYPE_16BIT;
    } else if (outDataType.isBF16()) {
        return ODUDataBitWidth::ODU_DTYPE_16BIT;
    } else if (mlir::isa<mlir::Float8E4M3FNType>(outDataType)) {
        return ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (mlir::isa<mlir::Float8E5M2Type>(outDataType)) {
        return ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (outDataType.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return ODUDataBitWidth::ODU_DTYPE_32BIT;
    } else if (outDataType.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (outDataType.isSignedInteger(4)) {
        return ODUDataBitWidth::ODU_DTYPE_4BIT;
    } else if (outDataType.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (outDataType.isInteger(4)) {
        return ODUDataBitWidth::ODU_DTYPE_4BIT;
    } else if (outDataType.isInteger(2)) {
        return ODUDataBitWidth::ODU_DTYPE_2BIT;
    } else if (outDataType.isInteger(1)) {
        return ODUDataBitWidth::ODU_DTYPE_1BIT;
    } else if (mlir::isa<mlir::quant::QuantizedType>(outDataType)) {
        return getOutDataWidth(mlir::cast<mlir::quant::QuantizedType>(outDataType).getStorageType());
    }

    return outDataWidth;
}

int64_t getRangeSize(int64_t start, int64_t end) {
    return end - start + 1;
}

int64_t getZeroPoint(vpux::NDTypeInterface type) {
    int64_t zeroPointVal = 0;
    auto elemType = type.getElementType();
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(qType.getStorageType())) {
            if (intType.getWidth() == 8) {
                zeroPointVal = VPUIPDPU::getZeroPoints<uint8_t>(type.getElementType())[0];
            } else if (intType.getWidth() == 16) {
                zeroPointVal = VPUIPDPU::getZeroPoints<uint16_t>(type.getElementType())[0];
            }
        }
    }
    return zeroPointVal;
}

}  // namespace VPUIPDPU
}  // namespace vpux
