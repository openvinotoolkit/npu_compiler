//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux {
namespace VPUIPDPU {

IOType getIOType(mlir::Type type) {
    auto baseType = getBaseType(type);
    if (baseType.isa<mlir::IntegerType>()) {
        return IOType::INT;
    } else if (baseType.isa<mlir::FloatType>()) {
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
    if (!type.isa<mlir::quant::QuantizedType>()) {
        return type;
    }

    if (isPalletModeEnabled) {
        VPUX_THROW_UNLESS(
                type.isa<mlir::quant::QuantileQuantizedType>() || type.isa<mlir::quant::QuantileQuantizedPerAxisType>(),
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

    auto quantType = type.cast<mlir::quant::QuantizedType>();
    auto quantStorageType = quantType.getStorageType();
    if (quantStorageType.isFloat8E5M2()) {
        return mlir::Float8E5M2Type::get(type.getContext());
    }

    if (quantStorageType.isFloat8E4M3FN()) {
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
    auto asIntegerType = outActType.dyn_cast<mlir::IntegerType>();
    auto asFloatType = outActType.dyn_cast<mlir::FloatType>();
    VPUX_THROW_UNLESS(asIntegerType || asFloatType, "Not a Float or Integer Type");

    const auto width = asIntegerType ? asIntegerType.getWidth() : asFloatType.getWidth();
    const auto oduBitWidth = VPUIPDPU::symbolizeODUDataBitWidth(log2(width));
    VPUX_THROW_UNLESS(oduBitWidth.has_value(), "Unable to determine data bit width from out_activations {0}",
                      outActType);

    return oduBitWidth.value();
}

}  // namespace VPUIPDPU
}  // namespace vpux
