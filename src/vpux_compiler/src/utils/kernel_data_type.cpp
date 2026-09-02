//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/kernel_data_type.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace vpux {

sw_params::DataType getDataTypeFromMlirType(mlir::Type type) {
    if (auto quantileType = mlir::dyn_cast<vpux::type::QuantileType>(type)) {
        if (quantileType.getStorageWidth() == 4 &&
            (quantileType.getQuantiles() == vpux::type::NF4Type::getSpecQuantiles())) {
            return sw_params::DataType::NN_NF4;
        }

        const auto storageType = quantileType.getStorageType();
        if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
            const auto isSigned = quantileType.shouldDefaultToSigned();
            switch (integerType.getWidth()) {
            case 16:
                return isSigned ? sw_params::DataType::NN_I16 : sw_params::DataType::NN_U16;
            case 8:
                return isSigned ? sw_params::DataType::NN_I8 : sw_params::DataType::NN_U8;
            case 4:
                return isSigned ? sw_params::DataType::NN_I4 : sw_params::DataType::NN_U4;
            case 2:
                return isSigned ? sw_params::DataType::NN_I2 : sw_params::DataType::NN_U2;
            }
        }
    }
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
        auto typeWidth = floatType.getWidth();
        switch (typeWidth) {
        case 64:
            return sw_params::DataType::NN_FP64;
        case 32:
            return sw_params::DataType::NN_FP32;
        case 16:
            if (type.isBF16()) {
                return sw_params::DataType::NN_BF16;
            }
            return sw_params::DataType::NN_FP16;
        case 8:
            if (mlir::isa<mlir::Float8E4M3FNType>(floatType)) {
                return sw_params::DataType::NN_HF8;
            } else if (mlir::isa<mlir::Float8E5M2Type>(floatType)) {
                return sw_params::DataType::NN_BF8;
            }
            break;
        }
    } else if (auto integerType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        if (integerType.isSigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_I64;
            case 32:
                return sw_params::DataType::NN_I32;
            case 16:
                return sw_params::DataType::NN_I16;
            case 8:
                return sw_params::DataType::NN_I8;
            case 4:
                return sw_params::DataType::NN_I4;
            case 2:
                return sw_params::DataType::NN_I2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        } else if (integerType.isUnsigned()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_U64;
            case 32:
                return sw_params::DataType::NN_U32;
            case 16:
                return sw_params::DataType::NN_U16;
            case 8:
                return sw_params::DataType::NN_U8;
            case 4:
                return sw_params::DataType::NN_U4;
            case 2:
                return sw_params::DataType::NN_U2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        } else if (integerType.isSignless()) {
            auto typeWidth = integerType.getWidth();
            switch (typeWidth) {
            case 64:
                return sw_params::DataType::NN_I64;
            case 32:
                return sw_params::DataType::NN_I32;
            case 16:
                return sw_params::DataType::NN_I16;
            case 8:
                return sw_params::DataType::NN_I8;
            case 4:
                return sw_params::DataType::NN_I4;
            case 2:
                return sw_params::DataType::NN_I2;
            case 1:
                return sw_params::DataType::NN_BIN;
            }
        }
    } else if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        const auto isSigned = qType.isSigned();
        auto storageType = qType.getStorageType();
        auto isQuantileType = mlir::isa<vpux::type::QuantileType>(storageType);
        // Unwrap QuantileType to get the actual integer storage type for bitwidth
        if (const auto quantileStorageType = mlir::dyn_cast<vpux::type::QuantileType>(storageType)) {
            storageType = quantileStorageType.getStorageType();
        }
        auto bitWidth = storageType.getIntOrFloatBitWidth();
        auto isFloatStorage = mlir::isa<mlir::FloatType>(storageType);

        switch (bitWidth) {
        case 16:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I16 : sw_params::DataType::NN_U16;
            }
            break;
        case 8:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I8 : sw_params::DataType::NN_U8;
            }
            if (!isQuantileType && isFloatStorage) {
                if (mlir::isa<mlir::Float8E4M3FNType>(storageType)) {
                    return sw_params::DataType::NN_HF8;
                } else if (mlir::isa<mlir::Float8E5M2Type>(storageType)) {
                    return sw_params::DataType::NN_BF8;
                }
            }
            break;
        case 4:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I4 : sw_params::DataType::NN_U4;
            }
            if (isNF4SpecQuantized(qType)) {
                return sw_params::DataType::NN_NF4;
            }
            break;
        case 2:
            if (!isQuantileType && !isFloatStorage) {
                return isSigned ? sw_params::DataType::NN_I2 : sw_params::DataType::NN_U2;
            }
            break;
        }
    }
    VPUX_THROW("Conversion to sw_params::DataType failed for {0}", type);
    return sw_params::DataType::NN_UNDEFINED;
}

}  // namespace vpux
