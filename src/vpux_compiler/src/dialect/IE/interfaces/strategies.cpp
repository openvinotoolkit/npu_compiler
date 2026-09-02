//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/interfaces/convert_weights_to_unsigned_strategy.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <algorithm>

using namespace vpux;

void vpux::IE::setIEStrategyFactory(mlir::MLIRContext* context, std::unique_ptr<IE::StrategyFactory> factory) {
    auto& registeredInterface = getCache<IE::StrategyFactoryCache, IE::IEDialect>(context);
    registeredInterface.setStrategyFactory(std::move(factory));
}

const std::unique_ptr<IE::StrategyFactory>& vpux::IE::getIEStrategyFactory(mlir::MLIRContext* context) {
    auto& registeredInterface = getCache<IE::StrategyFactoryCache, IE::IEDialect>(context);
    return registeredInterface.getStrategyFactory();
}

void vpux::IE::IConvertWeightsToUnsignedStrategy::addTypeConversions(mlir::TypeConverter& typeConverter) const {
    typeConverter.addConversion([this](vpux::NDTypeInterface tensor) -> vpux::NDTypeInterface {
        if (const auto quantType = mlir::dyn_cast_if_present<mlir::quant::QuantizedType>(tensor.getElementType())) {
            const auto newQuantType = tryChangeStorageTypeToUnsigned(quantType);
            if (newQuantType != quantType) {
                return tensor.changeElemType(newQuantType);
            }
        }

        return tensor;
    });
}

mlir::quant::QuantizedType vpux::IE::changeStorageTypeToUnsigned(mlir::quant::QuantizedType originQType,
                                                                 mlir::Type unsignedStorageType) {
    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(originQType)) {
        const auto low = uniformType.getStorageTypeMin();
        return mlir::quant::UniformQuantizedType::get(0, unsignedStorageType, uniformType.getExpressedType(),
                                                      uniformType.getScale(), uniformType.getZeroPoint() - low,
                                                      uniformType.getStorageTypeMin() - low,
                                                      uniformType.getStorageTypeMax() - low);
    }

    if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(originQType)) {
        const auto low = perAxisType.getStorageTypeMin();
        const auto zeroPoints = perAxisType.getZeroPoints();

        SmallVector<int64_t> newZeroPoints(zeroPoints.size());
        std::transform(zeroPoints.begin(), zeroPoints.end(), newZeroPoints.begin(), [low](int64_t zp) {
            return zp - low;
        });

        return mlir::quant::UniformQuantizedPerAxisType::get(
                0, unsignedStorageType, perAxisType.getExpressedType(), perAxisType.getScales(), newZeroPoints,
                perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin() - low,
                perAxisType.getStorageTypeMax() - low);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originQType);
}
