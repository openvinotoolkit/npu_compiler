//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>
#include <tuple>

namespace vpux {

//
// Utilities for quantized types
//

mlir::LogicalResult validateQuantElemType(mlir::Location loc, mlir::ShapedType mainType);

mlir::Type normalizeQuantStorageType(mlir::quant::QuantizedType qType);

mlir::quant::UniformQuantizedPerAxisType expandScalesAndZP(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                           ShapeRef padBefore, ShapeRef padAfter);

mlir::quant::UniformQuantizedPerAxisType tileScalesAndZP(mlir::quant::UniformQuantizedPerAxisType perAxisQType,
                                                         ShapeRef shape, ShapeRef offsets);

bool canBeMerged(mlir::quant::UniformQuantizedPerAxisType type1, mlir::quant::UniformQuantizedPerAxisType type2);
mlir::quant::UniformQuantizedPerAxisType concatScalesAndZP(ArrayRef<mlir::quant::UniformQuantizedPerAxisType> types);

using Scales = SmallVector<double>;
using ZeroPoints = SmallVector<int64_t>;

std::pair<Scales, ZeroPoints> extractScalesAndZeroPoints(mlir::Type tensorElemType);

uint16_t getQuantMultFromScale(double quantScale);
uint8_t getQuantShiftFromScale(double quantScale);
std::pair<uint8_t, int8_t> getQuantShiftAndPostShiftFromScale(double quantScale);

//
// FakeQuantize support
//

mlir::quant::QuantizedType getQuantizedType(mlir::Attribute lowConstAttr, mlir::Attribute highConstAttr, int64_t levels,
                                            mlir::FloatType realType, bool isSigned, mlir::Location loc);

void getFakeQuantParams(mlir::quant::UniformQuantizedType qElemType, int64_t& levels, float& rMin, float& rMax);

void getFakeQuantParams(mlir::quant::UniformQuantizedPerAxisType qElemType, int64_t& levels,
                        SmallVectorImpl<float>& rMinVals, SmallVectorImpl<float>& rMaxVals);

void getFakeQuantParams(mlir::ShapedType qType, int64_t& levels, mlir::RankedTensorType& attrType,
                        mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr);

std::tuple<double, int64_t> calcScaleAndZeroPoint(int64_t qMin, int64_t qMax, double rMin, double rMax, bool isSigned);

//
// Dequantize support
//

float dequantize(int64_t qVal, double scale, int64_t zeroPoint);

//
// Convert real numbers to fixed point S16.16 format.
//

int32_t toFixedPoint(const double realVal);

}  // namespace vpux
