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

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>
#include <tuple>

namespace vpux {

//
// FakeQuantize support
//

mlir::quant::QuantizedType expandScalesAndZP(mlir::quant::UniformQuantizedPerAxisType perAxisQType, ShapeRef padBefore,
                                             ShapeRef padAfter);
std::tuple<double, int64_t> calcScaleAndZeroPoint(int64_t qMin, int64_t qMax, double rMin, double rMax, bool isSigned);

mlir::quant::QuantizedType getQuantizedType(Const::ContentAttr lowConst, Const::ContentAttr highConst, int64_t levels,
                                            mlir::FloatType realType, mlir::Location loc);

mlir::LogicalResult getFakeQuantParams(mlir::ShapedType qType, int64_t& levels, mlir::RankedTensorType& attrType,
                                       mlir::DenseElementsAttr& rMinAttr, mlir::DenseElementsAttr& rMaxAttr,
                                       mlir::Location loc);

mlir::Type normalizeQuantStorageType(mlir::quant::QuantizedType qType);

//
// Dequantize support
//

float dequantize(int64_t qVal, double scale, int64_t zeroPoint);

//
// Convert real numbers to fixed point S16.16 format.
//

int32_t toFixedPoint(const double realVal);

}  // namespace vpux
