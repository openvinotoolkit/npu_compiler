//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux {
//
// updateFunctionSignature
//

mlir::LogicalResult updateFunctionSignature(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                            ArrayRef<mlir::Type> newResultTypes, Logger log = Logger::global());

//
// convertFunc
//

using CvtOpBuilderCb = FuncRef<mlir::Operation*(mlir::OpBuilder&, mlir::Location, mlir::Value, vpux::NDTypeInterface)>;

mlir::LogicalResult convertFunc(mlir::func::FuncOp funcOp, ArrayRef<mlir::Type> newArgTypes,
                                ArrayRef<mlir::Type> newResultTypes, CvtOpBuilderCb cvtOpBuilder,
                                Logger log = Logger::global());

//
// getDefaultGreedyRewriteConfig
//

mlir::GreedyRewriteConfig getDefaultGreedyRewriteConfig();

//
// appendLoc
//

mlir::Location appendLoc(mlir::Location baseLoc, StringRef suffix);
mlir::Location appendLoc(mlir::Location baseLoc, const formatv_object_base& suffix);
mlir::Location appendLoc(mlir::Location baseLoc, mlir::StringAttr suffix);

template <typename Arg0, typename... Args>
mlir::Location appendLoc(mlir::Location baseLoc, StringLiteral format, Arg0&& arg0, Args&&... args) {
    return appendLoc(baseLoc, formatv(format.data(), std::forward<Arg0>(arg0), std::forward<Args>(args)...));
}

//
// takeOpLoc
//

// Equivalent to appendLoc(op->getLoc(), ...)
template <typename... Args>
mlir::Location takeOpLoc(mlir::Operation* op, Args&&... args) {
    return appendLoc(op->getLoc(), std::forward<Args>(args)...);
}

//
// extendOpLoc
//

// Equivalent to op->setLoc(appendLoc(op->getLoc(), ...))
template <typename... Args>
void extendOpLoc(mlir::Operation* op, Args&&... args) {
    const mlir::Location newLoc = takeOpLoc(op, std::forward<Args>(args)...);
    op->setLoc(newLoc);
}

//
// dummyConverter
//

template <class ConcreteType>
mlir::Value dummyConverter(mlir::OpBuilder& builder, ConcreteType type, mlir::ValueRange inputs, mlir::Location loc) {
    SmallVector<mlir::Value> results;
    builder.createOrFold<mlir::UnrealizedConversionCastOp>(results, loc, type, inputs);
    return results.front();
}

//
// BufferizeTypeConverterBase
//

class BufferizeTypeConverterBase : public mlir::TypeConverter {
public:
    BufferizeTypeConverterBase();
};

//
// BufferizeTypeConverter
//

class BufferizeTypeConverter : public BufferizeTypeConverterBase {
public:
    BufferizeTypeConverter();
};

//
// BufferizeOneShotTypeConverter
//

class BufferizeOneShotTypeConverter : public BufferizeTypeConverterBase {
public:
    BufferizeOneShotTypeConverter();
};

//
// getOneShotBufferizationOptions
//

mlir::bufferization::OneShotBufferizationOptions getOneShotBufferizationOptions();

//
// getBufferType
//

vpux::NDTypeInterface getBufferType(mlir::Type type);

// convenience overload that forwards value.getType()
vpux::NDTypeInterface getBufferType(mlir::Value value);

// Converts a buffer type to its "origin" tensor type counterpart.
mlir::Type reconstructTensorType(mlir::Type memrefType);

//
// getBuffer
//

mlir::Value getBuffer(mlir::RewriterBase& rewriter, mlir::Value value);

//
// bufferizeOperands
//

// Converts tensor operands to memrefs (One-Shot Bufferization).
SmallVector<mlir::Value> bufferizeOperands(mlir::RewriterBase& rewriter, mlir::OperandRange operands);

//
// populateBufferizeMaterializationLegality
//

void populateBufferizeMaterializationLegality(mlir::ConversionTarget& target);

//
// inferReturnTypes
//

enum class InferShapedTypeMode : uint32_t {
    SHAPE = 1 << 0,
    ELEM_TYPE = 1 << 1,
    LAYOUT = 1 << 2,
    MEM_SPACE = 1 << 3,

    ALL = std::numeric_limits<uint32_t>::max()
};

inline InferShapedTypeMode operator|(InferShapedTypeMode lhs, InferShapedTypeMode rhs) {
    return static_cast<InferShapedTypeMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline InferShapedTypeMode operator&(InferShapedTypeMode lhs, InferShapedTypeMode rhs) {
    return static_cast<InferShapedTypeMode>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(InferShapedTypeMode bits, InferShapedTypeMode bit) {
    return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

void inferReturnTypes(mlir::Operation* op, InferShapedTypeMode mode);

}  // namespace vpux
