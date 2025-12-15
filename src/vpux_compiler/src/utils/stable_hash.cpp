//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/stable_hash.hpp"
#include "vpux/utils/core/format.hpp"

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace {
// Note: this is what UniformQuantizedPerAxisTypeStorage::getHashValue() does,
// so has to be good enough for stable hashing.
mlir::ArrayRef<int64_t> castDoublesToInts(mlir::ArrayRef<double> scales) {
    static_assert(sizeof(double) == sizeof(int64_t), "Cannot cast double to bytes");
    auto* scalesCast = llvm::bit_cast<int64_t*>(scales.data());
    return mlir::ArrayRef<int64_t>(scalesCast, scales.size());
}
}  // namespace

namespace vpux {
llvm::hash_code getStableHash(mlir::Type type) {
    // Note: the least evil thing to do is to serialize a type into string or
    // bytecode. bytecode serialization looks complex to implement, thus, use a
    // string instead. while this is (still) horrible, this is sane enough to
    // avoid the need to implement a full-blown facility that supports builtins
    // and dialect-defined types. special cases that are known to be slow are
    // manually dispatched to a better hashing procedure.

    return mlir::TypeSwitch<mlir::Type, llvm::hash_code>(type)
            // Note: quantile per-axis is *also* uniform per-axis
            // (UniformQuantizedPerAxisType is the base type), so type-checking
            // for quantile case must precede uniform case.
            .Case([](mlir::quant::QuantileQuantizedPerAxisType perAxisType) {
                const auto quantiles = perAxisType.getQuantiles();
                const auto scales = perAxisType.getScales();
                return llvm::hash_combine(perAxisType.getFlags(), getStableHash(perAxisType.getStorageType()),
                                          getStableHash(perAxisType.getQuantileType()),
                                          getStableHash(perAxisType.getExpressedType()), castDoublesToInts(quantiles),
                                          castDoublesToInts(scales), perAxisType.getZeroPoints(),
                                          perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin(),
                                          perAxisType.getStorageTypeMax());
            })
            .Case([](mlir::quant::UniformQuantizedPerAxisType perAxisType) {
                assert(!mlir::isa<mlir::quant::QuantileQuantizedPerAxisType>(perAxisType) &&
                       "Cannot hash quantile per-axis as uniform per-axis");

                const auto scales = perAxisType.getScales();
                return llvm::hash_combine(perAxisType.getFlags(), getStableHash(perAxisType.getStorageType()),
                                          getStableHash(perAxisType.getExpressedType()), castDoublesToInts(scales),
                                          perAxisType.getZeroPoints(), perAxisType.getQuantizedDimension(),
                                          perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
            })
            .Default([](mlir::Type t) {
                return llvm::hash_value(formatv("{0}", t).str());
            });
}
}  // namespace vpux
