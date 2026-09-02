//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/bit.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <optional>

namespace vpux::bytecode {

// Extract statically-known strides from a MemRef and assert the offset is a static zero.
// The bytecode buffer_type encoding cannot represent dynamic offsets or non-zero base offsets.
SmallVector<int64_t> getStridesWithStaticZeroOffset(mlir::MemRefType memrefType, StringRef diagnosticContext);

// i1 is conventionally unsigned (true=1, false=0). Sign-extending a 1-bit APInt would map true
// to -1, which silently breaks JE-based cf.cond_br/cf.switch lowering. Wider integers keep
// sign-extension so e.g. `arith.constant -1 : i64` round-trips as -1 in the i64 payload.
inline int64_t apIntToI64ImmediateBits(const mlir::APInt& value) {
    return value.getBitWidth() == 1 ? static_cast<int64_t>(value.getZExtValue()) : value.getSExtValue();
}

inline std::optional<int64_t> getI64ImmediateValueBits(mlir::TypedAttr attr) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        if (intAttr.getValue().getBitWidth() > 64) {
            return std::nullopt;
        }
        return apIntToI64ImmediateBits(intAttr.getValue());
    }
    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
        const auto floatBits = floatAttr.getValue().bitcastToAPInt();
        if (floatBits.getBitWidth() > 64) {
            return std::nullopt;
        }
        return llvm::bit_cast<int64_t>(floatBits.getZExtValue());
    }
    return std::nullopt;
}

inline mlir::Value materializeI64ImmediateRegister(mlir::OpBuilder& builder, mlir::Location loc, int64_t value) {
    return builder.create<ImmRegisterOp>(loc, builder.getI64IntegerAttr(value)).getResult();
}

inline mlir::Value materializeSymbolIndexRegister(mlir::OpBuilder& builder, mlir::Location loc,
                                                  mlir::SymbolRefAttr symbol) {
    auto registerValue = builder.create<VirtualGeneralRegisterOp>(loc).getResult();
    builder.create<SetImmIdxOp>(loc, registerValue, symbol);
    return registerValue;
}

inline SmallVector<mlir::Value> materializeI64ImmediateRegisters(mlir::OpBuilder& builder, mlir::Location loc,
                                                                 llvm::ArrayRef<int64_t> values) {
    SmallVector<mlir::Value> registerValues;
    registerValues.reserve(values.size());
    for (auto value : values) {
        registerValues.push_back(materializeI64ImmediateRegister(builder, loc, value));
    }
    return registerValues;
}

}  // namespace vpux::bytecode
