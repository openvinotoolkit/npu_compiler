//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include "mlir/IR/QuantStorageTypeInterface.h"
#include "vpux/compiler/core/types/quantile_float/type_detail.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/array_ref.hpp"

namespace vpux {
namespace type {

//===----------------------------------------------------------------------===//
// QuantileType
//===----------------------------------------------------------------------===//

class QuantileType :
        public mlir::Type::TypeBase<QuantileType, mlir::Type, vpux::detail::QuantileTypeStorage,
                                    mlir::QuantStorageTypeInterface::Trait> {
public:
    using ImplType = vpux::detail::QuantileTypeStorage;
    using Base::Base;

    // Get the underlying type used for to store raw values.
    Type getStorageType() const;

    // Get primitive expressed type of data in quantiles.
    // Note that we may convert FP8 data to FP16 for storage,
    // but we should treat its expressed type as FP8 rather than FP16.
    Type getQuantileType() const;

    /// Return the quantile table of this float type.
    ArrayRef<double> getQuantiles() const;

    // Get a quantile float type with specified quantile table.
    static QuantileType get(mlir::MLIRContext* ctx, Type storageType, Type quantileType,
                            ArrayRef<double> quantiles = {});

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool classof(mlir::Type type);

    // Printer
    void print(mlir::AsmPrinter& printer) const;

    // Parser
    static mlir::Type parse(mlir::AsmParser& parser);

    static constexpr llvm::StringLiteral getMnemonic() {
        return {"quantile"};
    }

    static constexpr llvm::StringLiteral name = "quantile";

    // Returns true if the type defaults to signed (e.g., si8, i8 or float types), false otherwise
    bool shouldDefaultToSigned() const;

    // Get the bit width of the storage type.
    unsigned getStorageWidth() const;

    // Get the default minimum and maximum values for the storage type.
    int64_t getDefaultMinimum([[maybe_unused]] bool isSigned) const;
    int64_t getDefaultMaximum([[maybe_unused]] bool isSigned) const;

    // Get the string representation of the storage type
    std::string getStorageTypeName([[maybe_unused]] bool isSigned) const;

    // Get whether the type is a packed quantile float type
    bool isPacked() const;

    // Get the logical bit width of the quantile float type, which is the bit width of the represented floating point
    // value.
    unsigned getLogicalBitWidth() const;

    // Get the number of quantized values stored in one byte for this quantile float type.
    unsigned getElementsPerByte() const;

    // Get the preferred alignment in bytes for this quantile float type, if any.
    std::optional<unsigned> getPreferredAlignmentBytes() const;
};

/// NF4 quantile type (f4 storage, 4-bit packed quantile)
class NF4Type :
        public mlir::Type::TypeBase<NF4Type, QuantileType, vpux::detail::QuantileTypeStorage,
                                    mlir::QuantStorageTypeInterface::Trait> {
private:
    static const SmallVector<double> specQuantiles;

public:
    using Base::Base;
    static NF4Type get(mlir::MLIRContext* ctx, Type storageType, Type quantileType, ArrayRef<double> quantiles = {});
    static constexpr llvm::StringLiteral name = "nf4";
    static ArrayRef<double> getSpecQuantiles();
    static bool classof(mlir::Type type);
};
}  // namespace type
}  // namespace vpux
