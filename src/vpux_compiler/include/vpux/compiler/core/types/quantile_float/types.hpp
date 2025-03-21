// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include "vpux/compiler/core/types/quantile_float/type_detail.hpp"
#include "vpux/utils/core/array_ref.hpp"

using namespace mlir;
using namespace llvm;

namespace vpux {
namespace type {

//===----------------------------------------------------------------------===//
// QuantileFloatType
//===----------------------------------------------------------------------===//

class QuantileFloatType : public mlir::Type {
public:
    using Type::Type;

    /// Return the bitwidth of the storage type.
    unsigned getWidth() const;

    // Get NF4 instance.
    static QuantileFloatType getNF4(mlir::MLIRContext* ctx, ArrayRef<double> quantiles = {});

    /// Return the quantile table of this float type.
    ArrayRef<double> getQuantiles() const;

    // Get a quantile float type with specified quantile table.
    static QuantileFloatType getQuantileFloat(MLIRContext* ctx, unsigned bitWidth, ArrayRef<double> quantiles = {});

    /// Methods for support type inquiry through isa, cast, and dyn_cast.
    static bool classof(mlir::Type type);

    // Printer
    void print(mlir::AsmPrinter& printer) const;

    // Parser
    static mlir::Type parse(mlir::AsmParser& parser);

    static constexpr llvm::StringLiteral getMnemonic() {
        return {"quantileFloat"};
    }
};

class NF4Type : public mlir::Type::TypeBase<NF4Type, QuantileFloatType, vpux::detail::QuantileFloatTypeStorage> {
public:
    using Base::Base;
    static NF4Type get(mlir::MLIRContext* context, unsigned width, ArrayRef<double> quantiles);
    static constexpr llvm::StringLiteral name = "nf4";

    unsigned getWidth() const;
    ArrayRef<double> getQuantiles() const;
};

}  // namespace type
}  // namespace vpux
