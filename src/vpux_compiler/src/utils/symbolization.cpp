//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/symbolization.hpp"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>

namespace vpux {

mlir::FlatSymbolRefAttr buildSymbolicName(mlir::Operation* op, mlir::MLIRContext* ctx,
                                          const std::optional<std::string>& taskTypeString,
                                          std::optional<size_t> counter) {
    VPUX_THROW_UNLESS(op->getNumResults() == 1,
                      "Default symbolic converter only supports ops with exactly one result. For {0} got {1}",
                      op->getName(), op->getNumResults());
    constexpr size_t symbolicNameInlineCapacity = 64;

    auto opName = op->getName().stripDialect();

    llvm::SmallString<symbolicNameInlineCapacity> name;
    name.append(opName.begin(), opName.end());

    llvm::raw_svector_ostream os(name);
    if (taskTypeString.has_value()) {
        os << '_' << taskTypeString.value();
    }
    if (auto indexType = mlir::dyn_cast<vpux::VPURegMapped::IndexType>(op->getResult(0).getType())) {
        os << '_' << indexType.getTileIdx();
        os << '_' << indexType.getListIdx();
        os << '_' << indexType.getValue();
    } else if (counter.has_value()) {
        os << '_' << counter.value();
    }
    return mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(ctx, name));
}

}  // namespace vpux
