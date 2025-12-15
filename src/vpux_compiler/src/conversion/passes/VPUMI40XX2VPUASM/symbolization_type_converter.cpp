//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPUMI40XX2VPUASM/symbolization_type_converter.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

namespace {

template <typename Type>
std::optional<mlir::LogicalResult> doNotConvert(Type type, llvm::SmallVectorImpl<mlir::Type>& results) {
    results.push_back(type);
    return mlir::success();
}

}  // namespace

SymbolizationTypeConverter::SymbolizationTypeConverter() {
    addConversion(doNotConvert<VPURegMapped::IndexType>);
    addConversion(doNotConvert<mlir::MemRefType>);
    addConversion(doNotConvert<VPUIP::DistributedBufferType>);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
