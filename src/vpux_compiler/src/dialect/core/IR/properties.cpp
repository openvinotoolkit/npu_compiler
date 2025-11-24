//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/properties.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

//
// std::optional<int64_t> property
//

mlir::LogicalResult vpux::convertFromAttribute(std::optional<int64_t>& prop, mlir::Attribute attr,
                                               llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
    auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr);
    if (arrayAttr == nullptr) {
        emitError() << "expected std::optional<int64_t> property to materialize as array";
        return mlir::failure();
    }
    if (arrayAttr.size() > 1) {
        emitError() << "expected std::optional<int64_t> property to become 0- or 1-element arrays";
        return mlir::failure();
    }
    if (arrayAttr.empty()) {
        prop = std::nullopt;
        return mlir::success();
    }
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(arrayAttr[0]);
    if (intAttr == nullptr) {
        emitError() << "expected element in std::optional<int64_t> property to materialize as integer";
        return mlir::failure();
    }
    prop = intAttr.getValue().getSExtValue();
    return mlir::success();
}

mlir::Attribute vpux::convertToAttribute(mlir::MLIRContext* ctx, const std::optional<int64_t>& prop) {
    if (!prop.has_value()) {
        return mlir::ArrayAttr::get(ctx, {});
    }
    return mlir::ArrayAttr::get(ctx, getIntAttr(ctx, prop.value()));
}

mlir::LogicalResult vpux::readFromMlirBytecode(mlir::DialectBytecodeReader&, std::optional<int64_t>& /*prop*/) {
    VPUX_THROW("MLIR bytecode is not supported for std::optional<int64_t>");
}

void vpux::writeToMlirBytecode(mlir::DialectBytecodeWriter&, const std::optional<int64_t>& /*prop*/) {
    VPUX_THROW("MLIR bytecode is not supported for std::optional<int64_t>");
}

//
// std::optional<bool> property
//

mlir::LogicalResult vpux::convertFromAttribute(std::optional<bool>& prop, mlir::Attribute attr,
                                               llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
    auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr);
    if (arrayAttr == nullptr) {
        emitError() << "expected std::optional<bool> property to materialize as array";
        return mlir::failure();
    }
    if (arrayAttr.size() > 1) {
        emitError() << "expected std::optional<bool> property to become 0- or 1-element arrays";
        return mlir::failure();
    }
    if (arrayAttr.empty()) {
        prop = std::nullopt;
        return mlir::success();
    }
    auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(arrayAttr[0]);
    if (boolAttr == nullptr) {
        emitError() << "expected element in std::optional<bool> property to materialize as boolean";
        return mlir::failure();
    }
    prop = boolAttr.getValue();
    return mlir::success();
}

mlir::Attribute vpux::convertToAttribute(mlir::MLIRContext* ctx, const std::optional<bool>& prop) {
    if (!prop.has_value()) {
        return mlir::ArrayAttr::get(ctx, {});
    }
    return mlir::ArrayAttr::get(ctx, mlir::BoolAttr::get(ctx, prop.value()));
}

mlir::LogicalResult vpux::readFromMlirBytecode(mlir::DialectBytecodeReader&, std::optional<bool>& /*prop*/) {
    VPUX_THROW("MLIR bytecode is not supported for std::optional<bool>");
}

void vpux::writeToMlirBytecode(mlir::DialectBytecodeWriter&, const std::optional<bool>& /*prop*/) {
    VPUX_THROW("MLIR bytecode is not supported for std::optional<bool>");
}

//
// SmallVector<int64_t> property
//

mlir::LogicalResult vpux::convertFromAttribute(SmallVector<uint8_t>& storage, mlir::Attribute attr,
                                               llvm::function_ref<mlir::InFlightDiagnostic()>) {
    auto arrayAttr = llvm::dyn_cast_if_present<::mlir::ArrayAttr>(attr);
    if (!arrayAttr) {
        return mlir::failure();
    }
    storage = parseIntArrayAttr<uint8_t>(arrayAttr);
    return mlir::success();
}

mlir::Attribute vpux::convertToAttribute(mlir::MLIRContext* ctx, ArrayRef<uint8_t> storage) {
    return getIntArrayAttr(ctx, storage);
}

llvm::hash_code vpux::hash_value(ArrayRef<uint8_t> storage) {
    return llvm::hash_value(storage);
}
