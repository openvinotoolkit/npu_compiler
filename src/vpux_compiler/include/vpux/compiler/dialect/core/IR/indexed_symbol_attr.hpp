//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {

class IndexedSymbolAttr : public mlir::Attribute {
public:
    using mlir::Attribute::Attribute;

public:
    static bool classof(mlir::Attribute attr);

public:
    static IndexedSymbolAttr get(mlir::MLIRContext* context, ArrayRef<mlir::Attribute> array);
    static IndexedSymbolAttr get(mlir::MLIRContext* context, StringRef name);
    static IndexedSymbolAttr get(mlir::MLIRContext* context, StringRef name, size_t id);
    static IndexedSymbolAttr get(mlir::StringAttr name);
    static IndexedSymbolAttr get(mlir::StringAttr name, size_t id);

public:
    std::optional<IndexedSymbolAttr> getNestedReference() const;

public:
    mlir::FlatSymbolRefAttr getRootReference() const;
    mlir::StringAttr getRootNameAttr() const;
    StringRef getRootName() const;

    mlir::FlatSymbolRefAttr getLeafReference() const;
    mlir::StringAttr getLeafNameAttr() const;
    StringRef getLeafName() const;

    mlir::SymbolRefAttr getFullReference() const;

public:
    std::optional<mlir::IntegerAttr> getIndexAttr() const;
    std::optional<int64_t> getIndex() const;
};

}  // namespace vpux
