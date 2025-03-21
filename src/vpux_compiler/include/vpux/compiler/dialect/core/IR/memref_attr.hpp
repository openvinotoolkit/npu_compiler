//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/core/interfaces/attr_interfaces.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Diagnostics.h>

#include <type_traits>

namespace vpux {

//
// MemRefAttr
//

class MemRefAttr : public mlir::DictionaryAttr {
public:
    using mlir::DictionaryAttr::DictionaryAttr;

    // Note: this value is used to optimize the storage for the HW-specific
    // fields of the MemRefAttr. Update this constant when you see that the real
    // maximum of HW-specific fields is larger than this value.
    static constexpr size_t maxCountOfHwSpecificFields = 8;
    using HwFields = mlir::SmallVector<vpux::HwSpecificMemRefField, MemRefAttr::maxCountOfHwSpecificFields>;

    static bool classof(mlir::Attribute attr);

    // E#104246: Consider removing this implicit conversion
    operator mlir::MemRefLayoutAttrInterface() const {
        return mlir::cast<mlir::MemRefLayoutAttrInterface>(*this);
    }

    static MemRefAttr get(mlir::AffineMapAttr order, mlir::ArrayAttr optionalStrides,
                          mlir::IntegerAttr optionalAllocSize, mlir::MLIRContext* ctx) {
        return MemRefAttr::get(order, optionalStrides, optionalAllocSize, {}, ctx);
    }

    static MemRefAttr get(mlir::AffineMapAttr order, mlir::ArrayAttr optionalStrides,
                          mlir::IntegerAttr optionalAllocSize,
                          mlir::ArrayRef<vpux::HwSpecificMemRefField> hwSpecificFields, mlir::MLIRContext* ctx);

    mlir::AffineMapAttr order() const;
    mlir::ArrayAttr strides() const;
    mlir::IntegerAttr allocSize() const;
    HwFields hwSpecificFields() const;

    /// Returns a HW-specific field of the specified type from the MemRefAttr.
    template <typename AttributeType>
    AttributeType hwSpecificField() const {
        static_assert(std::is_base_of_v<vpux::HwSpecificMemRefField::Trait<AttributeType>, AttributeType>,
                      "AttributeType must implement HwSpecificMemRefFieldInterface");
        auto field = hwSpecificField(vpux::HwSpecificMemRefField::Model<AttributeType>::memRefKey());
        return llvm::cast_if_present<AttributeType>(field);
    }

private:
    mlir::Attribute hwSpecificField(mlir::StringRef key) const;
};

//
// MemRefAttrLayout
//

class MemRefAttrLayout final : public mlir::MemRefLayoutAttrInterface::ExternalModel<MemRefAttrLayout, MemRefAttr> {
public:
    using ConcreteEntity = mlir::DictionaryAttr;

    mlir::AffineMap getAffineMap(mlir::Attribute attr) const;

    bool isIdentity(mlir::Attribute) const {
        return false;
    }

    mlir::LogicalResult verifyLayout(mlir::Attribute attr, mlir::ArrayRef<int64_t> shape,
                                     FuncRef<mlir::InFlightDiagnostic()> emitError) const;
};

}  // namespace vpux
