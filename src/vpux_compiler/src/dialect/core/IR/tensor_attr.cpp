//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;

constexpr StringLiteral orderName = "order";
constexpr StringLiteral memSpaceName = "mem_space";
constexpr StringLiteral boundsName = "bounds";

template <class T>
bool checkAttr(mlir::DictionaryAttr derived, StringRef attrName, int& numAbsentAttrs) {
    auto attr = derived.get(attrName);
    if (attr == nullptr) {
        ++numAbsentAttrs;
        return true;
    }

    return attr.isa<T>();
}

bool vpux::TensorAttr::classof(mlir::Attribute attr) {
    if (attr == nullptr) {
        return false;
    }

    auto derived = attr.dyn_cast<mlir::DictionaryAttr>();
    if (derived == nullptr) {
        return false;
    }

    int numAbsentAttrs = 0;

    if (!checkAttr<mlir::AffineMapAttr>(derived, orderName, numAbsentAttrs)) {
        return false;
    }
    if (!checkAttr<vpux::IndexedSymbolAttr>(derived, memSpaceName, numAbsentAttrs)) {
        return false;
    }
    if (!checkAttr<mlir::ArrayAttr>(derived, boundsName, numAbsentAttrs)) {
        return false;
    }

    return (derived.size() + numAbsentAttrs) == 3;
}

TensorAttr vpux::TensorAttr::get(mlir::MLIRContext* context, mlir::AffineMapAttr order,
                                 vpux::IndexedSymbolAttr memSpace, mlir::ArrayAttr bounds) {
    SmallVector<mlir::NamedAttribute> fields;

    if (order != nullptr) {
        auto orderId = mlir::StringAttr::get(context, orderName);
        fields.emplace_back(orderId, order);
    }

    if (memSpace != nullptr) {
        auto memSpaceId = mlir::StringAttr::get(context, memSpaceName);
        fields.emplace_back(memSpaceId, memSpace);
    }

    if (bounds != nullptr) {
        auto boundsId = mlir::StringAttr::get(context, boundsName);
        fields.emplace_back(boundsId, bounds);
    }

    auto dict = mlir::DictionaryAttr::get(context, fields);
    return dict.dyn_cast<TensorAttr>();
}

template <class T>
T getAttr(mlir::DictionaryAttr derived, StringRef attrName) {
    auto attr = derived.get(attrName);
    if (attr == nullptr) {
        return nullptr;
    }
    VPUX_THROW_WHEN(!attr.isa<T>(), "incorrect {0} Attribute type found: {1}", attrName, attr);
    return attr.cast<T>();
}

mlir::AffineMapAttr TensorAttr::getOrder() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    return getAttr<mlir::AffineMapAttr>(derived, orderName);
}

vpux::IndexedSymbolAttr TensorAttr::getMemSpace() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    return getAttr<vpux::IndexedSymbolAttr>(derived, memSpaceName);
}

mlir::ArrayAttr TensorAttr::getBounds() const {
    auto derived = this->cast<mlir::DictionaryAttr>();
    return getAttr<mlir::ArrayAttr>(derived, boundsName);
}

//
// Helpers
//

TensorAttr vpux::getTensorAttr(mlir::AffineMapAttr order, IndexedSymbolAttr memSpace, mlir::ArrayAttr bounds) {
    // Initially, tensors do not have an encoding attribute, which is equivalent to an empty TensorAttr.
    // But in fact, such tensors have a different type: `tensor<1x8x4x2xf16> != tensor<1x8x4x2xf16, {}>`.
    // So let's not use empty attributes to avoid ambiguous representation of the same type.
    if ((order == nullptr || order.getValue().isIdentity()) && memSpace == nullptr && bounds == nullptr) {
        return nullptr;
    }

    mlir::MLIRContext* ctx;
    if (order != nullptr) {
        ctx = order.getContext();
    } else if (memSpace != nullptr) {
        ctx = memSpace.getContext();
    } else {
        ctx = bounds.getContext();
    }

    return TensorAttr::get(ctx, order, memSpace, bounds);
}

TensorAttr vpux::getTensorAttr(mlir::AffineMap order, IndexedSymbolAttr memSpace, mlir::ArrayAttr bounds) {
    return vpux::getTensorAttr(mlir::AffineMapAttr::get(order), memSpace, bounds);
}

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order, IndexedSymbolAttr memSpace,
                               mlir::ArrayAttr bounds) {
    return vpux::getTensorAttr(order.toAffineMap(ctx), memSpace, bounds);
}

TensorAttr vpux::getTensorAttr(mlir::RankedTensorType type) {
    if (const auto encoding = type.getEncoding()) {
        const auto tensorAttr = encoding.dyn_cast<TensorAttr>();
        VPUX_THROW_UNLESS(tensorAttr != nullptr, "Unsupported tensor encoding attribute '{0}'", encoding);
        return tensorAttr;
    }

    return nullptr;
}

mlir::AffineMap vpux::getOrder(mlir::RankedTensorType type) {
    if (const auto desc = vpux::getTensorAttr(type)) {
        if (const auto orderAttr = desc.getOrder()) {
            return orderAttr.getValue();
        }
    }

    const auto numDims = checked_cast<uint32_t>(type.getRank());
    return mlir::AffineMap::getMinorIdentityMap(numDims, numDims, type.getContext());
}

IndexedSymbolAttr vpux::getMemorySpace(mlir::RankedTensorType type) {
    if (const auto desc = vpux::getTensorAttr(type)) {
        return desc.getMemSpace();
    }

    return nullptr;
}

mlir::ArrayAttr vpux::getBounds(mlir::Value value) {
    if (value == nullptr) {
        return nullptr;
    }

    const auto type = value.getType();
    if (const auto boundedType = mlir::dyn_cast_or_null<BoundedTypeInterface>(type)) {
        return boundedType.getBounds();
    }

    return nullptr;
}

mlir::ArrayAttr vpux::getBounds(mlir::RankedTensorType type) {
    if (const auto desc = vpux::getTensorAttr(type)) {
        return desc.getBounds();
    }

    return nullptr;
}
