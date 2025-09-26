//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"

using namespace vpux;

constexpr StringLiteral orderName = "order";
constexpr StringLiteral memSpaceName = "mem_space";
constexpr StringLiteral boundsName = "bounds";
constexpr StringLiteral dynamicDimsMaskName = "dynamic_dims_mask";

template <class T>
bool checkAttr(mlir::DictionaryAttr derived, StringRef attrName, int& numAbsentAttrs) {
    auto attr = derived.get(attrName);
    if (attr == nullptr) {
        ++numAbsentAttrs;
        return true;
    }

    return mlir::isa<T>(attr);
}

bool vpux::TensorAttr::classof(mlir::Attribute attr) {
    if (attr == nullptr) {
        return false;
    }

    auto derived = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
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
    if (!checkAttr<Const::OpaqueI64ElementsAttr>(derived, boundsName, numAbsentAttrs)) {
        return false;
    }
    if (!checkAttr<Const::OpaqueI64ElementsAttr>(derived, dynamicDimsMaskName, numAbsentAttrs)) {
        return false;
    }

    return (derived.size() + numAbsentAttrs) == 4;
}

Const::OpaqueI64ElementsAttr createOpaqueI64ElementsAttr(mlir::MLIRContext* context, ArrayRef<int64_t> bounds) {
    const auto elemType = mlir::IntegerType::get(context, 64, mlir::IntegerType::Signed);
    const auto dataStorageType = mlir::RankedTensorType::get({checked_cast<int64_t>(bounds.size())}, elemType);
    return Const::OpaqueI64ElementsAttr::get(dataStorageType, bounds);
}

TensorAttr vpux::TensorAttr::get(mlir::MLIRContext* context, mlir::AffineMapAttr order,
                                 vpux::IndexedSymbolAttr memSpace, BoundsRef bounds,
                                 DynamicDimsMaskRef dynamicDimsMask) {
    VPUX_THROW_WHEN(!bounds.empty() && !dynamicDimsMask.empty(),
                    "Ambiguous tensor representation. Both Bounds {0} and DynamicDimsMask {1} were provided.", bounds,
                    dynamicDimsMask);

    SmallVector<mlir::NamedAttribute> fields;

    if (order != nullptr) {
        auto orderId = mlir::StringAttr::get(context, orderName);
        fields.emplace_back(orderId, order);
    }

    if (memSpace != nullptr) {
        auto memSpaceId = mlir::StringAttr::get(context, memSpaceName);
        fields.emplace_back(memSpaceId, memSpace);
    }

    if (!bounds.empty()) {
        auto boundsId = mlir::StringAttr::get(context, boundsName);
        const auto opaqueAttr = createOpaqueI64ElementsAttr(context, bounds.raw());
        fields.emplace_back(boundsId, opaqueAttr);
    }

    if (!dynamicDimsMask.empty()) {
        bool onlyZeroOrOne = llvm::all_of(dynamicDimsMask, [](auto value) {
            return value == 0 || value == 1;
        });
        VPUX_THROW_UNLESS(onlyZeroOrOne, "Dynamic dims mask must have only 0 or 1, got {0}", dynamicDimsMask);

        bool allZeroes = llvm::all_of(dynamicDimsMask, [](auto value) {
            return value == 0;
        });
        VPUX_THROW_UNLESS(!allZeroes, "Dynamic dims mask must contain 1's that represent dynamic dimensions, got {0}",
                          dynamicDimsMask);

        auto dynamicDimsMaskId = mlir::StringAttr::get(context, dynamicDimsMaskName);
        const auto opaqueAttr = createOpaqueI64ElementsAttr(context, dynamicDimsMask.raw());
        fields.emplace_back(dynamicDimsMaskId, opaqueAttr);
    }

    auto dict = mlir::DictionaryAttr::get(context, fields);
    return mlir::dyn_cast<vpux::TensorAttr>(dict);
}

template <class T>
T getAttr(mlir::DictionaryAttr derived, StringRef attrName) {
    auto attr = derived.get(attrName);
    if (attr == nullptr) {
        return nullptr;
    }
    VPUX_THROW_WHEN(!mlir::isa<T>(attr), "incorrect {0} Attribute type found: {1}", attrName, attr);
    return mlir::cast<T>(attr);
}

mlir::AffineMapAttr TensorAttr::getOrder() const {
    auto derived = mlir::cast<mlir::DictionaryAttr>(this);
    return getAttr<mlir::AffineMapAttr>(*derived, orderName);
}

vpux::IndexedSymbolAttr TensorAttr::getMemSpace() const {
    auto derived = mlir::cast<mlir::DictionaryAttr>(this);
    return getAttr<vpux::IndexedSymbolAttr>(*derived, memSpaceName);
}

BoundsRef TensorAttr::getBounds() const {
    auto derived = mlir::cast<mlir::DictionaryAttr>(this);
    auto bounds = getAttr<Const::OpaqueI64ElementsAttr>(*derived, boundsName);
    if (bounds != nullptr) {
        return BoundsRef(bounds.getValue());
    }

    return {};
}

DynamicDimsMaskRef TensorAttr::getDynamicDimsMask() const {
    auto derived = mlir::cast<mlir::DictionaryAttr>(this);
    auto dynamicDimsMask = getAttr<Const::OpaqueI64ElementsAttr>(*derived, dynamicDimsMaskName);
    if (dynamicDimsMask != nullptr) {
        return DynamicDimsMaskRef(dynamicDimsMask.getValue());
    }

    return {};
}

//
// Helpers
//

//
// Default getTensorAttr
//

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, mlir::AffineMapAttr order, IndexedSymbolAttr memSpace,
                               BoundsRef bounds, DynamicDimsMaskRef dynamicDimsMask) {
    // Initially, tensors do not have an encoding attribute, which is equivalent to an empty TensorAttr.
    // But in fact, such tensors have a different type: `tensor<1x8x4x2xf16> != tensor<1x8x4x2xf16, {}>`.
    // So let's not use empty attributes to avoid ambiguous representation of the same type.
    if ((order == nullptr || order.getValue().isIdentity()) && memSpace == nullptr && bounds.raw().empty() &&
        dynamicDimsMask.raw().empty()) {
        return nullptr;
    }

    return TensorAttr::get(ctx, order, memSpace, bounds, dynamicDimsMask);
}

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, mlir::AffineMap order, IndexedSymbolAttr memSpace,
                               BoundsRef bounds, DynamicDimsMaskRef dynamicDimsMask) {
    return vpux::getTensorAttr(ctx, mlir::AffineMapAttr::get(order), memSpace, bounds, dynamicDimsMask);
}

TensorAttr vpux::getTensorAttr(mlir::MLIRContext* ctx, DimsOrder order, IndexedSymbolAttr memSpace, BoundsRef bounds,
                               DynamicDimsMaskRef dynamicDimsMask) {
    return vpux::getTensorAttr(ctx, order.toAffineMap(ctx), memSpace, bounds, dynamicDimsMask);
}

TensorAttr vpux::getTensorAttr(mlir::RankedTensorType type) {
    if (const auto encoding = type.getEncoding()) {
        const auto tensorAttr = mlir::dyn_cast<vpux::TensorAttr>(encoding);
        VPUX_THROW_UNLESS(tensorAttr != nullptr, "Unsupported tensor encoding attribute '{0}'", encoding);
        return tensorAttr;
    }

    return nullptr;
}

TensorAttr vpux::getTensorAttr(mlir::RankedTensorType type, mlir::AffineMap order, IndexedSymbolAttr memSpace,
                               mlir::ArrayRef<int64_t> dynamicAttr) {
    if (mlir::isa<Core::BoundedTensorType>(type)) {
        return vpux::getTensorAttr(type.getContext(), order, memSpace, BoundsRef(dynamicAttr));
    }
    if (mlir::isa<Core::DynamicDimsMaskTensorType>(type)) {
        return vpux::getTensorAttr(type.getContext(), order, memSpace, /*Bounds=*/{}, DynamicDimsMaskRef(dynamicAttr));
    }
    return vpux::getTensorAttr(type.getContext(), order, memSpace);
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

BoundsRef vpux::getBounds(mlir::Type type) {
    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(type)) {
        return boundedType.getBounds();
    }

    return {};
};

DynamicDimsMaskRef vpux::getDynamicDimsMask(mlir::Type type) {
    if (auto dynamicDimsMaskType = mlir::dyn_cast<Core::DynamicDimsMaskTensorType>(type)) {
        return dynamicDimsMaskType.getDynamicDimsMask();
    }

    return {};
}
