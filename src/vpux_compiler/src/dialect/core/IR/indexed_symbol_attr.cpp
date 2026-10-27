//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool vpux::IndexedSymbolAttr::classof(mlir::Attribute attr) {
    if (mlir::isa<mlir::FlatSymbolRefAttr>(attr)) {
        return true;
    }

    auto indexedSym = mlir::dyn_cast<mlir::ArrayAttr>(attr);
    if (indexedSym == nullptr) {
        return false;
    }

    if (indexedSym.empty() || indexedSym.size() > 3) {
        return false;
    }

    if (!mlir::isa<mlir::FlatSymbolRefAttr>(indexedSym[0])) {
        return false;
    }

    if (indexedSym.size() == 1) {
        return true;
    }

    if (indexedSym.size() == 2) {
        return mlir::isa<mlir::IntegerAttr>(indexedSym[1]) || mlir::isa<vpux::IndexedSymbolAttr>(indexedSym[1]);
    }

    return mlir::isa<mlir::IntegerAttr>(indexedSym[1]) && mlir::isa<vpux::IndexedSymbolAttr>(indexedSym[2]);
}

IndexedSymbolAttr vpux::IndexedSymbolAttr::get(mlir::MLIRContext* context, ArrayRef<mlir::Attribute> array) {
    VPUX_THROW_WHEN(array.empty(), "Can not create empty indexed symbol attribute");

    if (array.size() == 1) {
        auto flatSymbolRef = mlir::dyn_cast_or_null<mlir::FlatSymbolRefAttr>(array[0]);
        VPUX_THROW_WHEN(flatSymbolRef == nullptr, "Unsupported symbol attr type '{0}'", array[0]);

        return mlir::cast<vpux::IndexedSymbolAttr>(flatSymbolRef);
    }

    auto arrayAttr = mlir::ArrayAttr::get(context, array);
    VPUX_THROW_UNLESS(IndexedSymbolAttr::classof(arrayAttr),
                      "Array content does not match with IndexedSymbolAttr type '{0}'", array);

    return mlir::cast<vpux::IndexedSymbolAttr>(arrayAttr);
}

IndexedSymbolAttr vpux::IndexedSymbolAttr::get(mlir::MLIRContext* context, StringRef name) {
    return get(context, {mlir::FlatSymbolRefAttr::get(context, name)});
}

IndexedSymbolAttr vpux::IndexedSymbolAttr::get(mlir::MLIRContext* context, StringRef name, size_t id) {
    return get(context, {mlir::FlatSymbolRefAttr::get(context, name), getIntAttr(context, checked_cast<int64_t>(id))});
}

IndexedSymbolAttr vpux::IndexedSymbolAttr::get(mlir::StringAttr name) {
    return get(name.getContext(), {mlir::FlatSymbolRefAttr::get(name)});
}

IndexedSymbolAttr vpux::IndexedSymbolAttr::get(mlir::StringAttr name, size_t id) {
    auto* context = name.getContext();
    return get(context, {mlir::FlatSymbolRefAttr::get(name), getIntAttr(context, checked_cast<int64_t>(id))});
}

std::optional<IndexedSymbolAttr> vpux::IndexedSymbolAttr::getNestedReference() const {
    if (mlir::isa<mlir::FlatSymbolRefAttr>(*this)) {
        return std::nullopt;
    }

    auto arrayAttr = mlir::cast<mlir::ArrayAttr>(*this);
    if (arrayAttr.size() == 1) {
        return std::nullopt;
    }

    auto symIdx = arrayAttr.size() > 2 ? 2 : 1;
    if (auto symAttr = mlir::dyn_cast<vpux::IndexedSymbolAttr>(arrayAttr[symIdx])) {
        return symAttr;
    }

    return std::nullopt;
}

mlir::FlatSymbolRefAttr vpux::IndexedSymbolAttr::getRootReference() const {
    if (mlir::isa<mlir::FlatSymbolRefAttr>(*this)) {
        return mlir::cast<mlir::FlatSymbolRefAttr>(*this);
    }

    auto arrayAttr = mlir::cast<mlir::ArrayAttr>(*this);
    return mlir::dyn_cast<mlir::FlatSymbolRefAttr>(arrayAttr[0]);
}

mlir::StringAttr vpux::IndexedSymbolAttr::getRootNameAttr() const {
    return getRootReference().getAttr();
}

StringRef vpux::IndexedSymbolAttr::getRootName() const {
    return getRootReference().getValue();
}

mlir::FlatSymbolRefAttr vpux::IndexedSymbolAttr::getLeafReference() const {
    if (const auto nested = getNestedReference()) {
        return nested->getLeafReference();
    }

    return getRootReference();
}

mlir::StringAttr vpux::IndexedSymbolAttr::getLeafNameAttr() const {
    return getLeafReference().getAttr();
}

StringRef vpux::IndexedSymbolAttr::getLeafName() const {
    return getLeafReference().getValue();
}

mlir::SymbolRefAttr vpux::IndexedSymbolAttr::getFullReference() const {
    const auto rootRef = getRootReference();

    SmallVector<mlir::FlatSymbolRefAttr> nestedRefs;
    for (auto nested = getNestedReference(); nested.has_value(); nested = nested->getNestedReference()) {
        nestedRefs.push_back(nested->getRootReference());
    }

    return mlir::SymbolRefAttr::get(rootRef.getAttr(), nestedRefs);
}

std::optional<mlir::IntegerAttr> vpux::IndexedSymbolAttr::getIndexAttr() const {
    if (mlir::isa<mlir::FlatSymbolRefAttr>(*this)) {
        return std::nullopt;
    }

    const auto arrayAttr = mlir::cast<mlir::ArrayAttr>(*this);

    if (arrayAttr.size() == 1) {
        return std::nullopt;
    }

    if (auto idxAttr = mlir::dyn_cast<mlir::IntegerAttr>(arrayAttr[1])) {
        return idxAttr;
    }

    return std::nullopt;
}

std::optional<int64_t> vpux::IndexedSymbolAttr::getIndex() const {
    if (const auto indAttr = getIndexAttr()) {
        return indAttr->getInt();
    }

    return std::nullopt;
}
