//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool vpux::IndexedSymbolAttr::classof(mlir::Attribute attr) {
    if (attr.isa<mlir::FlatSymbolRefAttr>()) {
        return true;
    }

    auto indexedSym = attr.dyn_cast<mlir::ArrayAttr>();
    if (indexedSym == nullptr) {
        return false;
    }

    if (indexedSym.empty() || indexedSym.size() > 3) {
        return false;
    }

    if (!indexedSym[0].isa<mlir::FlatSymbolRefAttr>()) {
        return false;
    }

    if (indexedSym.size() == 1) {
        return true;
    }

    if (indexedSym.size() == 2) {
        return indexedSym[1].isa<mlir::IntegerAttr>() || indexedSym[1].isa<IndexedSymbolAttr>();
    }

    return indexedSym[1].isa<mlir::IntegerAttr>() && indexedSym[2].isa<IndexedSymbolAttr>();
}

IndexedSymbolAttr vpux::IndexedSymbolAttr::get(mlir::MLIRContext* context, ArrayRef<mlir::Attribute> array) {
    VPUX_THROW_UNLESS(!array.empty(), "Can not create empty indexed symbol attribute");

    if (array.size() == 1) {
        auto flatSymbolRef = array[0].dyn_cast_or_null<mlir::FlatSymbolRefAttr>();
        VPUX_THROW_UNLESS(flatSymbolRef != nullptr, "Unsupported symbol attr type '{0}'", array[0]);

        return flatSymbolRef.cast<IndexedSymbolAttr>();
    }

    auto arrayAttr = mlir::ArrayAttr::get(context, array);
    VPUX_THROW_UNLESS(IndexedSymbolAttr::classof(arrayAttr),
                      "Array content does not match with IndexedSymbolAttr type '{0}'", array);

    return arrayAttr.cast<IndexedSymbolAttr>();
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
    if (isa<mlir::FlatSymbolRefAttr>()) {
        return std::nullopt;
    }

    auto arrayAttr = cast<mlir::ArrayAttr>();
    if (arrayAttr.size() == 1) {
        return std::nullopt;
    }

    auto symIdx = arrayAttr.size() > 2 ? 2 : 1;
    if (auto symAttr = arrayAttr[symIdx].dyn_cast<IndexedSymbolAttr>()) {
        return symAttr;
    }

    return std::nullopt;
}

mlir::FlatSymbolRefAttr vpux::IndexedSymbolAttr::getRootReference() const {
    if (isa<mlir::FlatSymbolRefAttr>()) {
        return cast<mlir::FlatSymbolRefAttr>();
    }

    auto arrayAttr = cast<mlir::ArrayAttr>();
    return arrayAttr[0].dyn_cast<mlir::FlatSymbolRefAttr>();
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
    if (isa<mlir::FlatSymbolRefAttr>()) {
        return std::nullopt;
    }

    const auto arrayAttr = cast<mlir::ArrayAttr>();

    if (arrayAttr.size() == 1) {
        return std::nullopt;
    }

    if (auto idxAttr = arrayAttr[1].dyn_cast<mlir::IntegerAttr>()) {
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
