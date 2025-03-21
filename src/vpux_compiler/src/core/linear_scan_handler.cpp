//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

using namespace vpux;

//
// Constructor
//

LinearScanHandler::LinearScanHandler(AddressType defaultAlignment): _defaultAlignment(defaultAlignment) {
}

void LinearScanHandler::markAsDead(mlir::Value val) {
    _aliveValues.erase(val);
}

void LinearScanHandler::markAllBuffersAsDead() {
    _aliveValues.clear();
}

void LinearScanHandler::markAsAlive(mlir::Value val) {
    _aliveValues.insert(val);
}

Byte LinearScanHandler::maxAllocatedSize() const {
    return _maxAllocatedSize;
}

void LinearScanHandler::markAsDynamicSpill(mlir::Value val) {
    _dynamicSpillValues.insert(val);
}

void LinearScanHandler::removeDynamicSpill(mlir::Value val) {
    VPUX_THROW_UNLESS(_dynamicSpillValues.count(val) > 0, "Value '{0}' was not dynamic spilled", val);

    _dynamicSpillValues.erase(val);
}

bool LinearScanHandler::isAlive(mlir::Value val) const {
    return _aliveValues.contains(val);
}

bool LinearScanHandler::isDynamicSpill(mlir::Value val) const {
    return _dynamicSpillValues.contains(val);
}

bool LinearScanHandler::isFixedAlloc(mlir::Value val) {
    return val.getDefiningOp<VPUIP::StaticAllocOp>() != nullptr;
}

AddressType LinearScanHandler::getSize(mlir::Value val) {
    AddressType size;
    const auto it = _sizeCache.find(val);
    if (it == _sizeCache.end()) {
        size = getTotalSize(val).count();
        _sizeCache.insert({val, size});
    } else {
        size = it->second;
    }
    return size;
}

AddressType LinearScanHandler::getAlignment(mlir::Value val) const {
    if (auto allocOp = val.getDefiningOp<mlir::memref::AllocOp>()) {
        if (auto alignment = allocOp.getAlignment()) {
            return checked_cast<AddressType>(alignment.value());
        }
    } else if (auto allocOp = val.getDefiningOp<VPURT::Alloc>()) {
        if (auto alignment = allocOp.getAlignment()) {
            return checked_cast<AddressType>(alignment.value());
        }
    } else if (auto allocOp = val.getDefiningOp<VPURT::AllocDistributed>()) {
        if (auto alignment = allocOp.getAlignment()) {
            return checked_cast<AddressType>(alignment.value());
        }
    }

    return _defaultAlignment;
}

AddressType LinearScanHandler::getAddress(mlir::Value val) const {
    if (auto staticAllocOp = val.getDefiningOp<VPUIP::StaticAllocOp>()) {
        return checked_cast<AddressType>(staticAllocOp.getOffset());
    }

    const auto it = _valOffsets.find(val);
    VPUX_THROW_UNLESS(it != _valOffsets.end(), "Value '{0}' was not allocated", val);

    return it->second;
}

void LinearScanHandler::setAddress(mlir::Value val, AddressType address) {
    const auto it = _valOffsets.find(val);
    if (it == _valOffsets.end()) {
        _valOffsets.insert({val, address});
    } else {
        it->second = address;
    }
}

void LinearScanHandler::allocated(mlir::Value val, AddressType addr) {
    VPUX_THROW_UNLESS(addr != InvalidAddress, "Trying to assign invalid address");
    VPUX_THROW_UNLESS(_valOffsets.count(val) == 0, "Value '{0}' was already allocated", val);

    _valOffsets.insert({val, addr});

    auto allocatedSize = static_cast<int64_t>(addr + getSize(val));
    _maxAllocatedSize = Byte(std::max(_maxAllocatedSize.count(), allocatedSize));
}

void LinearScanHandler::deallocate(mlir::Value val) {
    VPUX_THROW_UNLESS(_valOffsets.count(val) > 0, "Value '{0}' was not allocated", val);

    _valOffsets.erase(val);
}

mlir::DenseSet<mlir::Value> LinearScanHandler::getAliveValues() {
    return _aliveValues;
}

void LinearScanHandler::freed(mlir::Value val) {
    markAsDead(val);
}

int LinearScanHandler::getSpillWeight(mlir::Value) {
    VPUX_THROW("Spills are not implemented");
}

bool LinearScanHandler::spilled(mlir::Value) {
    VPUX_THROW("Spills are not implemented");
}
