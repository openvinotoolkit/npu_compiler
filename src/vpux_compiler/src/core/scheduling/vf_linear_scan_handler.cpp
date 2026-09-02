//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/vf_linear_scan_handler.hpp"

#include "vpux/utils/core/error.hpp"

#include <cassert>

using namespace vpux;

//
// Constructor
//

VFRegionLinearScanHandler::VFRegionLinearScanHandler(AddressType defaultAlignment)
        : _defaultAlignment(defaultAlignment == 0 ? 1 : defaultAlignment), _maxAllocatedSize(0) {
}

void VFRegionLinearScanHandler::markAsDead(mlir::Value val) {
    _aliveValues.erase(val);
}

void VFRegionLinearScanHandler::markAllBuffersAsDead() {
    _aliveValues.clear();
}

void VFRegionLinearScanHandler::markAsAlive(mlir::Value val) {
    _aliveValues.insert(val);
}

AddressType VFRegionLinearScanHandler::maxAllocatedSize() const {
    return _maxAllocatedSize;
}

bool VFRegionLinearScanHandler::isAlive(mlir::Value val) const {
    return _aliveValues.contains(val);
}

bool VFRegionLinearScanHandler::isAllocated(mlir::Value val) const {
    return _valOffsets.count(val) > 0;
}

bool VFRegionLinearScanHandler::isFixedAlloc(mlir::Value) {
    return false;
}

AddressType VFRegionLinearScanHandler::getSize(mlir::Value val) const {
    return getBufferData(val).size;
}

AddressType VFRegionLinearScanHandler::getAlignment(mlir::Value val) const {
    const auto a = getBufferData(val).alignment;
    return a == 0 ? _defaultAlignment : a;
}

AddressType VFRegionLinearScanHandler::getAddress(mlir::Value val) const {
    const auto it = _valOffsets.find(val);
    VPUX_THROW_UNLESS(it != _valOffsets.end(), "Value '{0}' was not allocated", val);

    return it->second;
}

void VFRegionLinearScanHandler::setAddress(mlir::Value val, AddressType address) {
    const auto it = _valOffsets.find(val);
    if (it == _valOffsets.end()) {
        _valOffsets.insert({val, address});
    } else {
        it->second = address;
    }
}

void VFRegionLinearScanHandler::allocated(mlir::Value val, AddressType addr) {
    setAddress(val, addr);
    const auto end = addr + getBufferData(val).size;
    if (end > _maxAllocatedSize) {
        _maxAllocatedSize = end;
    }
}

void VFRegionLinearScanHandler::deallocate(mlir::Value val) {
    VPUX_THROW_UNLESS(_valOffsets.count(val) > 0, "Value '{0}' was not allocated", val);

    _valOffsets.erase(val);
}

mlir::DenseSet<mlir::Value> VFRegionLinearScanHandler::getAliveValues() {
    return _aliveValues;
}

void VFRegionLinearScanHandler::freed(mlir::Value val) {
    markAsDead(val);
}

int VFRegionLinearScanHandler::getSpillWeight(mlir::Value val) const {
    return getBufferData(val).spillCost;
}

bool VFRegionLinearScanHandler::spilled(mlir::Value val) {
    _evictedValues.insert(val);
    return true;
}

void VFRegionLinearScanHandler::addBufferData(mlir::Value val, BufferCategory category, AddressType size,
                                              AddressType alignment, int spillCost) {
    VPUX_THROW_WHEN(!val, "VFRegionLinearScanHandler::addBufferData: null value");
    VPUX_THROW_WHEN(size == 0, "VFRegionLinearScanHandler::addBufferData: size must be > 0");
    BufferData bufferData;
    bufferData.category = category;
    bufferData.size = size;
    bufferData.alignment = alignment == 0 ? _defaultAlignment : alignment;
    bufferData.spillCost = spillCost;
    _valDatabase[val] = bufferData;
}

bool VFRegionLinearScanHandler::wasEverSpilled(mlir::Value val) const {
    return _evictedValues.find(val) != _evictedValues.end();
}

BufferCategory VFRegionLinearScanHandler::getCategory(mlir::Value val) const {
    return getBufferData(val).category;
}

const VFRegionLinearScanHandler::BufferData& VFRegionLinearScanHandler::getBufferData(mlir::Value val) const {
    const auto it = _valDatabase.find(val);
    VPUX_THROW_WHEN(it == _valDatabase.end(),
                    "VFRegionLinearScanHandler: no data for value (was addBufferData called?)");
    return it->second;
}
