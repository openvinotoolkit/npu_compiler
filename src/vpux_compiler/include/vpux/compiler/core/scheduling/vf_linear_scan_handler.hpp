//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/scheduling/classifier_vf.hpp"
#include "vpux/compiler/utils/partitioner.hpp"

#include "vpux/utils/core/dense_map.hpp"

#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/Value.h>

#include <cstdint>

namespace vpux {

//
// LinearScanHandler dedicated for loop allocation scheduling
//
// It differs from the one used directly by memory scheduler passes
// as it requires specific buffer data to be populated up front
//

class VFRegionLinearScanHandler {
public:
    explicit VFRegionLinearScanHandler(AddressType defaultAlignment = 1);

public:
    void markAsDead(mlir::Value val);
    void markAllBuffersAsDead();
    void markAsAlive(mlir::Value val);
    AddressType maxAllocatedSize() const;

public:
    bool isAlive(mlir::Value val) const;
    bool isAllocated(mlir::Value val) const;
    static bool isFixedAlloc(mlir::Value r);
    AddressType getSize(mlir::Value val) const;
    AddressType getAlignment(mlir::Value val) const;
    AddressType getAddress(mlir::Value val) const;
    void setAddress(mlir::Value val, AddressType address);
    void allocated(mlir::Value val, AddressType addr);
    void deallocate(mlir::Value val);
    mlir::DenseSet<mlir::Value> getAliveValues();
    void freed(mlir::Value val);
    int getSpillWeight(mlir::Value val) const;
    bool spilled(mlir::Value val);

public:
    void addBufferData(mlir::Value val, BufferCategory category, AddressType size, AddressType alignment,
                       int spillCost);
    bool wasEverSpilled(mlir::Value val) const;
    BufferCategory getCategory(mlir::Value val) const;

private:
    struct BufferData {
        BufferCategory category = BufferCategory::TEMPORARY;
        AddressType size = 0;
        AddressType alignment = 1;
        int spillCost = 0;
    };

    const BufferData& getBufferData(mlir::Value val) const;

    AddressType _defaultAlignment = 1;
    AddressType _maxAllocatedSize = 0;

    DenseMap<mlir::Value, BufferData> _valDatabase;
    DenseMap<mlir::Value, AddressType> _valOffsets;
    llvm::DenseSet<mlir::Value> _aliveValues;
    llvm::DenseSet<mlir::Value> _evictedValues;
};

}  // namespace vpux
