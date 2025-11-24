//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux::VPU::VF::v2 {

class VFConfig final {
public:
    VFConfig(VPU::VerticalFusionOp vfOp, bool enableVFPipelining = true, bool firstVFNeedsTiling = true,
             bool secondVFNeedsTiling = true);
    ~VFConfig() = default;

    VFConfig(const llvm::SetVector<mlir::Operation*>& operations);

    // get original subgraph
    VPU::VerticalFusionOp getSubgraph() const;

    // get the largest operation in the subgraph
    mlir::Operation* getLargestOp();

    // get all inputs
    const SmallVector<mlir::Operation*>& getInputs();

    // get all outputs
    const SmallVector<mlir::Operation*>& getOutputs();

    // get all oeprations in the subgraph
    const llvm::SetVector<mlir::Operation*>& getVFOperations();

    // get all oeprations in the subgraph
    SmallVector<mlir::Operation*> getOperationsForTiling();

    // check if subgraph might be pipelined
    bool isPipelined() const;

    // Reset cached data
    void invalidatePointers();

    // Get cached types for operation in VF
    SmallVector<NDTypeInterface> getOperationTypes(mlir::Operation* operation, const TileInfo& outTile,
                                                   const ArrayRef<TileInfo> inputTiles);
    SmallVector<NDTypeInterface> getOperationTypes(mlir::Operation* operation);

    // returns if first VF needs tiling
    bool firstVFNeedTiling() const;

    // returns if second VF needs tiling
    bool secondVFNeedTiling() const;

private:
    bool isVFPipelinePattern();
    void validateConfig();

    VPU::VerticalFusionOp _subgraph;
    mlir::Operation* _largestOp = nullptr;
    SmallVector<mlir::Operation*> _inputOps;
    SmallVector<mlir::Operation*> _outputOps;
    llvm::SetVector<mlir::Operation*> _vfOps;
    bool _isVFPipelineCandidate = false;
    bool _isPipelineEnabled = false;
    bool _firstVFNeedsTiling = true;
    bool _secondVFNeedsTiling = true;

    DenseMap<mlir::Operation*, std::map<Shape, SmallVector<NDTypeInterface>>> _tilesCache;
};
}  // namespace vpux::VPU::VF::v2
