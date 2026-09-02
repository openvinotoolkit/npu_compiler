//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_cache.hpp"

#include <llvm/ADT/DenseMap.h>

#include <functional>

namespace vpux::VPU::VF::v2 {

class VFConfig final {
public:
    VFConfig(VPU::VerticalFusionOp vfOp, VFCacheAnalysis& cache, bool enableVFPipelining = true,
             bool firstVFNeedsTiling = true, bool secondVFNeedsTiling = true);
    ~VFConfig() = default;

    VFConfig(const llvm::SetVector<mlir::Operation*>& operations, VFCacheAnalysis& cache);

    VFConfig(const VFConfig& other);

    VFConfig(VFConfig&& other);

    VFConfig& operator=(const VFConfig& other);

    VFConfig& operator=(VFConfig&& other);

    // Init vf ops, input/output ops and largest ops
    void init();

    // get original subgraph
    VPU::VerticalFusionOp getSubgraph() const;

    // get the largest operation in the subgraph
    mlir::Operation* getLargestOp() const;

    // get all inputs
    const SmallVector<mlir::Operation*>& getInputs() const;

    // get all outputs
    const SmallVector<mlir::Operation*>& getOutputs() const;

    // get all operations in the subgraph
    const llvm::SetVector<mlir::Operation*>& getVFOperations() const;

    // get all operations in the subgraph
    const SmallVector<mlir::Operation*>& getOperationsForTiling() const;

    // check if subgraph might be pipelined
    bool isPipelined() const;

    // Get cached types for operation in VF
    const SmallVector<NDTypeInterface>& getOperationTypes(mlir::Operation* operation, const TileInfo& outTile,
                                                          const ArrayRef<TileInfo> inputTiles);
    const SmallVector<NDTypeInterface>& getOperationTypes(mlir::Operation* operation);

    Byte getOperationRequiredCMX(mlir::Operation* operation, const TileInfo& outTile,
                                 const ArrayRef<TileInfo> inputTiles);

    std::optional<StrategyCost> getCachedStrategyCost(llvm::hash_code hash) const;
    void cacheStrategyCost(llvm::hash_code hash, StrategyCost cost);

    // returns if first VF needs tiling
    bool firstVFNeedTiling() const;

    // returns if second VF needs tiling
    bool secondVFNeedTiling() const;

private:
    bool isVFPipelinePattern() const;
    void validateConfig() const;
    llvm::hash_code computeOpShapeHash(mlir::Operation* operation, ShapeRef outShape) const;
    llvm::hash_code computeRequiredCMXHash(mlir::Operation* operation, const TileInfo& outTile,
                                           const ArrayRef<TileInfo> inputTiles) const;

    VPU::VerticalFusionOp _subgraph;
    mlir::Operation* _largestOp = nullptr;
    SmallVector<mlir::Operation*> _inputOps;
    SmallVector<mlir::Operation*> _outputOps;
    SmallVector<mlir::Operation*> _tilingOps;
    llvm::SetVector<mlir::Operation*> _vfOps;

    // the pass-level cache
    std::reference_wrapper<VFCacheAnalysis> _cache;
    bool _isVFPipelineCandidate = false;
    bool _isPipelineEnabled = false;
    bool _firstVFNeedsTiling = true;
    bool _secondVFNeedsTiling = true;
};
}  // namespace vpux::VPU::VF::v2
