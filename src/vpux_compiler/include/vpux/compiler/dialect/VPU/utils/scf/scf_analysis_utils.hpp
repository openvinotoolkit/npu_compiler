//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/dialect_processors.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/TilingInterface.h>
#include <algorithm>
#include <cstdint>
#include <memory>

namespace vpux::VPU {

// Constants for container sizes
constexpr size_t DEFAULT_ARG_SET_SIZE = 16;

// Attribute name for marking operations to be analyzed
constexpr llvm::StringLiteral ANALYZE_ATTR = "analyze";

// Attribute name for overriding loop upper bound
constexpr llvm::StringLiteral UPPERBOUND_ATTR = "upperbound";

// Attribute name for setting unique static blocks count
constexpr llvm::StringLiteral UNIQUE_STATIC_BLOCKS_ATTR = "uniqueStaticBlocks";

using ValueRangeMap = llvm::DenseMap<mlir::Value, SmallVector<int64_t>>;

struct InputRange {
    mlir::Value arg;
    SmallVector<int64_t> values;
    int64_t stride;
};

// Bound resolver installed by OpChainAnalysis for a tensor.extract whose source is not a constant
TensorBoundResolver getInterpolateScalesBoundResolver();

/**
 * @brief Utility class for analyzing and processing operation chains in MLIR
 *
 * The OpChainAnalysis class provides functionality to collect and evaluate
 * chains of operations in MLIR. It helps with tracking dependencies between
 * operations and computing values from OpFoldResult objects within the context
 * of various transformations (affine, arithmetic, SCF, etc.).
 *
 * Key features:
 * - Collects chains of related operations from a given value
 * - Evaluates OpFoldResult values with optional bounded shape considerations
 * - Caches operation chains for performance optimization
 * - Provides utilities for extracting maps and operands from operations
 *
 */
class OpChainAnalysis {
public:
    explicit OpChainAnalysis(const Logger& log = Logger::global().nest("opchain-analysis-utils"));

    // Copy constructor and assignment operator to make the class copyable
    OpChainAnalysis(const OpChainAnalysis& other);
    OpChainAnalysis& operator=(const OpChainAnalysis& other);

    // Move constructor and assignment operator
    OpChainAnalysis(OpChainAnalysis&& other) noexcept = default;
    OpChainAnalysis& operator=(OpChainAnalysis&& other) noexcept = default;

    // Destructor
    ~OpChainAnalysis();

    llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE> collectParentOpsChain(mlir::Value val);

    enum class MODE { MAX_VALUE, ALL_VALUES };

    /**
     * @brief Get the value from an OpFoldResult
     * @param val The OpFoldResult to process
     * @param valueMap Map of values to their possible ranges
     * @return The computed value, or nullopt if processing failed
     */
    std::optional<SmallVector<int64_t>> getOpFoldResultValue(mlir::OpFoldResult val, ValueRangeMap& valueMap,
                                                             MODE mode = MODE::MAX_VALUE);

    std::optional<int64_t> getIntegerFromValue(mlir::Value value, bool processOpChain = false);

    std::tuple<int64_t, int64_t, int64_t> getLoopBoundsAndStep(mlir::scf::ForOp forOp,
                                                               ValueRangeMap* valueHints = nullptr);

    SmallVector<int64_t> getForallInductionDimRange(mlir::scf::ForallOp forallOp, mlir::BlockArgument& dimInductionArg,
                                                    ValueRangeMap& valueMap);

    void traverseAndGetBlockArgs(mlir::Value val, llvm::SmallSetVector<mlir::Value, DEFAULT_ARG_SET_SIZE>& blockArgs);

private:
    void updateChainCache(mlir::Value val, const llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>& chain);

    /**
     * @brief Unified method to process a sequence of operations
     * This method handles both block operations and operation chains uniformly.
     * It creates a self-referential block processor that dialect processors can use
     * to recursively process nested blocks with operations from different dialects.
     */
    bool processOperations(llvm::ArrayRef<mlir::Operation*> operations, llvm::DenseMap<mlir::Value, int64_t>& valueMap);

    /**
     * @brief Evaluate operation chain and populate value map
     */
    bool evaluateOpChain(llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>& opChain,
                         llvm::DenseMap<mlir::Value, int64_t>& localOperandMap);

    std::optional<SmallVector<int64_t>> processCallChain(mlir::Value val, ValueRangeMap& valueMap,
                                                         MODE mode = MODE::MAX_VALUE);

    bool generateValueMap(llvm::ArrayRef<mlir::Value> blockOperands, ValueRangeMap& valueMap);
    std::unique_ptr<DialectProcessorRegistry> _registry;
    llvm::DenseMap<mlir::Value, llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>> _chainCache;
    Logger _log;
};

SmallVector<mlir::Operation*> collectOpsInTopologicalOrder(
        llvm::ArrayRef<mlir::Operation*> startNodes,
        llvm::function_ref<llvm::SmallSetVector<mlir::Operation*, DEFAULT_ARG_SET_SIZE>(mlir::Operation*)> getNeighbors,
        llvm::function_ref<bool(mlir::Operation*)> stopCheckFn);

class AnalysisContext {
public:
    AnalysisContext(llvm::ArrayRef<InputRange> inputRanges,
                    const vpux::Logger& log = vpux::Logger::global().nest("scf-analysis-context"))
            : _inputRanges(inputRanges.begin(), inputRanges.end()), _log(log) {
        _strides.reserve(_inputRanges.size());
        llvm::transform(_inputRanges, std::back_inserter(_strides), [](const InputRange& range) {
            return range.stride;
        });
    }

    ArrayRef<SmallVector<int64_t>> getIterationPoints() {
        _log.trace("Generating iteration points from input ranges. Total input ranges: {0}", _inputRanges.size());
        if (_inputRanges.empty()) {
            return {};
        }

        if (_points.empty()) {
            generateIterationPoints();
            return _points;
        }

        return _points;
    }

    ValueRangeMap getValueMapAtIndex(size_t idx) const {
        VPUX_THROW_WHEN(idx >= _points.size(),
                        "Invalid index {0} for iteration points (size: {1}). Index out of bounds.", idx,
                        _points.size());

        ValueRangeMap valueMap;
        for (size_t dimIdx = 0; dimIdx < _inputRanges.size(); ++dimIdx) {
            valueMap[_inputRanges[dimIdx].arg] = {_points[idx][dimIdx]};
        }
        return valueMap;
    }

    const SmallVector<InputRange>& getInputRanges() const {
        return _inputRanges;
    }

    SmallVector<int64_t> getStrides() const {
        return _strides;
    }

    bool arePointsContiguous(ArrayRef<int64_t> point1, ArrayRef<int64_t> point2);

private:
    void generateIterationPoints();

    // Iteration space interval for n-dimensional loops
    SmallVector<SmallVector<int64_t>> _points;

    SmallVector<int64_t> _strides;

    // Expected the input ranges are arranged slowest to fastest changing dimension
    SmallVector<InputRange> _inputRanges;
    vpux::Logger _log;
};

}  // namespace vpux::VPU
