//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace vpux::VPU {

using ScalarValueMap = llvm::DenseMap<mlir::Value, int64_t>;

// Forward declaration for block processor callback
using BlockProcessor = std::function<bool(mlir::Block*, ScalarValueMap&)>;

/**
 * @brief Recovers a bound for a tensor that carries none of its own, as a double bit pattern.
 *
 * Used by TensorDialectProcessor to resolve a tensor.extract whose source is not a constant.
 */
using TensorBoundResolver = std::function<std::optional<int64_t>(mlir::Value)>;

/**
 * @brief Get integer value from a tensor::DimOp using bounds attribute
 * @param dimOp The tensor::DimOp to process
 * @param log Optional logger for diagnostic messages
 * @return Optional integer value extracted from bounds or static shape
 */
std::optional<int64_t> getIntValueFromDimOp(mlir::tensor::DimOp dimOp, const Logger& log = Logger::global());

/**
 * @brief Get the yield operation from a given block
 * @param block The block to search for yield operation
 * @return Pointer to yield operation if found, nullptr otherwise
 */
mlir::scf::YieldOp getYieldOperation(mlir::Block* block);

/**
 * @brief Get the terminator operation from a given block
 * @param block The block to get terminator from
 * @return Pointer to terminator operation, nullptr if block is null or has no terminator
 */
mlir::Operation* getBlockTerminator(mlir::Block* block);

class IDialectProcessor;  // Forward declaration

/**
 * @brief Registry for managing dialect processors
 */
class DialectProcessorRegistry {
public:
    /**
     * @brief Register a dialect processor
     */
    void registerProcessor(std::unique_ptr<IDialectProcessor> processor);

    /**
     * @brief Get a processor for the given operation
     */
    IDialectProcessor* getProcessor(mlir::Operation* op) const;

    /**
     * @brief Check if a processor exists for the given operation
     */
    bool hasProcessor(mlir::Operation* op) const;

    /**
     * @brief Create a registry with default processors
     * @param tensorBoundResolver Optional fallback used by the tensor processor for non-constant sources
     */
    static std::unique_ptr<DialectProcessorRegistry> createDefault(TensorBoundResolver tensorBoundResolver = nullptr);

private:
    std::vector<std::unique_ptr<IDialectProcessor>> _processors;
    mutable llvm::DenseMap<mlir::Dialect*, IDialectProcessor*> _dialectCache;
};

/**
 * @brief Abstract interface for processing operations from specific MLIR dialects
 */
class IDialectProcessor {
public:
    virtual ~IDialectProcessor() = default;

    /**
     * @brief Check if this processor can handle the given operation
     */
    virtual bool canProcess(mlir::Operation* op) const = 0;

    /**
     * @brief Process an operation and update the value map
     * @param op The operation to process
     * @param valueMap Map of values to their computed integer values
     * @param blockProcessor Optional callback for processing nested blocks (used by SCF)
     * @return true if processing succeeded, false otherwise
     */
    virtual bool processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                                  BlockProcessor blockProcessor = nullptr) const = 0;

    /**
     * @brief Get the name of the dialect this processor handles
     */
    virtual llvm::StringRef getDialectName() const = 0;

protected:
    // Protected constructor to prevent direct instantiation
    IDialectProcessor() = default;
};

/**
 * @brief Affine dialect processor
 */
class AffineDialectProcessor : public IDialectProcessor {
public:
    explicit AffineDialectProcessor(Logger log): _log(log) {
    }
    bool canProcess(mlir::Operation* op) const override;
    bool processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                          BlockProcessor blockProcessor = nullptr) const override;
    llvm::StringRef getDialectName() const override {
        return "affine";
    }

private:
    std::pair<mlir::AffineMap, mlir::ValueRange> getAffineMapAndOperands(mlir::Operation* op) const;
    int64_t getAffineResult(mlir::Operation* op, llvm::ArrayRef<int64_t> results) const;
    Logger _log;
};

/**
 * @brief Arithmetic dialect processor
 */
class ArithmeticDialectProcessor : public IDialectProcessor {
public:
    explicit ArithmeticDialectProcessor(Logger log): _log(log) {
    }
    bool canProcess(mlir::Operation* op) const override;
    bool processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                          BlockProcessor blockProcessor = nullptr) const override;
    llvm::StringRef getDialectName() const override {
        return "arith";
    }

private:
    Logger _log;
};

/**
 * @brief SCF (Structured Control Flow) dialect processor
 */
class SCFDialectProcessor : public IDialectProcessor {
public:
    explicit SCFDialectProcessor(Logger log): _log(log) {
    }
    bool canProcess(mlir::Operation* op) const override;
    bool processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                          BlockProcessor blockProcessor = nullptr) const override;
    llvm::StringRef getDialectName() const override {
        return "scf";
    }

private:
    bool processIfOp(mlir::scf::IfOp ifOp, ScalarValueMap& valueMap, const BlockProcessor& blockProcessor) const;

    Logger _log;
};

/**
 * @brief Tensor dialect processor
 */
class TensorDialectProcessor : public IDialectProcessor {
public:
    explicit TensorDialectProcessor(Logger log, TensorBoundResolver boundResolver = nullptr)
            : _log(log), _boundResolver(std::move(boundResolver)) {
    }
    bool canProcess(mlir::Operation* op) const override;
    bool processOperation(mlir::Operation* op, ScalarValueMap& valueMap,
                          BlockProcessor blockProcessor = nullptr) const override;
    llvm::StringRef getDialectName() const override {
        return "tensor";
    }

private:
    bool processExtractOp(mlir::tensor::ExtractOp extractOp, ScalarValueMap& valueMap) const;

    Logger _log;
    TensorBoundResolver _boundResolver;
};

}  // namespace vpux::VPU
