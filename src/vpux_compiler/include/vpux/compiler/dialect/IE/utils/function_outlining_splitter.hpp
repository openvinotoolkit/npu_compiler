//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/function_outlining_splitter.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <variant>

namespace vpux {
namespace IE {

//
// FunctionOutlinerNaive
//

class FunctionOutlinerNaive final : public IFunctionOutliner {
public:
    FunctionOutlinerNaive(size_t numSplits, Logger log);

    // Returns a list of targets for function outlining
    // In case the intention is to split the IR into separate individual functions, each OutliningInstance will have one
    // element
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    size_t _numSplits;
    Logger _log;
};

//
// FunctionOutlinerExhaustive
//

class FunctionOutlinerExhaustive final : public IFunctionOutliner {
public:
    FunctionOutlinerExhaustive(Logger log);
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    Logger _log;
};

//
// FunctionOutlinerRepeatingBlocks
//

class FunctionOutlinerRepeatingBlocks final : public IFunctionOutliner {
public:
    FunctionOutlinerRepeatingBlocks(size_t minOpsInBlock, size_t maxNumIterations, bool weightsAsInputs, Logger log);

    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    size_t _minOpsInBlock;
    size_t _maxNumIterations;
    bool _weightsAsInputs;
    Logger _log;
};

//
// FunctionOutlinerBatching
//

class FunctionOutlinerBatching final : public IFunctionOutliner {
public:
    FunctionOutlinerBatching(Logger log);
    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp mainFunction) override;

private:
    Logger _log;
};

//
// Option parser
//

struct NaiveOptions {
    static constexpr size_t NUM_PARTS_DEFAULT = 2;

    size_t numParts;
};

struct RepeatingBlocksOptions {
    static constexpr size_t MIN_OPS_IN_BLOCK_DEFAULT = 30;
    static constexpr size_t MAX_NUM_ITERATIONS_DEFAULT = 100;
    static constexpr bool WEIGHTS_AS_INPUTS_DEFAULT = true;

    size_t minOpsInBlock;
    size_t maxNumIterations;
    bool weightsAsInputs;
};

struct BatchingOptions {};

//
// OutlinerPassOptions
//

class OutlinerPassOptions {
    std::vector<std::variant<NaiveOptions, RepeatingBlocksOptions, BatchingOptions>> _options;

public:
    template <class T>
    const T* getIf(size_t i) const {
        return std::get_if<T>(&_options[i]);
    }

    size_t count() const {
        return _options.size();
    }

    static OutlinerPassOptions createFromString(StringRef param);
};

}  // namespace IE
}  // namespace vpux
