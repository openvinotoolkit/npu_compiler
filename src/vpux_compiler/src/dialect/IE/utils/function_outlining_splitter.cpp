//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/utils/options.hpp"

namespace {

struct NaiveOptions : mlir::PassPipelineOptions<NaiveOptions> {
    vpux::IntOption numParts{*this, "num-parts", llvm::cl::desc("Number of functions to split the IR into"),
                             ::llvm::cl::init(vpux::IE::NaiveOptions::NUM_PARTS_DEFAULT)};
};

struct RepeatingBlocksOptions : mlir::PassPipelineOptions<RepeatingBlocksOptions> {
    vpux::IntOption minOpsInBlock{*this, "min-ops-in-block",
                                  llvm::cl::desc("Minimum number of operations allowed per block"),
                                  ::llvm::cl::init(vpux::IE::RepeatingBlocksOptions::MIN_OPS_IN_BLOCK_DEFAULT)};
    vpux::IntOption maxNumIterations{*this, "max-num-iterations",
                                     llvm::cl::desc("Maximum number of iterations to find a solution"),
                                     llvm::cl::init(vpux::IE::RepeatingBlocksOptions::MAX_NUM_ITERATIONS_DEFAULT)};
    vpux::BoolOption weightsAsInputs{
            *this, "weights-as-inputs",
            llvm::cl::desc("Add const.DeclareOp's to the function argument list of each block"),
            llvm::cl::init(vpux::IE::RepeatingBlocksOptions::WEIGHTS_AS_INPUTS_DEFAULT)};
};

struct RepeatingBlocksSeparateFunctionsOptions : mlir::PassPipelineOptions<RepeatingBlocksSeparateFunctionsOptions> {
    vpux::IntOption minOpsInBlock{
            *this, "min-ops-in-block", llvm::cl::desc("Minimum number of operations allowed per block"),
            ::llvm::cl::init(vpux::IE::RepeatingBlocksSeparateFunctionsOptions::MIN_OPS_IN_BLOCK_DEFAULT)};
    vpux::IntOption maxNumIterations{
            *this, "max-num-iterations", llvm::cl::desc("Maximum number of iterations to find a solution"),
            llvm::cl::init(vpux::IE::RepeatingBlocksSeparateFunctionsOptions::MAX_NUM_ITERATIONS_DEFAULT)};
};

struct BatchingOptions : mlir::PassPipelineOptions<BatchingOptions> {};

}  // namespace

namespace vpux {
namespace IE {

OutlinerPassOptions OutlinerPassOptions::createFromString(StringRef param) {
    // An example value for param can be:
    // "repeating-blocks='max-num-iterations=30 min-ops-in-block=16', naive='num-parts=2'"

    // Split a string into multiple parts along a delimiter as known from other programming
    // languages.
    auto split = [](StringRef str, char delim) -> SmallVector<StringRef, 8> {
        SmallVector<StringRef, 8> result;
        str.split(result, delim, -1, false);
        return result;
    };

    // trim all characters from left and right that are in trimSet
    auto trim = [](StringRef str, const llvm::DenseSet<char>& trimSet) {
        auto pred = [&trimSet](char c) {
            return trimSet.contains(c);
        };

        auto itBegin = llvm::find_if_not(str, pred);
        auto itEnd = llvm::find_if_not(llvm::reverse(str), pred);

        size_t beginIndex = std::distance(str.begin(), itBegin);
        size_t endIndex = std::distance(str.begin(), itEnd.base());

        return str.substr(beginIndex, endIndex - beginIndex);
    };

    auto paramTrimmed = trim(param, {' ', '"', '\''});
    auto optionBlocks = split(paramTrimmed, ',');

    OutlinerPassOptions options;

    for (auto optionBlock : optionBlocks) {
        auto [modeString, argumentsString] = optionBlock.split('=');
        auto modeStringTrimmed = trim(modeString, {' '});
        auto argumentStringTrimmed = trim(argumentsString, {' ', '\''});

        if (modeStringTrimmed == "naive") {
            auto opt = ::NaiveOptions::createFromString(argumentStringTrimmed);
            VPUX_THROW_WHEN(opt.get() == nullptr, "Cannot create naive options from string: {0}",
                            argumentStringTrimmed);
            VPUX_THROW_WHEN(opt->numParts < 0, "num-parts must be non-negative");

            size_t numParts = opt->numParts;

            options._options.emplace_back(NaiveOptions{numParts});
        } else if (modeStringTrimmed == "repeating-blocks") {
            auto opt = ::RepeatingBlocksOptions::createFromString(argumentStringTrimmed);
            VPUX_THROW_WHEN(opt.get() == nullptr, "Cannot create repeating-blocks options from string: {0}",
                            argumentStringTrimmed);
            VPUX_THROW_WHEN(opt->minOpsInBlock < 0, "min-ops-in-block must be non-negative");
            VPUX_THROW_WHEN(opt->maxNumIterations < 0, "max-num-iterations must be non-negative");

            size_t minOpsInBlock = opt->minOpsInBlock;
            size_t maxNumIterations = opt->maxNumIterations;
            bool weightsAsInputs = opt->weightsAsInputs;

            options._options.emplace_back(RepeatingBlocksOptions{minOpsInBlock, maxNumIterations, weightsAsInputs});
        } else if (modeStringTrimmed == "repeating-blocks-separate-functions") {
            auto opt = ::RepeatingBlocksSeparateFunctionsOptions::createFromString(argumentStringTrimmed);
            VPUX_THROW_WHEN(opt.get() == nullptr,
                            "Cannot create repeating-blocks-separate-functions options from string: {0}",
                            argumentStringTrimmed);
            VPUX_THROW_WHEN(opt->minOpsInBlock < 0, "min-ops-in-block must be non-negative");
            VPUX_THROW_WHEN(opt->maxNumIterations < 0, "max-num-iterations must be non-negative");

            size_t minOpsInBlock = opt->minOpsInBlock;
            size_t maxNumIterations = opt->maxNumIterations;

            options._options.emplace_back(RepeatingBlocksSeparateFunctionsOptions{minOpsInBlock, maxNumIterations});
        } else if (modeStringTrimmed == "batching") {
            auto opt = ::BatchingOptions::createFromString(argumentStringTrimmed);
            VPUX_THROW_WHEN(opt.get() == nullptr, "Cannot create batching options from string: {0}",
                            argumentStringTrimmed);
            options._options.emplace_back(BatchingOptions{});
        } else {
            VPUX_THROW("Unknown outlining mode '{0}'", modeStringTrimmed);
        }
    }

    return options;
}

}  // namespace IE
}  // namespace vpux
