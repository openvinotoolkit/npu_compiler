//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include <cstdint>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONCATINITRESULTS
#define GEN_PASS_DEF_CONCATINITRESULTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// TODO: this is an ad-hoc re-implementation of the -introduce-init-pass logic
// that works on top of vpux::utils::ArgumentCache<vpux::VPU::ConstArg>.
std::vector<VPU::ConstArg> convertSplitsToMainConstArgs(ArrayRef<VPU::TransformationsSplit> splits) {
    // Note: deduplication is automatic via hashing
    mlir::SetVector<VPU::ConstArg, std::vector<VPU::ConstArg>> args;
    for (const auto& split : splits) {
        // Note: main inputs are init outputs
        args.insert(VPU::ConstArg(split.take(VPU::WeightsSeparationSchedule::Main)));
    }
    return args.takeVector();
}

struct ObfuscationInfo {
    std::vector<std::vector<size_t>> inputGroupsToObfuscate;
    std::vector<size_t> inputsToPreserve;
};

ObfuscationInfo matchInitResultsToMainInputs(const Logger& log, net::NetworkInfoOp netInfo,
                                             const std::vector<std::vector<VPU::ConstArg>>& resultsOfAllInits) {
    ObfuscationInfo info;

    log.debug("Analyzing NetworkInfo to match init results to indices");
    DenseMap<mlir::StringRef, size_t> argToIndex;
    const auto dataInfoOps = to_small_vector(netInfo.getInputsInfo().getOps<net::DataInfoOp>());
    for (size_t i = 0; i < dataInfoOps.size(); ++i) {
        net::DataInfoOp dataInfo = dataInfoOps[i];
        const auto name = dataInfo.getName();
        if (bool toBeObfuscated = name.starts_with(VPU::INIT_OUTPUT_PREFIX); toBeObfuscated) {
            log.nest().debug("{0} data info entry (index {1}) is an init result", dataInfo, i);
            assert(!argToIndex.contains(name) && "Every entry can appear exactly once");
            argToIndex[name] = i;
        } else {
            log.nest().debug("{0} data info entry (index {1}) is NOT an init result", dataInfo, i);
            info.inputsToPreserve.push_back(i);
        }
    }

    for (const auto& singleInitResults : resultsOfAllInits) {
        auto& currentGroup = info.inputGroupsToObfuscate.emplace_back();
        currentGroup.reserve(singleInitResults.size());

        for (const auto& arg : singleInitResults) {
            const auto uniqueName = arg.getUniqueName();
            const auto it = argToIndex.find(uniqueName);
            VPUX_THROW_WHEN(it == argToIndex.end(), "Matching init result to main input failed for \"{0}\"",
                            uniqueName);

            log.nest().debug("Init result \"{0}\" has index {1}", uniqueName, it->second);
            currentGroup.push_back(it->second);

            argToIndex.erase(it);
        }
    }

    // sanity check
    VPUX_THROW_UNLESS(argToIndex.empty(),
                      "Matching init results to main inputs failed - there are {0} unmatched entries",
                      argToIndex.size());

    return info;
}

// Returns a unique name for concatenated init results (and, consequently, for
// main inputs).
std::string getUniqueConcatenatedNameOfInitResults(ArrayRef<VPU::ConstArg> args, int64_t initPart) {
    if (args.size() == 1) {
        // Note: preserve the original name when the argument is unchanged
        return args[0].getUniqueName();
    }

    llvm::hash_code hashCode{0};
    for (const auto& arg : args) {
        const size_t hash = Const::ContentAttr::getTransformationHash(arg.content, arg.transformations);
        hashCode = llvm::hash_combine(hashCode, hash);
    }
    return formatv("{0}{1}_hash_{2}_concat", vpux::VPU::INIT_OUTPUT_PREFIX, initPart, hashCode);
}

// Returns a new ranked tensor without the tensor encoding.
mlir::RankedTensorType stripEncoding(mlir::RankedTensorType origin) {
    return mlir::RankedTensorType::get(origin.getShape(), origin.getElementType());
}

class ConcatInitResults final : public VPU::impl::ConcatInitResultsBase<ConcatInitResults> {
public:
    explicit ConcatInitResults(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit ConcatInitResults(StringRef wsExtractionModeString, std::optional<int64_t> initPart,
                               std::optional<Byte> limit, const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
        this->wsExtractionMode = wsExtractionModeString.str();
        if (initPart.has_value()) {
            this->initPart = initPart.value();
        }
        if (limit.has_value()) {
            this->memoryLimit = limit.value().count();
        }
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext*) final;
    mlir::LogicalResult deferredInitialize(mlir::ModuleOp moduleOp);
    void safeRunOnModule() final;

    void updateInit(mlir::func::FuncOp initFunc);
    void updateNetworkInfoForInit(net::NetworkInfoOp netInfo, mlir::func::FuncOp initFunc,
                                  const std::vector<std::vector<VPU::ConstArg>>& resultsOfAllInits);

    std::vector<size_t> updateTopLevelMain(net::NetworkInfoOp netInfo, mlir::func::FuncOp mainFunc,
                                           const std::vector<std::vector<VPU::ConstArg>>& resultsOfAllInits);
    void updateNetworkInfoForMain(net::NetworkInfoOp netInfo, mlir::func::FuncOp mainFunc,
                                  ArrayRef<size_t> preservedArgIndices,
                                  const std::vector<std::vector<VPU::ConstArg>>& resultsOfAllInits);

    const char* stringifyEnum(VPU::WeightsSeparationMode mode) {
        switch (mode) {
        case VPU::WeightsSeparationMode::GenerateMain:
            return "gen-main";
        case VPU::WeightsSeparationMode::GenerateInit:
            return "gen-init";
        default:
            return "UNKNOWN";
        }
    }

    static constexpr int64_t DEFAULT_INIT_PART = -1;
    static constexpr vpux::Byte DEFAULT_MEMORY_LIMIT = vpux::Byte(std::numeric_limits<int64_t>::max());

    VPU::WeightsSeparationMode _mode = VPU::WeightsSeparationMode::Unspecified;
    int64_t _initPart = DEFAULT_INIT_PART;
    vpux::Byte _memoryLimit = DEFAULT_MEMORY_LIMIT;
};

void ConcatInitResults::updateInit(mlir::func::FuncOp initFunc) {
    std::vector<size_t> outputIndices(initFunc.getNumResults(), 0);
    std::iota(outputIndices.begin(), outputIndices.end(), 0);
    // all outputs become single blob in gen-init
    _log.debug("Running obfuscateOutputs():");
    VPU::obfuscateOutputs(_log.nest(), appendLoc(initFunc.getLoc(), "obfuscated_outputs"), initFunc, outputIndices,
                          [](mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<mlir::Value> inputs, int64_t axis) {
                              return builder.create<IE::ConcatOp>(loc, inputs, axis);
                          });
}

void ConcatInitResults::updateNetworkInfoForInit(net::NetworkInfoOp netInfo, mlir::func::FuncOp initFunc,
                                                 const std::vector<std::vector<VPU::ConstArg>>& resultsOfAllInits) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&getContext(), &builderLog);

    // update output types
    auto& outputsRegion = netInfo.getOutputsInfo();
    net::eraseSectionEntries(outputsRegion);
    builder.setInsertionPointToStart(&outputsRegion.front());

    const auto& thisInitResults = resultsOfAllInits[_initPart];

    const auto outputName = getUniqueConcatenatedNameOfInitResults(thisInitResults, _initPart);
    // Note: guaranteed single result by definition of this pass
    const auto outputType = stripEncoding(mlir::cast<mlir::RankedTensorType>(initFunc.getFunctionType().getResult(0)));
    builder.create<net::DataInfoOp>(appendLoc(netInfo.getLoc(), "concat_out"), outputName, outputType);

    _log.debug("Updating network info for init:");
    _log.nest().debug("Added \"DataInfo\" {0} : {1}", outputName, outputType);
}

std::vector<size_t> ConcatInitResults::updateTopLevelMain(
        net::NetworkInfoOp netInfo, mlir::func::FuncOp mainFunc,
        const std::vector<std::vector<VPU::ConstArg>>& resultsOfAllInits) {
    const auto obfuscationInfo = matchInitResultsToMainInputs(_log, netInfo, resultsOfAllInits);
    _log.debug("Running obfuscateInputGroups():");
    VPU::obfuscateInputGroups(_log.nest(), appendLoc(mainFunc.getLoc(), "obfuscated_inputs"), mainFunc,
                              obfuscationInfo.inputGroupsToObfuscate,
                              [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                 ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
                                  return builder.create<VPU::SliceOp>(loc, input, offsets, sizes);
                              });
    return obfuscationInfo.inputsToPreserve;
}

void ConcatInitResults::updateNetworkInfoForMain(net::NetworkInfoOp netInfo, mlir::func::FuncOp mainFunc,
                                                 ArrayRef<size_t> preservedArgIndices,
                                                 const std::vector<std::vector<VPU::ConstArg>>& resultsOfAllInits) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&getContext(), &builderLog);

    auto log = _log;
    log.debug("Updating network info for main");
    log = log.nest();

    // it is generally easier to re-create the whole InputsInfo section from
    // scratch than to update it in-place (removing arbitrary entries)
    auto& inputsRegion = netInfo.getInputsInfo();
    builder.setInsertionPointToStart(&inputsRegion.front());

    log.debug("Copying original data info entries ({0} in total) that must remain:", preservedArgIndices.size());
    const auto originalDataInfos = to_small_vector(inputsRegion.getOps<net::DataInfoOp>());
    for (size_t i : preservedArgIndices) {
        auto originalDataInfo = originalDataInfos[i];
        builder.clone(*originalDataInfo);
        log.nest().debug("Copied data info with name {0}", originalDataInfo.getName());
    }

    // the amount of preserved inputs acts as an offset to find correct "new"
    // argument in the main function
    const auto newInputsOffset = preservedArgIndices.size();

    log.debug("Adding new data info entries ({0} in total) for obfuscated init results:", resultsOfAllInits.size());
    // take naming convention from init results - the order must match by
    // definition of the main update procedure
    for (size_t i = 0; i < resultsOfAllInits.size(); ++i) {
        const auto initPart = static_cast<int64_t>(i);
        const auto& initPartResults = resultsOfAllInits[i];

        const auto inputName = getUniqueConcatenatedNameOfInitResults(initPartResults, initPart);
        const auto inputType = stripEncoding(
                mlir::cast<mlir::RankedTensorType>(mainFunc.getFunctionType().getInput(newInputsOffset + i)));
        builder.create<net::DataInfoOp>(appendLoc(netInfo.getLoc(), "concat_in{0}", i), inputName, inputType);

        log.nest().debug("Added \"DataInfo\" {0} : {1}", inputName, inputType);
    }

    // once refreshed data entries are added, delete original entries
    net::eraseSectionEntries(inputsRegion, newInputsOffset + resultsOfAllInits.size());
}

void ConcatInitResults::safeRunOnModule() {
    auto moduleOp = getOperation();
    if (mlir::failed(deferredInitialize(moduleOp))) {
        signalPassFailure();
        return;
    }

    // Note: as this pass is run multiple times (at least twice: for init and
    // main), this debug line helps to split the logs based on this criterion.
    _log.debug("Running this pass in '{0}' mode, init part = {1} (memory limit = {2:F} KB)", stringifyEnum(_mode),
               _initPart, (static_cast<double>(_memoryLimit.count()) / 1024.));
    _log = _log.nest();
    VPUX_SCOPE_EXIT {
        _log = _log.unnest();
    };

    auto [netInfo, entryPointFunc] = net::getFromModule(moduleOp);

    std::vector<std::vector<VPU::ConstArg>> resultsOfAllInits = [&]() {
        auto infoOpt = getCachedAnalysis<VPU::WeightsSeparationInfo>();
        VPUX_THROW_UNLESS(infoOpt.has_value(), "VPU::WeightsSeparationInfo analysis must be cached");
        const auto& info = infoOpt->get();

        auto splits = info.getCollectedSplits();
        if (_log.isActive(LogLevel::Debug)) {
            _log.debug("The following transformation splits are collected in '{0}':", stringifyEnum(_mode));
            for (const auto& split : splits) {
                _log.nest().debug("{0} at loc '{1}'", split.getContentAttr(), split.getLoc());
            }
        }

        std::vector<std::vector<VPU::ConstArg>> data;

        // TODO: move stable-sort inside WeigbhtsSeparationInfo once we get rid
        // of the topLevelMainArgs
        std::stable_sort(splits.begin(), splits.end());

        const auto slicedSplits = VPU::sliceAccordingToMemoryLimit(_log, splits, _memoryLimit);
        llvm::transform(slicedSplits, std::back_inserter(data), convertSplitsToMainConstArgs);

        return data;
    }();

    if (_log.isActive(LogLevel::Debug)) {
        _log.debug("The amount of inits in '{0}' is {1}", stringifyEnum(_mode), resultsOfAllInits.size());
        _log.debug("Transformation splits for every init in '{0}':", stringifyEnum(_mode));
        for (const auto& [i, splits] : resultsOfAllInits | indexed) {
            for (const auto& [j, split] : splits | indexed) {
                _log.nest().debug("Init part #{0}, arg #{1}: {2}", i, j, split);
            }
        }
    }

    switch (_mode) {
    case VPU::WeightsSeparationMode::GenerateInit: {
        VPUX_THROW_UNLESS(entryPointFunc.getSymName().starts_with("init"), "Expected init function, got {0}",
                          entryPointFunc.getSymName());
        updateInit(entryPointFunc);
        updateNetworkInfoForInit(netInfo, entryPointFunc, resultsOfAllInits);
        break;
    }
    case VPU::WeightsSeparationMode::GenerateMain: {
        const auto preservedArgIndices = updateTopLevelMain(netInfo, entryPointFunc, resultsOfAllInits);
        updateNetworkInfoForMain(netInfo, entryPointFunc, preservedArgIndices, resultsOfAllInits);
        break;
    }
    default:
        VPUX_THROW("Invalid mode encountered");
    }
}

mlir::LogicalResult ConcatInitResults::initialize(mlir::MLIRContext*) {
    if (wsExtractionMode.hasValue()) {
        auto modeString = wsExtractionMode.getValue();

        if (modeString == "gen-main") {
            _mode = VPU::WeightsSeparationMode::GenerateMain;
        } else if (modeString == "gen-init") {
            _mode = VPU::WeightsSeparationMode::GenerateInit;
        } else {
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult ConcatInitResults::deferredInitialize(mlir::ModuleOp moduleOp) {
    const auto limit = memoryLimit.hasValue() ? vpux::Byte(memoryLimit.getValue()) : DEFAULT_MEMORY_LIMIT;
    const int64_t initIndex = initPart.hasValue() ? initPart.getValue() : DEFAULT_INIT_PART;

    const bool limitSpecified = limit != DEFAULT_MEMORY_LIMIT;
    const bool initPartSpecified = initIndex != DEFAULT_INIT_PART;

    // verify correctness
    switch (_mode) {
    case VPU::WeightsSeparationMode::GenerateInit: {
        const bool validGenerateInit = (limitSpecified == initPartSpecified);
        if (!validGenerateInit) {
            moduleOp->emitError(
                    formatv("Both {0} and {1} should be either present or unspecified. {0} is: {2} and {1} is: {3}",
                            memoryLimit.getArgStr(), initPart.getArgStr(), limit, initIndex));
            return mlir::failure();
        }
        break;
    }
    case VPU::WeightsSeparationMode::GenerateMain: {
        if (initPartSpecified) {
            moduleOp->emitError(formatv("{0} is not supported in monolithic mode", initPart.getArgStr()));
            return mlir::failure();
        }
        break;
    }
    default:
        return mlir::failure();
    }

    // Note: use 0 instead of -1 to simplify the logic of picking init part.
    _initPart = initPartSpecified ? initIndex : 0;
    _memoryLimit = limit;
    return mlir::success();
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createConcatInitResultsPass(const Logger& log) {
    return std::make_unique<ConcatInitResults>(log);
}

std::unique_ptr<mlir::Pass> vpux::VPU::createConcatInitResultsPass(StringRef wsExtractionModeString,
                                                                   std::optional<int64_t> initPart,
                                                                   std::optional<Byte> limit, const Logger& log) {
    return std::make_unique<ConcatInitResults>(wsExtractionModeString, initPart, limit, log);
}
