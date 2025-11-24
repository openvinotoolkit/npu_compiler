//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/net/network_info_utils.hpp"
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

struct DerivedWeightsSeparationInfo {
    std::vector<VPU::ConstArg> topLevelMainArgs;
    std::vector<std::vector<VPU::ConstArg>> slicedSplits;
};

// Returns a unique name for concatenated init results (and, consequently, for
// main inputs).
std::string getUniqueConcatenatedNameOfInitResults(ArrayRef<VPU::ConstArg> args, int64_t initPart) {
    if (args.size() == 1) {
        // Note: preserve the original name when the argument is unchanged
        return args[0].getUniqueName();
    }

    llvm::hash_code hashCode{0};
    for (const auto& arg : args) {
        const size_t hash = Const::ContentAttr::getTransformationHash(arg.transformations);
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
    enum class Mode { Unspecified, GenerateMain, GenerateInit };

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
                                  const DerivedWeightsSeparationInfo& info);

    size_t updateTopLevelMain(mlir::func::FuncOp mainFunc, const DerivedWeightsSeparationInfo& info);
    void updateNetworkInfoForMain(net::NetworkInfoOp netInfo, mlir::func::FuncOp mainFunc, size_t newInputsOffset,
                                  const DerivedWeightsSeparationInfo& info);

    const char* stringifyEnum(Mode mode) {
        switch (mode) {
        case Mode::GenerateMain:
            return "gen-main";
        case Mode::GenerateInit:
            return "gen-init";
        default:
            return "UNKNOWN";
        }
    }

    static constexpr int64_t DEFAULT_INIT_PART = -1;
    static constexpr vpux::Byte DEFAULT_MEMORY_LIMIT = vpux::Byte(std::numeric_limits<int64_t>::max());

    Mode _mode = Mode::Unspecified;
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
                                                 const DerivedWeightsSeparationInfo& info) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&getContext(), &builderLog);

    // update output types
    auto& outputsRegion = netInfo.getOutputsInfo();
    net::eraseSectionEntries(outputsRegion);
    builder.setInsertionPointToStart(&outputsRegion.front());

    const auto& thisInitResults = info.slicedSplits[_initPart];

    const auto outputName = getUniqueConcatenatedNameOfInitResults(thisInitResults, _initPart);
    // Note: guaranteed single result by definition of this pass
    const auto outputType = stripEncoding(mlir::cast<mlir::RankedTensorType>(initFunc.getFunctionType().getResult(0)));
    builder.create<net::DataInfoOp>(appendLoc(netInfo.getLoc(), "concat_out"), outputName, outputType);

    _log.debug("Updating network info for init:");
    _log.nest().debug("Added \"DataInfo\" {0} : {1}", outputName, outputType);
}

std::vector<size_t> matchInitPartOutputsToMainInputs(const std::vector<VPU::ConstArg>& allNewMainInputs,
                                                     const std::vector<VPU::ConstArg>& args, size_t offsetToNewInputs) {
    std::vector<size_t> blockArgs;
    blockArgs.reserve(args.size());

    for (const auto& arg : args) {
        auto it = llvm::find(allNewMainInputs, arg);
        VPUX_THROW_WHEN(it == allNewMainInputs.end(), "Init result not found in main inputs");
        const auto argIndex = std::distance(allNewMainInputs.begin(), it);
        blockArgs.push_back(offsetToNewInputs + argIndex);
    }

    return blockArgs;
}

size_t ConcatInitResults::updateTopLevelMain(mlir::func::FuncOp mainFunc, const DerivedWeightsSeparationInfo& info) {
    auto allNewMainInputs = info.topLevelMainArgs;
    const size_t oldBlockArgsBegin = static_cast<size_t>(mainFunc.getNumArguments()) - allNewMainInputs.size();

    const auto& slicedSplits = info.slicedSplits;
    for (size_t i = 0; i < slicedSplits.size(); ++i) {
        auto indices = matchInitPartOutputsToMainInputs(allNewMainInputs, slicedSplits[i], oldBlockArgsBegin);
        _log.debug("Running obfuscateInputs():");
        VPU::obfuscateInputs(_log.nest(), appendLoc(mainFunc.getLoc(), "obfuscated_inputs{0}", i), mainFunc, indices,
                             [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
                                 return builder.create<VPU::SliceOp>(loc, input, offsets, sizes);
                             });

        // during input obfuscation, old arguments are deleted. this means that
        // main inputs have changed and the next iteration has to be adjusted.
        llvm::sort(indices, std::not_fn(std::less<size_t>{}));
        for (size_t index : indices) {
            const auto i = index - oldBlockArgsBegin;
            allNewMainInputs.erase(allNewMainInputs.begin() + i);
        }
    }

    return oldBlockArgsBegin;
}

void ConcatInitResults::updateNetworkInfoForMain(net::NetworkInfoOp netInfo, mlir::func::FuncOp mainFunc,
                                                 size_t newInputsOffset, const DerivedWeightsSeparationInfo& info) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder builder(&getContext(), &builderLog);

    // update input types
    auto& inputsRegion = netInfo.getInputsInfo();
    // Note: preserve original, non-constant inputs information
    net::eraseSectionEntries(inputsRegion, newInputsOffset);
    builder.setInsertionPointToEnd(&inputsRegion.front());

    _log.debug("Updating network info for main:");
    // take naming convention from init results - the order must match by
    // definition of the main update procedure
    for (size_t i = 0; i < info.slicedSplits.size(); ++i) {
        const auto initPart = static_cast<int64_t>(i);
        const auto& initPartResults = info.slicedSplits[i];

        const auto inputName = getUniqueConcatenatedNameOfInitResults(initPartResults, initPart);
        const auto inputType = stripEncoding(
                mlir::cast<mlir::RankedTensorType>(mainFunc.getFunctionType().getInput(newInputsOffset + i)));
        builder.create<net::DataInfoOp>(appendLoc(netInfo.getLoc(), "concat_in{0}", i), inputName, inputType);

        _log.nest().debug("Added \"DataInfo\" {0} : {1}", inputName, inputType);
    }
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

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp entryPointFunc;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, entryPointFunc);

    DerivedWeightsSeparationInfo info = [&]() {
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

        DerivedWeightsSeparationInfo data;

        // Note: default-sorted splits collected through the tree is *exactly* the
        // splits that are going into the main function (order-wise).
        data.topLevelMainArgs = convertSplitsToMainConstArgs(splits);

        llvm::sort(splits);
        const auto slicedSplits = VPU::sliceAccordingToMemoryLimit(_log, splits, _memoryLimit);
        llvm::transform(slicedSplits, std::back_inserter(data.slicedSplits), convertSplitsToMainConstArgs);

        return data;
    }();

    if (_log.isActive(LogLevel::Debug)) {
        _log.debug("Top-level main arguments in '{0}':", stringifyEnum(_mode));
        for (const auto& [index, arg] : info.topLevelMainArgs | indexed) {
            _log.nest().debug("Arg #{0}: {1}", index, arg);
        }

        _log.debug("The amount of inits in '{0}' is {1}", stringifyEnum(_mode), info.slicedSplits.size());
        _log.debug("Transformation splits for every init in '{0}':", stringifyEnum(_mode));
        for (const auto& [i, splits] : info.slicedSplits | indexed) {
            for (const auto& [j, split] : splits | indexed) {
                _log.nest().debug("Init part #{0}, arg #{1}: {2}", i, j, split);
            }
        }
    }

    switch (_mode) {
    case Mode::GenerateInit: {
        VPUX_THROW_UNLESS(entryPointFunc.getSymName().starts_with("init"), "Expected init function, got {0}",
                          entryPointFunc.getSymName());
        updateInit(entryPointFunc);
        updateNetworkInfoForInit(netInfo, entryPointFunc, info);
        break;
    }
    case Mode::GenerateMain: {
        const auto offset = updateTopLevelMain(entryPointFunc, info);
        updateNetworkInfoForMain(netInfo, entryPointFunc, offset, info);
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
            _mode = Mode::GenerateMain;
        } else if (modeString == "gen-init") {
            _mode = Mode::GenerateInit;
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
    case Mode::GenerateInit: {
        const bool validGenerateInit = (limitSpecified == initPartSpecified);
        if (!validGenerateInit) {
            moduleOp->emitError(
                    formatv("Both {0} and {1} should be either present or unspecified. {0} is: {2} and {1} is: {3}",
                            memoryLimit.getArgStr(), initPart.getArgStr(), limit, initIndex));
            return mlir::failure();
        }
        break;
    }
    case Mode::GenerateMain: {
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
