//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_legalization.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation_ir_modification.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/constant_call_stack.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/ir_modification.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/Type.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>

#include <limits>
#include <sstream>

namespace vpux::VPU {
#define GEN_PASS_DECL_INTRODUCEINITFUNCTION
#define GEN_PASS_DEF_INTRODUCEINITFUNCTION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {
void verifyNetworkInfoWithRespectToNewArguments(net::NetworkInfoOp netInfo) {
    std::ignore = netInfo;

    // Note: This is a "debug" check, similar to the verifier, so it is placed
    // under a developer-build guard. This is, so far, also very WS-specific so
    // is not part of net::NetworkInfoOp::verify(). For example, right now
    // compiler allows vpux_ow_<id> entries to appear multiple times (due to
    // duplicate weight buffers).
#ifdef VPUX_DEVELOPER_BUILD
    llvm::DenseSet<mlir::StringRef> dataInfoNames;

    const auto checkDataInfo = [&](net::DataInfoOp dataInfo, StringRef regionName) {
        const auto name = dataInfo.getName();
        const bool initAddedEntry = name.starts_with(VPU::INIT_OUTPUT_PREFIX);
        if (!initAddedEntry) {
            return;
        }

        const bool isUniqueName = dataInfoNames.insert(name).second;
        VPUX_THROW_UNLESS(isUniqueName, "[WS] Found a duplicate net::DataInfo name '{0}' in \"{1}\"", name, regionName);
    };

    for (auto dataInfoOp : netInfo.getInputsInfo().getOps<net::DataInfoOp>()) {
        checkDataInfo(dataInfoOp, "inputsInfo");
    }
    for (auto dataInfoOp : netInfo.getOutputsInfo().getOps<net::DataInfoOp>()) {
        checkDataInfo(dataInfoOp, "outputsInfo");
    }
#endif
}

struct InitSpecificLoggerBase {
    virtual ~InitSpecificLoggerBase() = default;
    virtual void analyzeInitFunction(mlir::func::FuncOp) = 0;
    virtual void print(const Logger&) = 0;
};

struct InitSpecificNullLogger : InitSpecificLoggerBase {
    void analyzeInitFunction(mlir::func::FuncOp) override {
    }
    void print(const Logger&) override {
    }
};

// Simple helper that gives write access to the underlying data if and only if a
// specified key has never been seen before by the helper.
template <typename T, typename Key>
class Uniqued {
    mlir::DenseSet<Key> _uniquenessChecker;
    T _data{};

public:
    // Returns a pointer to a mutable data. Returns a nullptr if the data cannot
    // be accessed.
    T* operator()(const Key& key) {
        const bool firstOccurrence = _uniquenessChecker.insert(key).second;
        if (firstOccurrence) {
            return std::addressof(_data);
        }
        return nullptr;
    }

    // Returns a pointer to an immutable data.
    const T* operator->() const {
        return std::addressof(_data);
    }
};

class InitSpecificMetaInfoLogger : public InitSpecificLoggerBase {
    struct ConstantInfo {
        size_t count{0};
        Byte size{0};
    };
    Uniqued<ConstantInfo, mlir::StringRef> _importedWeights;            // weights from original model
    Uniqued<ConstantInfo, mlir::StringRef> _availableWeights;           // weights still present in pre-init IR
    Uniqued<ConstantInfo, Const::ContentAttr> _ovConstants;             // OV-originated constant ops
    Uniqued<ConstantInfo, Const::ContentAttr> _usedOvConstants;         // supported by weights separation
    Uniqued<ConstantInfo, Const::ContentAttr> _unusedOvConstants;       // supported by weights separation but ignored
    Uniqued<ConstantInfo, Const::ContentAttr> _unsupportedOvConstants;  // unsupported by weights separation

    ConstantInfo _currentInitInputs;
    ConstantInfo _currentInitOutputs;

    static double toKb(vpux::Byte bytes);
    static double percentify(vpux::Byte n, vpux::Byte m);

public:
    InitSpecificMetaInfoLogger(mlir::ModuleOp moduleOp, bool skipViewLikeOnly);
    void analyzeInitFunction(mlir::func::FuncOp) override;
    void print(const Logger&) override;
};

InitSpecificMetaInfoLogger::InitSpecificMetaInfoLogger(mlir::ModuleOp moduleOp, bool skipViewLikeOnly) {
    // Note: to collect *all* OV-imported weights, use blob manager (backing
    // container for dense_resource<>).
    const auto& manager = mlir::DenseResourceElementsHandle::getManagerInterface(moduleOp.getContext());
    manager.getBlobManager().getBlobMap(
            [&](const llvm::StringMap<mlir::DialectResourceBlobManager::BlobEntry>& allBlobs) {
                for (const auto& entry : allBlobs) {
                    const auto& blob = entry.getValue();
                    const auto key = blob.getKey();
                    if (!key.starts_with(Const::IMPORTED_WEIGHT_PREFIX)) {
                        continue;
                    }

                    // Note: blob entries are unique (and so are keys)
                    if (auto* info = _importedWeights(key)) {
                        info->count++;
                        info->size += vpux::Byte(static_cast<int64_t>(blob.getBlob()->getData().size()));
                    }
                }
            });

    moduleOp->walk([&](Const::DeclareOp constOp) {
        if (!Const::isOpenVINOConstant(constOp)) {
            return;
        }

        const auto attr = constOp.getContentAttr();
        if (auto* info = _availableWeights(getResourceName(attr.getBaseContent()))) {
            info->count++;
            info->size += vpux::getExpectedBufferSize(attr.getBaseContent().getType());
        }

        // Note: due to the nature of the IR, duplicate constants are assumed to
        // be fused. However, there's a slight chance that the same constant can
        // be used in two different functions?
        if (auto* info = _ovConstants(attr)) {
            info->count++;
            info->size += vpux::getExpectedBufferSize(constOp.getContentAttr().getType());
        }

        // if suitable, recorded into used constants
        if (VPU::isSuitableForWeightlessCompilation(constOp, skipViewLikeOnly)) {
            if (auto* info = _usedOvConstants(attr)) {
                info->count++;
                info->size += vpux::getExpectedBufferSize(attr.getType());
            }
            return;
        }

        // if not suitable, but trivial, recorded into unused constants
        // otherwise - unsupported
        auto* weightsCategory = VPU::isTrivialForWeightsSeparation(constOp, skipViewLikeOnly)
                                        ? &_unusedOvConstants
                                        : &_unsupportedOvConstants;
        if (auto* info = (*weightsCategory)(attr)) {
            info->count++;
            info->size += vpux::getExpectedBufferSize(constOp.getContentAttr().getType());
        }
    });
}

void InitSpecificMetaInfoLogger::analyzeInitFunction(mlir::func::FuncOp initFunc) {
    const auto calculateSize = [](ArrayRef<mlir::Type> c) {
        return std::accumulate(c.begin(), c.end(), vpux::Byte(0), [](vpux::Byte i, mlir::Type argType) {
            return i + vpux::getExpectedBufferSize(argType);
        });
    };
    _currentInitInputs = {initFunc.getNumArguments(), calculateSize(initFunc.getArgumentTypes())};
    _currentInitOutputs = {initFunc.getNumResults(), calculateSize(initFunc.getResultTypes())};
}

void InitSpecificMetaInfoLogger::print(const Logger& log) {
    log.info("Summary about constants:");
    auto generalStats = log.nest(1);
    generalStats.info("All imported unique weights: {0} ({1:F} KB)", _importedWeights->count,
                      toKb(_importedWeights->size));
    generalStats.info("Available unique weights[1]: {0} ({1:F} KB which is {2:P})", _availableWeights->count,
                      toKb(_availableWeights->size), percentify(_availableWeights->size, _importedWeights->size));
    generalStats.info("Unique weights used by schedule (from available): {0} ({1:F} KB which is {2:P})",
                      _currentInitInputs.count, toKb(_currentInitInputs.size),
                      percentify(_currentInitInputs.size, _availableWeights->size));

    generalStats.info("OV-originated constants[2] in IR: {0} ({1:F} KB)", _ovConstants->count,
                      toKb(_ovConstants->size));
    generalStats.info("Unused constants[3]: {0} ({1:F} KB which is {2:P})", _unusedOvConstants->count,
                      toKb(_unusedOvConstants->size), percentify(_unusedOvConstants->size, _ovConstants->size));
    generalStats.info("Unsupported constants[4]: {0} ({1:F} KB which is {2:P})", _unsupportedOvConstants->count,
                      toKb(_unsupportedOvConstants->size),
                      percentify(_unsupportedOvConstants->size, _ovConstants->size));
    generalStats.info("Size percentage of *used* constants: {0:P}",
                      percentify(_usedOvConstants->size, _ovConstants->size));

    generalStats.info("Generated schedule's total I/O size: {0:F} KB ({1:F} KB inputs + {2:F} KB outputs)",
                      toKb(_currentInitInputs.size + _currentInitOutputs.size), toKb(_currentInitInputs.size),
                      toKb(_currentInitOutputs.size));

    generalStats.info("");  // dummy line
    generalStats.info("[1]: available unique weights - weights that come from original model and are used in the "
                      "compiled schedule (via constant operations)");
    generalStats.info("[2]: OV-originated constants - constant operations that combine OV weights with transformations "
                      "(e.g. subview, reorder)");
    generalStats.nest(1).info("Note: the same unique weight could be used in multiple constants");
    generalStats.info("[3]: unused constants - OV-originated constants that are ignored by weights separation (e.g. "
                      "splats, only trivial transformations)");
    generalStats.info("[4]: unsupported constants - OV-originated constants that have unsupported transformations");
}

double InitSpecificMetaInfoLogger::toKb(vpux::Byte bytes) {
    constexpr auto multiplier = vpux::MemMultiplier<MemType::KB, MemType::Byte>::value;
    return static_cast<double>(bytes.count()) / multiplier;
}

double InitSpecificMetaInfoLogger::percentify(vpux::Byte n, vpux::Byte m) {
    return static_cast<double>(n.count()) / static_cast<double>(m.count());
}

// Casts the resulting value to its storage type counterpart. This is normally
// done in init and thus in IE dialect.
mlir::Value castToStorageType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                              const VPU::IoBoundaryAdapter::TypeInfo& info) {
    if (!info.valid()) {
        return input;
    }
    return builder.create<IE::QuantizeCastOp>(appendLoc(loc, "quant_cast"), input, info.storageType);
}

/// Weights-separation specific argument cache.
using WsArgumentCache = vpux::utils::ArgumentCache<vpux::VPU::ConstArg>;

//
// IntroduceInitFunctionPass
//

class IntroduceInitFunctionPass final : public VPU::impl::IntroduceInitFunctionBase<IntroduceInitFunctionPass> {
public:
    explicit IntroduceInitFunctionPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    IntroduceInitFunctionPass(StringRef wsExtractionModeString, std::optional<int64_t> initPart,
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

    using InitResults = std::vector<std::tuple<VPU::ConstArg, mlir::Value>>;

    std::tuple<VPU::WsArgumentCache, InitResults> buildInitFunctionBody(mlir::func::FuncOp initFuncOp,
                                                                        ArrayRef<VPU::TransformationsSplit> splits);
    std::tuple<mlir::func::FuncOp, VPU::WsArgumentCache, InitResults> buildInitFunction(
            mlir::OpBuilder& moduleBuilder, mlir::Location loc, ArrayRef<VPU::TransformationsSplit> splits,
            StringRef name);
    std::tuple<mlir::func::FuncOp, VPU::WsArgumentCache, InitResults> buildInitFunction(
            mlir::func::FuncOp mainFuncOp, const VPU::WeightsSeparationInfo& wsAnalysis);
    struct SplitSlice {
        std::vector<VPU::TransformationsSplit> splits;
        bool initIsSliced = false;  // specifies whether init schedule was sliced and only a slice is returned.
    };
    SplitSlice collectSplitsAccordingToSettings(const VPU::WeightsSeparationInfo& wsAnalysis);

    VPU::WsArgumentCache updateMainAndOutlinedFunctions(mlir::ModuleOp moduleOp, mlir::func::FuncOp mainFuncOp,
                                                        const VPU::CallChainTree& tree);

    // configures NetworkInfo to assume init-schedule is the entry-point.
    void setNetworkEntryPointToInit(net::NetworkInfoOp netInfo, mlir::func::FuncOp initFuncOp,
                                    const VPU::WsArgumentCache& argCache, const InitResults& initResults);
    // configures NetworkInfo to assume *updated* main-schedule is the
    // entry-point. the behaviour is to be considered equivalent to setting the
    // entry-point to init.
    void setNetworkEntryPointToMain(net::NetworkInfoOp netInfo, const VPU::WsArgumentCache& topLevelMainArgCache);

    mlir::LogicalResult initialize(mlir::MLIRContext* context) final;
    mlir::LogicalResult deferredInitialize(mlir::ModuleOp moduleOp);
    void safeRunOnModule() final;

    static constexpr int64_t DEFAULT_INIT_PART = -1;
    static constexpr vpux::Byte DEFAULT_MEMORY_LIMIT = vpux::Byte(std::numeric_limits<int64_t>::max());

    WeightsSeparationMode _mode = WeightsSeparationMode::Unspecified;
    int64_t _initPart = DEFAULT_INIT_PART;
    vpux::Byte _memoryLimit = DEFAULT_MEMORY_LIMIT;
};

void IntroduceInitFunctionPass::setNetworkEntryPointToInit(net::NetworkInfoOp netInfo, mlir::func::FuncOp initFuncOp,
                                                           const VPU::WsArgumentCache& argCache,
                                                           const InitResults& initResults) {
    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(netInfo, &listener);
    // We replace the old netInfo with a new one where all the sections are cleared. Especially, the profiling sections
    // are not supported for @init().
    // TODO E#176454: Revert to the previous logic.
    const auto entrySymbol = mlir::FlatSymbolRefAttr::get(&getContext(), initFuncOp.getSymName());
    auto newNetInfo = builder.create<net::NetworkInfoOp>(netInfo.getLoc(), entrySymbol, /*withProfiling=*/false);
    net::setupSections(newNetInfo, false);
    netInfo->erase();

    // update input types
    auto& inputsRegion = newNetInfo.getInputsInfo();
    net::eraseSectionEntries(inputsRegion);
    builder.setInsertionPointToStart(&inputsRegion.front());

    for (auto it : argCache.getSortedArgs()) {
        const auto& [entry, blockArg] = *it;
        const auto type = blockArg.getType();
        const auto name = getResourceName(entry.content);
        builder.create<net::DataInfoOp>(appendLoc(newNetInfo.getLoc(), name), name, type);
    }

    // update output types
    auto& outputsRegion = newNetInfo.getOutputsInfo();
    net::eraseSectionEntries(outputsRegion);
    builder.setInsertionPointToStart(&outputsRegion.front());

    for (const auto& [entry, value] : initResults) {
        const auto outputName = entry.getUniqueName();
        builder.create<net::DataInfoOp>(appendLoc(newNetInfo.getLoc(), outputName), outputName, value.getType());
    }

    verifyNetworkInfoWithRespectToNewArguments(newNetInfo);
}

void printConstCallStack(std::vector<vpux::VPU::TransformationsSplit> splits, mlir::MLIRContext* ctx,
                         const Logger& log) {
    if (!log.isActive(LogLevel::Trace) || !isDeveloperBuild()) {
        return;
    }

    auto& csCache = vpux::getCache<vpux::Const::CallStackCache, vpux::Const::ConstDialect>(ctx);
    auto handle = csCache.lockCallStack();
    const auto& callStack = *handle;

    std::stringstream stream;
    if (callStack.empty()) {
        stream << "No constant call stacks were recorded.";
    } else {
        for (const auto& split : splits) {
            stream << "Base content: ";
            auto contentAttr = split.getContentAttr();
            auto baseContent = contentAttr.getBaseContent();
            const auto resourceName = getResourceName(baseContent);
            stream << resourceName.str();

            auto transformations = contentAttr.getTransformations();
            for (auto transformation : transformations) {
                auto trace = csCache.getSpecificCallStack(baseContent, transformation);
                std::string transformationStr;
                llvm::raw_string_ostream os(transformationStr);
                transformation.print(os);
                stream << "\tTransformation: " << os.str() << '\n';
                stream << "\tTrace: " << trace << '\n';
            }
        }
    }

    log.trace("The following call stack of constant operations is found:\n{0}\n", stream.str());
}

std::tuple<mlir::func::FuncOp, VPU::WsArgumentCache, IntroduceInitFunctionPass::InitResults>
IntroduceInitFunctionPass::buildInitFunction(mlir::OpBuilder& moduleBuilder, mlir::Location initLoc,
                                             ArrayRef<VPU::TransformationsSplit> splits, StringRef name) {
    // create empty func @name() : () -> ()
    auto initFuncOp =
            moduleBuilder.create<mlir::func::FuncOp>(initLoc, name, mlir::FunctionType::get(&getContext(), {}, {}));
    auto bodyBlock = initFuncOp.addEntryBlock();

    // construct init body
    auto [argCache, initResults] = buildInitFunctionBody(initFuncOp, splits);

    // set function signature
    const auto resultValueRange =
            to_std_vector(initResults | transformed([](const InitResults::value_type& pair) -> mlir::Value {
                              return std::get<1>(pair);
                          }));
    auto initBuilder = mlir::OpBuilder::atBlockEnd(bodyBlock, moduleBuilder.getListener());
    auto returnOp =
            initBuilder.create<mlir::func::ReturnOp>(appendLoc(initFuncOp.getLoc(), "return"), resultValueRange);

    auto initFuncType =
            mlir::FunctionType::get(&getContext(), bodyBlock->getArgumentTypes(), returnOp.getOperands().getTypes());
    initFuncOp.setFunctionType(initFuncType);

    printConstCallStack(splits, &getContext(), _log);
    if (_log.isActive(LogLevel::Debug)) {
        int64_t inByteCount = 0;
        int64_t outByteCount = 0;

        for (auto type : initFuncType.getInputs()) {
            inByteCount += mlir::cast<NDTypeInterface>(type).getTotalAllocSize().count();
        }

        for (auto type : initFuncType.getResults()) {
            outByteCount += mlir::cast<NDTypeInterface>(type).getTotalAllocSize().count();
        }

        auto statsLogger = _log.nest(1);
        statsLogger.debug("Constructed init part called {0}", name);
        statsLogger.debug("Argument count: {0}", initFuncType.getNumInputs());
        statsLogger.debug("Result count: {0}", initFuncType.getNumResults());
        statsLogger.debug("In-byte count: {0}", inByteCount);
        statsLogger.debug("Out-byte count {0}", outByteCount);
        statsLogger.debug("Signature: {0}", initFuncType);
    }

    return {initFuncOp, argCache, initResults};
}

std::tuple<VPU::WsArgumentCache, IntroduceInitFunctionPass::InitResults>
IntroduceInitFunctionPass::buildInitFunctionBody(mlir::func::FuncOp initFuncOp,
                                                 ArrayRef<VPU::TransformationsSplit> splits) {
    // Note: deduplication is automatic via hashing
    mlir::SetVector<InitResults::value_type, InitResults> initResults;

    const auto addInitResult = [&](const VPU::TransformationsSplit& split, mlir::Value value) {
        // init result is the input to main, so use main's argument to
        // "re-create" the result.
        auto projection = split.take(VPU::WeightsSeparationSchedule::Main);
        VPUX_THROW_UNLESS(value.getType() == projection.argType,
                          "The generated init schedule IR does not match main arguments: result #{0} has type '{1}' vs "
                          "expected type '{2}'",
                          initResults.size(), value.getType(), projection.argType);
        initResults.insert({VPU::ConstArg(projection), value});
    };

    VPU::WsArgumentCache initArgDeduplicator(initFuncOp);
    VPU::ConstOpConverter initConverter(initFuncOp, _log);

    // in init, we need to handle the output boundary - "dequantization" of
    // outputs has to be done
    vpux::VPU::IoBoundaryAdapter initIoAdaptor{/*wrapInput=*/&vpux::VPU::IoBoundaryAdapter::identity,
                                               /*wrapOutput=*/&castToStorageType};

    for (const auto& [idx, split] : splits | indexed) {
        auto projection = split.take(VPU::WeightsSeparationSchedule::Init);
        const auto uniqueLoc = appendLoc(split.getLoc(), "init_cst{0}", idx);
        auto initArg = initArgDeduplicator.addArgument(uniqueLoc, VPU::ConstArg(projection), projection.argType);
        auto valueInInit = initConverter.convertToIrForm(uniqueLoc, projection, initArg, initIoAdaptor,
                                                         VPU::WeightsSeparationSchedule::Init);
        addInitResult(split, valueInInit);
    }

    return {initArgDeduplicator, initResults.takeVector()};
}

std::tuple<mlir::func::FuncOp, VPU::WsArgumentCache, IntroduceInitFunctionPass::InitResults>
IntroduceInitFunctionPass::buildInitFunction(mlir::func::FuncOp mainFuncOp,
                                             const VPU::WeightsSeparationInfo& wsAnalysis) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder moduleBuilder(&getContext(), &builderLog);
    moduleBuilder.setInsertionPoint(mainFuncOp);

    auto [splits, initIsSliced] = collectSplitsAccordingToSettings(wsAnalysis);
    const std::string functionName = initIsSliced ? formatv("init_part{0}", _initPart).str() : "init";
    return buildInitFunction(moduleBuilder, appendLoc(mainFuncOp.getLoc(), functionName), splits, functionName);
}

IntroduceInitFunctionPass::SplitSlice IntroduceInitFunctionPass::collectSplitsAccordingToSettings(
        const VPU::WeightsSeparationInfo& wsAnalysis) {
    auto splits = wsAnalysis.getCollectedSplits();

    // when generating init, acknowledge init part and memory limit
    assert(_mode == WeightsSeparationMode::GenerateInit && "Init generation only happens in gen-init");
    VPUX_THROW_WHEN(splits.empty(), "Cannot generate empty init schedule");
    // Note: sort "globally" to prepare for slicing; use stable sort to not
    // shuffle "equal" elements; this plays nicely into the IR-based ordering
    // since WeightsSeparationInfo already "orders" data based on the IR
    // structure.
    std::stable_sort(splits.begin(), splits.end());

    auto slicedSplits = VPU::sliceAccordingToMemoryLimit(_log, splits, _memoryLimit);
    VPUX_THROW_WHEN((_initPart < 0 || _initPart >= checked_cast<int64_t>(slicedSplits.size())),
                    "Cannot generate init schedule part #{0}, only {1} parts are available", _initPart,
                    slicedSplits.size());
    return {slicedSplits[checked_cast<size_t>(_initPart)], /*initIsSliced=*/slicedSplits.size() > 1};
}

VPU::WsArgumentCache IntroduceInitFunctionPass::updateMainAndOutlinedFunctions(mlir::ModuleOp moduleOp,
                                                                               mlir::func::FuncOp mainFuncOp,
                                                                               const VPU::CallChainTree& tree) {
    // Traverse the call-chain tree, eagerly converting all constant operations
    // into VPU IR ops inside of the main / any outlined functions.
    VPU::MainFunctionUpdater mainUpdater(_log, moduleOp, [](Const::DeclareOp constOp) {
        return VPU::isSuitableForWeightlessCompilation(constOp, /*skipViewLikeOnly=*/true);
    });
    tree.apply(mainUpdater);

    return mainUpdater.takeArgCache(mainFuncOp);
}

void IntroduceInitFunctionPass::setNetworkEntryPointToMain(net::NetworkInfoOp netInfo,
                                                           const VPU::WsArgumentCache& topLevelMainArgCache) {
    // update network IO info
    auto& inputsRegion = netInfo.getInputsInfo();
    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);
    builder.setInsertionPointToEnd(&inputsRegion.front());

    for (auto it : topLevelMainArgCache.getSortedArgs()) {
        const auto& [entry, blockArg] = *it;
        // Note: init results match exactly the input arguments of main, so one
        // can "re-create" the unique name of init's result by taking the
        // respective main input argument.
        const auto inputName = entry.getUniqueName();
        builder.create<net::DataInfoOp>(appendLoc(netInfo.getLoc(), inputName), inputName, blockArg.getType());
    }

    verifyNetworkInfoWithRespectToNewArguments(netInfo);
}

mlir::LogicalResult IntroduceInitFunctionPass::initialize(mlir::MLIRContext*) {
    if (wsExtractionMode.hasValue()) {
        auto modeString = wsExtractionMode.getValue();

        if (modeString == "gen-main") {
            _mode = WeightsSeparationMode::GenerateMain;
        } else if (modeString == "gen-init") {
            _mode = WeightsSeparationMode::GenerateInit;
        } else {
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult IntroduceInitFunctionPass::deferredInitialize(mlir::ModuleOp moduleOp) {
    const auto limit = memoryLimit.hasValue() ? vpux::Byte(memoryLimit.getValue()) : DEFAULT_MEMORY_LIMIT;
    const int64_t initIndex = initPart.hasValue() ? initPart.getValue() : DEFAULT_INIT_PART;

    const bool limitSpecified = limit != DEFAULT_MEMORY_LIMIT;
    const bool initPartSpecified = initIndex != DEFAULT_INIT_PART;

    // verify correctness
    switch (_mode) {
    case WeightsSeparationMode::GenerateInit: {
        const bool validGenerateInit = (limitSpecified == initPartSpecified);
        if (!validGenerateInit) {
            moduleOp->emitError(
                    formatv("Both {0} and {1} should be either present or unspecified. {0} is: {2} and {1} is: {3}",
                            memoryLimit.getArgStr(), initPart.getArgStr(), limit, initIndex));
            return mlir::failure();
        }
        break;
    }
    case WeightsSeparationMode::GenerateMain: {
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

// Note: this is here for debugging purposes only.
struct TreePrinter final : VPU::CallChainTree::Visitor {
    std::ostream& stream;
    size_t indentation = 0;

    TreePrinter(std::ostream& s): stream(s) {
    }

    bool visit(const Node& node) final {
        auto funcOp = node.data().second;
        constexpr StringLiteral prefix = "|- ";
        stream << std::string(prefix.size() * indentation, ' ')
               << formatv("{0}{1}\n", prefix, funcOp.getSymName()).str();
        ++indentation;
        return true;
    }

    void endVisit(const Node&) final {
        --indentation;
    }
};

void eraseMainAndOutlinedFunctions(const VPU::CallChainTree& tree) {
    // Note: order of deletion does not seem to be important, but it is
    // important to avoid double-free. thus, use a set to uniquify functions.
    mlir::DenseSet<mlir::func::FuncOp> toBeErased;
    utils::CallbackVisitor<VPU::CallChainData> funcDeleter(
            [&](const VPU::CallChainTree::Node& node) {
                auto funcOp = node.data().second;
                toBeErased.insert(funcOp);
                return true;
            },
            nullptr);
    tree.apply(funcDeleter);

    for (auto funcOp : toBeErased) {
        funcOp.erase();
    }
}

void IntroduceInitFunctionPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    if (mlir::failed(deferredInitialize(moduleOp))) {
        signalPassFailure();
        return;
    }

    // This is a temporary workaround to avoid ShapeInfo-related compilation failures. Permanent solution will be
    // tracked in E#222554.
    moduleOp->removeAttr(IE::SHAPE_VERIFICATION_ENABLED);

    auto [netInfo, mainFuncOp] = net::getFromModule(moduleOp);

    auto infoOpt = getCachedAnalysis<VPU::WeightsSeparationInfo>();
    VPUX_THROW_UNLESS(infoOpt.has_value(), "VPU::WeightsSeparationInfo analysis must be cached");
    const auto& wsAnalysis = infoOpt->get();

    auto tree = VPU::getOutliningRepresentation(mainFuncOp);

    if (_log.isActive(LogLevel::Debug)) {
        std::stringstream stream;
        TreePrinter printer{stream};
        tree.apply(printer);
        _log.debug("The following operation tree is found:\n{0}\n", stream.str());
    }

    auto statisticsLogger = [&]() -> std::unique_ptr<InitSpecificLoggerBase> {
        if (_log.isActive(LogLevel::Info)) {
            return std::make_unique<InitSpecificMetaInfoLogger>(moduleOp, /*skipViewLikeOnly=*/true);
        }
        return std::make_unique<InitSpecificNullLogger>();
    }();

    switch (_mode) {
    case WeightsSeparationMode::GenerateInit: {
        auto [initFuncOp, initArgCache, initResults] = buildInitFunction(mainFuncOp, wsAnalysis);
        setNetworkEntryPointToInit(netInfo, initFuncOp, initArgCache, initResults);
        eraseMainAndOutlinedFunctions(tree);

        statisticsLogger->analyzeInitFunction(initFuncOp);
        statisticsLogger->print(_log);
        break;
    }
    case WeightsSeparationMode::GenerateMain: {
        auto mainArgCache = updateMainAndOutlinedFunctions(moduleOp, mainFuncOp, tree);
        setNetworkEntryPointToMain(netInfo, mainArgCache);
        break;
    }
    default: {
        // silence the unhandled case error by the compiler
        moduleOp->emitError("Encountered invalid mode: This should not happen!");
        signalPassFailure();
        break;
    }
    }
}

}  // namespace

//
// createIntroduceInitFunctionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createIntroduceInitFunctionPass(const Logger& log) {
    return std::make_unique<IntroduceInitFunctionPass>(log);
}

std::unique_ptr<mlir::Pass> vpux::VPU::createIntroduceInitFunctionPass(StringRef wsExtractionModeString,
                                                                       std::optional<int64_t> initPart,
                                                                       std::optional<Byte> limit, const Logger& log) {
    return std::make_unique<IntroduceInitFunctionPass>(wsExtractionModeString, initPart, limit, log);
}
