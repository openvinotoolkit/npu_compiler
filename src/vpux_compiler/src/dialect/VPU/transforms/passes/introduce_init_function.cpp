//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/abstract_tree.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/ir_modification.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/IR/Type.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LogicalResult.h>

#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <iterator>

namespace vpux::VPU {
#define GEN_PASS_DECL_INTRODUCEINITFUNCTION
#define GEN_PASS_DEF_INTRODUCEINITFUNCTION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

// Casts the resulting value to its storage type counterpart. This is normally
// done in init and thus in IE dialect.
static mlir::Value castToStorageType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                     const VPU::IoBoundaryAdapter::TypeInfo& info) {
    if (!info.valid()) {
        return input;
    }
    return builder.create<IE::QuantizeCastOp>(appendLoc(loc, "_quant_cast"), input, info.storageType);
}

// Casts the resulting value "back" to its original quantized type. This is
// normally done in main and thus in VPU dialect.
static mlir::Value castToQuantizedType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                       const VPU::IoBoundaryAdapter::TypeInfo& info) {
    if (!info.valid()) {
        return input;
    }
    return builder.create<VPU::QuantizeCastOp>(appendLoc(loc, "_quant_cast"), input, info.quantizedType);
}

/// Weights-separation specific argument cache.
using WsArgumentCache = vpux::utils::ArgumentCache<vpux::VPU::ConstArg>;

// Each DeclareOp corresponds to a single result value of init. A result value can belong to multiple DeclareOps if for
// example their underlying elements attributes and in-init-transformations are the same.
class DeclareOpOutputMap {
public:
    // We take this awkward looking ArrayRef of tuples instead of a DenseMap<mlir::Value, SmallVector<Const::DeclareOp>>
    // directly, so we can be sure about a deterministic order of result values that only depends on the IR itself and
    // not the context.
    DeclareOpOutputMap(ArrayRef<std::tuple<mlir::Value, Const::ContentAttr>> transformedConstants) {
        // Because of CSE-caching, multiple DeclareOps can be associated with the same result value. This is why
        // we want set semantics here.
        for (auto [resultValue, contentAttr] : transformedConstants) {
            // Note: SetVector's insertion peforms de-duplicating push_back.
            // This is why this algorithm works: we add result values in the
            // order they appear in the sequence passed to us as input.
            _resultValues.insert(resultValue);
        }

        // temporary map for easy lookup
        DenseMap<mlir::Value, SmallVector<Const::ContentAttr>> valueToContentAttrs;
        for (auto [resultValue, contentAttr] : transformedConstants) {
            valueToContentAttrs[resultValue].push_back(contentAttr);
        }

        _resultTypes.resize(_resultValues.size());
        _resultNames.resize(_resultValues.size());
        _transformHashes.resize(_resultValues.size());

        for (auto [resultIndex, resultValue] : _resultValues.getArrayRef() | indexed) {
            const auto& contentAttrs = valueToContentAttrs.at(resultValue);

            _resultTypes[resultIndex] = resultValue.getType();
            _resultNames[resultIndex] = getResourceName(contentAttrs.front().getBaseContent());
            assert(!_resultNames[resultIndex].empty() && "Only dense_resource<> constants are processed by the pass");

            // E#155816: there's a bug here - we cannot use transformation hash
            // as it is done here, consider:
            // * dense_resource<ov_1>, [#const.Add<1.0>, #const.SubView<[0, 0], [0, 0]>]
            // and:
            // * dense_resource<ov_1>, [#const.Add<1.0>, #const.SubView<[0, 1], [0, 0]>]
            // (both map to a single result value of init)
            _transformHashes[resultIndex] = static_cast<unsigned>(contentAttrs.front().getTransformationHash());
        }
    }

    ArrayRef<mlir::Type> getResultTypes() const {
        return _resultTypes;
    }

    ArrayRef<mlir::Value> getResultValues() const {
        return _resultValues.getArrayRef();
    }

    std::string getUniqueResultName(size_t index) const {
        return formatv("out_{0}_hash_{1}", _resultNames[index], _transformHashes[index]);
    }

private:
    mlir::SetVector<mlir::Value> _resultValues;
    mlir::SmallVector<mlir::Type> _resultTypes;
    // Note: result names are dense_resource keys - returned already as string
    // refs by MLIR API (their lifetime is bound to MLIR context).
    mlir::SmallVector<mlir::StringRef> _resultNames;
    mlir::SmallVector<unsigned> _transformHashes;
};

using CallChainData = std::pair<mlir::func::CallOp, mlir::func::FuncOp>;
using CallChainTree = utils::AbstractTree<CallChainData>;

std::vector<CallChainData> collectCallChains(mlir::func::FuncOp funcOp) {
    std::vector<CallChainData> functions;
    funcOp.walk([&](mlir::func::CallOp callOp) {
        functions.push_back({callOp, getCalledFunction(callOp)});
    });
    return functions;
}

//
// IntroduceInitFunctionPass
//

class IntroduceInitFunctionPass final : public VPU::impl::IntroduceInitFunctionBase<IntroduceInitFunctionPass> {
public:
    enum class Mode { Unspecified, GenerateMain, GenerateInit, GenerateAll };

    explicit IntroduceInitFunctionPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    std::tuple<mlir::func::FuncOp, DeclareOpOutputMap> buildInitFunction(
            mlir::func::FuncOp mainFuncOp, const CallChainTree& tree,
            mlir::DenseMap<mlir::func::FuncOp, WsArgumentCache>& argCaches);

    // configures NetworkInfo to assume init-schedule is the entry-point.
    void setNetworkEntryPointToInit(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                    const WsArgumentCache& argCache, const DeclareOpOutputMap& outputMap);
    // configures NetworkInfo to assume *updated* main-schedule is the
    // entry-point. the behaviour is to be considered equivalent to setting the
    // entry-point to init.
    void setNetworkEntryPointToMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                    const DeclareOpOutputMap& outputMap);
    // creates new main that calls init and main in sequence. this function
    // becomes the new entry-point.
    void buildWrapperOpForInitAndMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp mainFuncOp,
                                      mlir::func::FuncOp initFuncOp, const WsArgumentCache& initArgCache);

    mlir::LogicalResult initialize(mlir::MLIRContext* context) final;
    void safeRunOnModule() final;

    Mode _mode = Mode::Unspecified;
};

void IntroduceInitFunctionPass::setNetworkEntryPointToInit(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                                           const WsArgumentCache& argCache,
                                                           const DeclareOpOutputMap& outputMap) {
    mainInfo.setEntryPoint(initFuncOp.getSymName());

    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);

    // update input types
    auto& inputsRegion = mainInfo.getInputsInfo();
    inputsRegion.getBlocks().clear();
    inputsRegion.getBlocks().push_back(new mlir::Block());
    builder.setInsertionPointToStart(&inputsRegion.front());

    for (auto it : argCache.getSortedArgs()) {
        const auto& [entry, blockArg] = *it;
        const auto type = blockArg.getType();
        const auto name = getResourceName(entry.content);
        auto inputName = mlir::StringAttr::get(&getContext(), formatv("in_{0}", name));
        builder.create<IE::DataInfoOp>(appendLoc(mainInfo.getLoc(), inputName), inputName, type,
                                       /*OptionalAttr originalShape*/ nullptr,
                                       /*OptionalAttr friendlyName*/ nullptr,
                                       /*OptionalAttr inputName*/ nullptr,
                                       /*OptionalAttr tensorNames*/ nullptr,
                                       /*profilingSectionsCount=*/0);
    }

    // update output types
    auto& outputsRegion = mainInfo.getOutputsInfo();
    outputsRegion.getBlocks().clear();
    outputsRegion.getBlocks().push_back(new mlir::Block());
    builder.setInsertionPointToStart(&outputsRegion.front());

    for (auto [index, type] : outputMap.getResultTypes() | indexed) {
        auto outputName = outputMap.getUniqueResultName(index);
        builder.create<IE::DataInfoOp>(appendLoc(mainInfo.getLoc(), outputName), outputName, type,
                                       /*OptionalAttr originalShape*/ nullptr,
                                       /*OptionalAttr friendlyName*/ nullptr,
                                       /*OptionalAttr inputName*/ nullptr,
                                       /*OptionalAttr tensorNames*/ nullptr,
                                       /*profilingSectionsCount=*/0);
    }
}

// A single-visit (with respect to functions) visitor for the call-chain tree
// that finishes building the IR for "main" functions and propagates arguments
// across call-chains.
class SingleShotMainUpdater final : public CallChainTree::Visitor {
    Logger _log;
    mlir::DenseMap<mlir::func::FuncOp, WsArgumentCache>& _argCaches;
    VPU::ConstOpConverter& _globalInitConverter;

    mlir::DenseSet<mlir::func::FuncOp> _visitationCache;

    // Appends new function arguments of callee to the caller's arguments.
    // During weights separation, a function's inner constants become inputs.
    // This happens across the call-chain and thus a caller must forward
    // callee's arguments from itself. This function would ensure that new
    // arguments of the callee appear in the caller's arguments.
    void hoistCalleeArgsToCaller(mlir::func::FuncOp callerOp, mlir::func::FuncOp calleeOp) {
        auto& callerArgDeduplicator = getNonConstArgCache(callerOp);
        const auto& calleeArgDeduplicator = _argCaches.at(calleeOp);
        for (auto it : calleeArgDeduplicator.getSortedArgs()) {
            const auto& [entry, blockArg] = *it;
            // propagate the argument from child to parent - there's nothing
            // else that has to be done just yet
            std::ignore = callerArgDeduplicator.addArgument(entry, blockArg.getType());
        }
    }

    // Updates the call-site according to the callee's modified arguments.
    // During weights separation, a function's inner constants become inputs.
    // This helper function would set up the call-site inside the caller to
    // correctly propagate arguments from the caller to the callee.
    void fixCallSite(mlir::OpBuilder& callerBuilder, mlir::func::FuncOp callerOp, mlir::func::FuncOp calleeOp,
                     mlir::func::CallOp oldCall) {
        const auto& callerDeduplicator = _argCaches.at(callerOp);
        const auto& calleeDeduplicator = _argCaches.at(calleeOp);

        auto newCallArguments = to_std_vector(oldCall.getOperands());
        const auto& afterInitCalleeArgs = calleeDeduplicator.getSortedArgs();
        // old arguments remain "as is", new arguments are appended
        newCallArguments.resize(newCallArguments.size() + afterInitCalleeArgs.size());
        for (auto it : afterInitCalleeArgs) {
            const auto& [entry, calleeArg] = *it;
            auto callerArg = callerDeduplicator.findArgument(entry);
            assert(calleeArg.getArgNumber() >= oldCall.getOperands().size() &&
                   "Call-site is invalidated: added arguments must always be present after the original ones");
            newCallArguments[calleeArg.getArgNumber()] = callerArg;
        }

        callerBuilder.setInsertionPoint(oldCall);
        auto newCall = callerBuilder.create<mlir::func::CallOp>(oldCall.getLoc(), calleeOp, newCallArguments);
        oldCall.replaceAllUsesWith(newCall.getResults());
        oldCall->erase();
    }

    bool hasSeenThisFunction(mlir::func::FuncOp op) {
        const bool firstOccurence = _visitationCache.insert(op).second;
        return !firstOccurence;
    }

    WsArgumentCache& getNonConstArgCache(mlir::func::FuncOp funcOp) {
        assert(_argCaches.contains(funcOp) && "Argument caches must already be set up and be functional");
        return _argCaches.find(funcOp)->second;
    }

public:
    SingleShotMainUpdater(Logger log, mlir::DenseMap<mlir::func::FuncOp, WsArgumentCache>& argCaches,
                          VPU::ConstOpConverter& initOpConverter)
            : _log(log), _argCaches(argCaches), _globalInitConverter(initOpConverter) {
    }

    bool visit(const Node& node) final {
        auto currOp = node.data().second;
        if (hasSeenThisFunction(node.data().second)) {
            // nothing to do
            return false;
        }

        // when visiting the function, move constant operations to init and
        // update IR inside the function accordingly.

        const auto splits = VPU::collectMoveWorthyTransformationSplits(currOp);
        if (_log.isActive(LogLevel::Trace)) {
            _log.trace("Found the following constants in {0}:", currOp.getSymName());
            for (const auto& [index, split] : splits | indexed) {
                _log.trace("  {0}: {1}", index, split.declareOp());
            }
        }

        // in init, we need to handle the output boundary - "dequantization" of
        // outputs has to be done
        vpux::VPU::IoBoundaryAdapter initIoAdaptor{/*wrapInput=*/&vpux::VPU::IoBoundaryAdapter::identity,
                                                   /*wrapOutput=*/&castToStorageType};
        auto& initArgDeduplicator = getNonConstArgCache(_globalInitConverter.getFunction());

        // in main we only care about input boundary - "quantization" has to be
        // done on input arguments to restore real types.
        vpux::VPU::IoBoundaryAdapter mainIoAdaptor{/*wrapInput=*/&castToQuantizedType,
                                                   /*wrapOutput=*/&vpux::VPU::IoBoundaryAdapter::identity};
        auto& mainArgDeduplicator = getNonConstArgCache(currOp);

        // Note: created externally once to ensure operation builder has correct
        // insertion point.
        VPU::ConstOpConverter funcConverter(currOp, _log);

        // Note: removal of operations is done separately, after construction of
        // new IR, to ensure that operation builder is not invalidated.
        SmallVector<Const::DeclareOp> opsToRemove;

        for (const auto& [index, split] : splits | indexed) {
            auto declareOp = split.declareOp();
            const auto baseLoc = appendLoc(currOp.getLoc(), "cst{0}", index);

            // init part
            {
                _log.trace("Converting '{0}' to IR in init", declareOp);
                auto projection = split.take(VPU::WeightsSeparationSchedule::Init);
                auto initArg = initArgDeduplicator.addArgument(VPU::ConstArg(projection), projection.argType);
                auto valueInInit =
                        _globalInitConverter.convertToIrForm(appendLoc(baseLoc, "init"), projection, initArg,
                                                             initIoAdaptor, VPU::WeightsSeparationSchedule::Init);
                _log.trace("Transformations of '{0}' resulted in '{1}' in init", declareOp, valueInInit);
            }

            // main part
            {
                _log.trace("Converting rest of '{0}' to IR in main", declareOp);
                auto projection = split.take(VPU::WeightsSeparationSchedule::Main);
                auto mainArg = mainArgDeduplicator.addArgument(VPU::ConstArg(projection), projection.argType);
                auto valueInMain = funcConverter.convertToIrForm(appendLoc(baseLoc, "main"), projection, mainArg,
                                                                 mainIoAdaptor, VPU::WeightsSeparationSchedule::Main);

                _log.trace("Replacing '{0}' with '{1}'", declareOp, valueInMain);
                declareOp.replaceAllUsesWith(valueInMain);
                opsToRemove.push_back(declareOp);
            }
        }

        for (auto op : opsToRemove) {
            op.erase();
        }

        return true;
    }

    void endVisit(const Node& node) final {
        auto currOp = node.data().second;
        auto bodyBlock = &currOp.getFunctionBody().front();
        OpBuilderLogger builderLog(_log.nest());
        auto currBuilder = mlir::OpBuilder::atBlockBegin(bodyBlock, &builderLog);

        // At the end of the visitation, it is certain that the children are
        // already processed (by definition of the procedure). Thus, we can
        // forward children's arguments up the call-chain and fix the calls to
        // children accordingly:
        // ```
        // func.func foo() {
        //   call bar()
        // }
        // ```
        // becomes:
        // ```
        // func.func foo(%bar_cst: ...) {
        //   call bar(%bar_cst)
        // }
        // ```
        for (const auto& child : node.children()) {
            auto [callOp, childOp] = child.data();
            hoistCalleeArgsToCaller(currOp, childOp);
            fixCallSite(currBuilder, currOp, childOp, callOp);
        }

        // Since argument propagation is done, function signature could be
        // updated.
        const auto mainFuncResults = currOp.getFunctionType().getResults();
        // in "main", only inputs change
        currOp.setFunctionType(
                mlir::FunctionType::get(currOp.getContext(), bodyBlock->getArgumentTypes(), mainFuncResults));

        _log.trace("Finished visiting {0} and its subtree", currOp.getSymName());
    }
};

std::tuple<mlir::func::FuncOp, DeclareOpOutputMap> IntroduceInitFunctionPass::buildInitFunction(
        mlir::func::FuncOp mainFuncOp, const CallChainTree& tree,
        mlir::DenseMap<mlir::func::FuncOp, WsArgumentCache>& argCaches) {
    OpBuilderLogger builderLog(_log.nest());

    // create empty @init() : () -> ()
    auto initFuncOp = [&]() {
        mlir::OpBuilder moduleBuilder(&getContext(), &builderLog);
        moduleBuilder.setInsertionPoint(mainFuncOp);
        auto initLoc = appendLoc(mainFuncOp.getLoc(), "_init");
        auto initFuncType = mlir::FunctionType::get(&getContext(), {}, {});
        return moduleBuilder.create<mlir::func::FuncOp>(initLoc, "init", initFuncType);
    }();
    auto bodyBlock = initFuncOp.addEntryBlock();

    argCaches.insert({initFuncOp, WsArgumentCache(initFuncOp)});
    // Traverse the call-chain tree, eagerly converting all constant operations
    // into IR ops (either in IE or VPU).
    VPU::ConstOpConverter globalInitConverter(initFuncOp, _log);
    SingleShotMainUpdater updater(_log, argCaches, globalInitConverter);
    tree.apply(updater);

    auto outputMap = DeclareOpOutputMap(globalInitConverter.getConvertedConsts());
    auto initBuilder = mlir::OpBuilder::atBlockEnd(bodyBlock, &builderLog);
    initBuilder.create<mlir::func::ReturnOp>(appendLoc(initFuncOp.getLoc(), "_return"), outputMap.getResultValues());
    auto initFuncType =
            mlir::FunctionType::get(&getContext(), bodyBlock->getArgumentTypes(), outputMap.getResultTypes());
    initFuncOp.setFunctionType(initFuncType);

    return {initFuncOp, outputMap};
}

void IntroduceInitFunctionPass::setNetworkEntryPointToMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp initFuncOp,
                                                           const DeclareOpOutputMap& outputMap) {
    // update network IO info
    auto& inputsRegion = mainInfo.getInputsInfo();
    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);
    builder.setInsertionPointToEnd(&inputsRegion.front());

    const auto initFuncType = initFuncOp.getFunctionType();
    for (auto [index, type] : initFuncType.getResults() | indexed) {
        auto name = mlir::StringAttr::get(&getContext(), outputMap.getUniqueResultName(index));
        builder.create<IE::DataInfoOp>(appendLoc(mainInfo.getLoc(), name), name, type,
                                       /*OptionalAttr originalShape*/ nullptr,
                                       /*OptionalAttr friendlyName*/ nullptr,
                                       /*OptionalAttr inputName*/ nullptr,
                                       /*OptionalAttr tensorNames*/ nullptr,
                                       /*profilingSectionsCount=*/0);
    }
}

void IntroduceInitFunctionPass::buildWrapperOpForInitAndMain(IE::CNNNetworkOp mainInfo, mlir::func::FuncOp mainFuncOp,
                                                             mlir::func::FuncOp initFuncOp,
                                                             const WsArgumentCache& initArgCache) {
    const auto mainFuncType = mainFuncOp.getFunctionType();
    const auto initFuncType = initFuncOp.getFunctionType();
    // Note: expect the below to never fail
    VPUX_THROW_WHEN(mainFuncType.getNumInputs() < initFuncType.getNumResults(),
                    "Main must be already updated to accept all init's outputs as additional inputs");
    // inputs of main are original inputs + init outputs:
    const auto inputs = mainFuncType.getInputs().drop_back(initFuncOp.getFunctionType().getNumResults());
    // results of main are untouched
    const auto results = mainFuncType.getResults();

    OpBuilderLogger builderLog(_log.nest());
    auto wrapperFuncOp = [&]() {
        mlir::OpBuilder moduleBuilder(&getContext(), &builderLog);
        moduleBuilder.setInsertionPointAfter(mainFuncOp);
        auto loc = appendLoc(mainFuncOp.getLoc(), "_wrapper");
        auto wrapperFuncType = mlir::FunctionType::get(&getContext(), inputs, results);
        return moduleBuilder.create<mlir::func::FuncOp>(loc, ("wrapper_" + mainFuncOp.getSymName()).str(),
                                                        wrapperFuncType);
    }();

    const auto locBase = wrapperFuncOp.getLoc();
    auto bodyBlock = wrapperFuncOp.addEntryBlock();
    auto builder = mlir::OpBuilder::atBlockEnd(bodyBlock, &builderLog);

    // create the declare ops without their transformations
    SmallVector<mlir::Value> inputValues;
    inputValues.reserve(initFuncType.getNumInputs());
    for (auto it : initArgCache.getSortedArgs()) {
        const auto& [entry, blockArg] = *it;
        auto baseContent = entry.content;
        inputValues.push_back(builder.create<Const::DeclareOp>(appendLoc(locBase, "_cst_{0}", blockArg.getArgNumber()),
                                                               baseContent.getType(),
                                                               Const::ContentAttr::get(baseContent)));
    }

    auto initCallOp = builder.create<mlir::func::CallOp>(appendLoc(locBase, "_call_init"), initFuncOp.getSymNameAttr(),
                                                         initFuncType.getResults(), inputValues);

    auto mainInputValues = [&]() {
        const auto blockArgs = bodyBlock->getArguments();
        const auto initResults = initCallOp.getResults();

        // main arguments go as follows: first original arguments (unmodified),
        // then new arguments (results of init)
        SmallVector<mlir::Value> values;
        values.reserve(bodyBlock->getNumArguments() + initCallOp.getNumResults());
        values.append(blockArgs.begin(), blockArgs.end());
        values.append(initResults.begin(), initResults.end());
        return values;
    }();
    builder.setInsertionPointAfter(initCallOp);
    auto mainCallOp = builder.create<mlir::func::CallOp>(appendLoc(locBase, "_call_main"), mainFuncOp.getSymNameAttr(),
                                                         mainFuncType.getResults(), mainInputValues);

    builder.setInsertionPointToEnd(bodyBlock);
    builder.create<mlir::func::ReturnOp>(appendLoc(locBase, "_return"), mainCallOp.getResults());

    mainInfo.setEntryPoint(wrapperFuncOp.getSymName());
}

mlir::LogicalResult IntroduceInitFunctionPass::initialize(mlir::MLIRContext*) {
    if (extractionMode.hasValue()) {
        auto modeString = extractionMode.getValue();

        if (modeString == "gen-main") {
            _mode = Mode::GenerateMain;
        } else if (modeString == "gen-init") {
            _mode = Mode::GenerateInit;
        } else if (modeString == "gen-all") {
            _mode = Mode::GenerateAll;
        } else {
            return mlir::failure();
        }
    }

    return mlir::success();
}

std::vector<CallChainData> findChildren(const CallChainTree::Node& node) {
    auto funcOp = node.data().second;
    auto chains = collectCallChains(funcOp);
    // Note: sort call-chains lexicographically (using function names) to ensure
    // outlining-independent processing. while this disregards the call
    // sequence, this allows to avoid differences in schedule generation when
    // independent calls get reordered in IR:
    // ```cpp
    //  %call1 = call @foo1(...)
    //  %call2 = call @foo2(...)
    //  // vs:
    //  %call2 = call @foo2(...)
    //  %call1 = call @foo1(...)
    //
    //  // independent usage of calls:
    //  %op1 = VPU.Convolution(%call1)
    //  %ops2 = VPU.Convolution(%call2)
    // ```
    std::sort(chains.begin(), chains.end(), [](const CallChainData& x, const CallChainData& y) {
        auto xFunc = x.second;
        auto yFunc = y.second;
        // lexicographical comparison
        return xFunc.getSymName() < yFunc.getSymName();
    });

    return chains;
}

// Note: this is here for debugging purposes only.
struct TreePrinter final : CallChainTree::Visitor {
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

void eraseMainAndOutlinedFunctions(const CallChainTree& tree) {
    // Note: order of deletion does not seem to be important, but it is
    // important to avoid double-free. thus, use a set to uniquify functions.
    mlir::DenseSet<mlir::func::FuncOp> toBeErased;
    utils::CallbackVisitor<CallChainData> funcDeleter(
            [&](const CallChainTree::Node& node) {
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

    IE::CNNNetworkOp mainInfo;
    mlir::func::FuncOp mainFuncOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, mainInfo, mainFuncOp);

    // construct a call-chain tree that represents the outlining structure. an
    // example of such a tree is:
    // ```
    // |- {nullptr, main}
    //    |- {"call foo1", foo1}
    //       |- {"call foo2", foo2}
    //    |- {"call foo3", foo3}
    // ```
    // where "call fooX" is a CallOp operation inside the respective function
    // and fooX is a standalone function produced by the outlining.
    CallChainTree tree({CallChainTree::Node(CallChainData{nullptr, mainFuncOp}, {})}, findChildren);

    if (_log.isActive(LogLevel::Debug)) {
        std::stringstream stream;
        TreePrinter printer{stream};
        tree.apply(printer);
        _log.debug("The following operation tree is found:\n{0}\n", stream.str());
    }

    auto argCaches = [&]() {
        mlir::DenseMap<mlir::func::FuncOp, WsArgumentCache> cache;
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            cache.insert({funcOp, WsArgumentCache(funcOp)});
        });
        return cache;
    }();

    auto [initFuncOp, outputMap] = buildInitFunction(mainFuncOp, tree, argCaches);
    auto initFuncType = initFuncOp.getFunctionType();

    if (_log.isActive(LogLevel::Debug)) {
        int64_t inByteCount = 0;
        int64_t outByteCount = 0;

        for (auto type : initFuncType.getInputs()) {
            inByteCount += mlir::cast<NDTypeInterface>(type).getTotalAllocSize().count();
        }

        for (auto type : initFuncType.getResults()) {
            outByteCount += mlir::cast<NDTypeInterface>(type).getTotalAllocSize().count();
        }

        auto statsLogger = _log.nest("init() stats", 1);
        statsLogger.debug("Argument count: {0}", initFuncType.getNumInputs());
        statsLogger.debug("Result  count: {0}", initFuncType.getNumResults());
        statsLogger.debug("In-byte count: {0}", inByteCount);
        statsLogger.debug("Out-byte count {0}", outByteCount);
        statsLogger.debug("Signature: {0}", initFuncType);
    }

    switch (_mode) {
    case Mode::GenerateInit:
        setNetworkEntryPointToInit(mainInfo, initFuncOp, argCaches.at(initFuncOp), outputMap);
        eraseMainAndOutlinedFunctions(tree);
        break;
    case Mode::GenerateMain:
        setNetworkEntryPointToMain(mainInfo, initFuncOp, outputMap);
        initFuncOp.erase();
        break;
    case Mode::GenerateAll:
        initFuncOp.setPrivate();
        mainFuncOp.setPrivate();
        buildWrapperOpForInitAndMain(mainInfo, mainFuncOp, initFuncOp, argCaches.at(initFuncOp));
        break;
    default:
        // silence the unhandled case error by the compiler
        moduleOp->emitError("Encountered invalid mode: This should not happen!");
        signalPassFailure();
        break;
    }
}

}  // namespace

//
// createIntroduceInitFunctionPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createIntroduceInitFunctionPass(const Logger& log) {
    return std::make_unique<IntroduceInitFunctionPass>(log);
}
