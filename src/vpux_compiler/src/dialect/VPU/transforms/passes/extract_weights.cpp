//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation_ir_modification.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/DenseMapInfo.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_EXTRACTWEIGHTS
#define GEN_PASS_DEF_EXTRACTWEIGHTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

//
// ExtractWeightsPass
//

using ExtractedWeightsResults = std::vector<std::tuple<VPU::ConstArg, mlir::Value>>;
class ExtractWeightsPass final : public VPU::impl::ExtractWeightsBase<ExtractWeightsPass> {
public:
    explicit ExtractWeightsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    ExtractedWeightsResults buildWeightsInMainFunction(mlir::OpBuilder& moduleBuilder,
                                                       ArrayRef<VPU::TransformationsSplit> splits);
    std::tuple<ExtractedWeightsResults, std::vector<VPU::TransformationsSplit>> extractWeightsToMainFunction(
            mlir::func::FuncOp mainFuncOp, const VPU::WeightsSeparationInfo& wsAnalysis);

private:
    void safeRunOnModule() final;
    Logger _log;
};

std::tuple<ExtractedWeightsResults, std::vector<VPU::TransformationsSplit>>
ExtractWeightsPass::extractWeightsToMainFunction(mlir::func::FuncOp mainFuncOp,
                                                 const VPU::WeightsSeparationInfo& wsAnalysis) {
    OpBuilderLogger builderLog(_log.nest());
    mlir::OpBuilder moduleBuilder(&getContext(), &builderLog);
    moduleBuilder.setInsertionPointToStart(&mainFuncOp.getFunctionBody().front());

    auto splits = wsAnalysis.getCollectedSplits();
    if (splits.empty()) {
        _log.trace("No weights to extract to main function");
        return {{}, {}};
    }
    auto extractedResults = buildWeightsInMainFunction(moduleBuilder, splits);
    return {extractedResults, splits};
}

// We cannot create IE/VPU dialect ops in the main function for host compilation so that
// transformations to Const ops are added
static mlir::Value transformToStorageType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                          const VPU::IoBoundaryAdapter::TypeInfo& info) {
    if (!info.valid()) {
        return input;
    }
    auto constOp = mlir::dyn_cast<Const::DeclareOp>(input.getDefiningOp());
    if (constOp == nullptr) {
        return input;
    }
    auto constAttr = constOp.transformContentAttr().castElemType(info.storageType).get();
    auto resultType = Const::inferFinalType(constAttr.getBaseContent().getType(), constAttr.getTransformations());
    auto newConstOp = builder.create<Const::DeclareOp>(loc, resultType, constAttr);
    constOp->replaceAllUsesWith(newConstOp);
    constOp->erase();
    return newConstOp;
}

ExtractedWeightsResults ExtractWeightsPass::buildWeightsInMainFunction(mlir::OpBuilder& moduleBuilder,
                                                                       ArrayRef<VPU::TransformationsSplit> splits) {
    // Note: deduplication is automatic via hashing
    mlir::SetVector<ExtractedWeightsResults::value_type, ExtractedWeightsResults> extractedResults;

    // in the func where weights are stored, we need to handle the output boundary - "dequantization" of
    // outputs has to be done
    vpux::VPU::IoBoundaryAdapter ioAdaptor{/*wrapInput=*/&vpux::VPU::IoBoundaryAdapter::identity,
                                           /*wrapOutput=*/&transformToStorageType};

    for (const auto& split : splits) {
        auto contentAttr = split.getContentAttr();
        auto transformations = split.getInitTransformations();

        auto resultAttr = Const::ContentAttr::get(contentAttr.getBaseContent(), transformations);
        auto resultType = Const::inferFinalType(contentAttr.getBaseContent().getType(), transformations);
        auto constOp =
                moduleBuilder.create<Const::DeclareOp>(appendLoc(split.getLoc(), "_extracted"), resultType, resultAttr);

        auto resultValue = ioAdaptor.wrapOutput(moduleBuilder, appendLoc(split.getLoc(), "dequant"), constOp,
                                                split.getIoTypeInfo());
        extractedResults.insert(
                {VPU::ConstArg(mlir::dyn_cast<mlir::DenseResourceElementsAttr>(contentAttr.getBaseContent()),
                               transformations),
                 resultValue});
    }

    return extractedResults.takeVector();
}

class KernelFuncsUpdater final : public MainFunctionUpdater {
    DenseMap<ConstArg, mlir::Value> _constArgToConstValue;

    bool isEntryPointFunc(mlir::func::FuncOp funcOp) const {
        auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
        assert(moduleOp != nullptr && "All functions are expected to be contained in a module");
        auto entryPointFuncOp = net::getFromModule(moduleOp).second;
        return entryPointFuncOp == funcOp;
    }

    bool visit(const Node& node) override {
        auto currOp = node.data().second;
        if (isEntryPointFunc(currOp)) {
            return true;
        }
        return MainFunctionUpdater::visit(node);
    }

    void endVisit(const Node& node) override {
        auto currOp = node.data().second;
        auto bodyBlock = &currOp.getFunctionBody().front();
        OpBuilderLogger builderLog(_log.nest());
        auto currBuilder = mlir::OpBuilder::atBlockBegin(bodyBlock, &builderLog);

        for (const auto& child : node.children()) {
            auto [callOp, childOp] = child.data();
            hoistCalleeArgsToCaller(currOp, childOp);
            fixCallSite(currBuilder, currOp, childOp, callOp);
        }

        if (isEntryPointFunc(currOp)) {
            // No need to update the signature of entry point function, since we are not adding any new arguments to it
            _log.trace("Finished visiting entryPoint func {0}", currOp.getSymName());
            return;
        }

        // Since argument propagation is done, function signature could be
        // updated.
        const auto mainFuncResults = currOp.getFunctionType().getResults();
        currOp.setFunctionType(
                mlir::FunctionType::get(currOp.getContext(), bodyBlock->getArgumentTypes(), mainFuncResults));

        _log.trace("Finished visiting {0}", currOp.getSymName());
    }

    void fixCallSite(mlir::OpBuilder& callerBuilder, mlir::func::FuncOp callerOp, mlir::func::FuncOp calleeOp,
                     mlir::func::CallOp oldCall) override {
        if (isEntryPointFunc(callerOp)) {
            const auto& calleeDeduplicator = _argCaches.at(calleeOp);

            auto newCallArguments = to_std_vector(oldCall.getOperands());
            const auto& afterInitCalleeArgs = calleeDeduplicator.getSortedArgs();
            // old arguments remain "as is", new arguments are appended
            newCallArguments.resize(newCallArguments.size() + afterInitCalleeArgs.size());
            for (auto it : afterInitCalleeArgs) {
                const auto& [entry, calleeArg] = *it;
                // We have not added any new arguments for the entry point function in hoistCalleeArgsToCaller,
                // so new caller argument is expected to be found in the map of extracted constants
                auto callerArg = _constArgToConstValue.at(entry);
                assert(calleeArg.getArgNumber() >= oldCall.getOperands().size() &&
                       "Call-site is invalidated: added arguments must always be present after the original ones");
                newCallArguments[calleeArg.getArgNumber()] = callerArg;
            }

            callerBuilder.setInsertionPoint(oldCall);
            auto newCall = callerBuilder.create<mlir::func::CallOp>(oldCall.getLoc(), calleeOp, newCallArguments);
            oldCall.replaceAllUsesWith(newCall.getResults());
            oldCall->erase();
        } else {
            MainFunctionUpdater::fixCallSite(callerBuilder, callerOp, calleeOp, oldCall);
        }
    }

    void hoistCalleeArgsToCaller(mlir::func::FuncOp callerOp, mlir::func::FuncOp calleeOp) override {
        if (isEntryPointFunc(callerOp)) {
            return;
        }
        MainFunctionUpdater::hoistCalleeArgsToCaller(callerOp, calleeOp);
    }

public:
    KernelFuncsUpdater(const Logger& log, mlir::ModuleOp moduleOp,
                       DenseMap<ConstArg, mlir::Value>&& constArgToConstValue, VPU::IsWorthyToCollect _isWorthy)
            : MainFunctionUpdater(log, moduleOp, std::move(_isWorthy)),
              _constArgToConstValue(std::move(constArgToConstValue)) {
    }
};

void ExtractWeightsPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    VPU::WeightsSeparationInfo::Options options;
    options.weightsAnalysisMode = VPU::WeightsSeparationInfo::Options::WeightsAnalysisMode::HostCompile;
    VPU::WeightsSeparationInfo::setOptions(moduleOp, options);

    auto mainFuncOp = net::getFromModule(moduleOp).second;
    const auto& wsAnalysis = getAnalysis<VPU::WeightsSeparationInfo>();
    mlir::DenseSet<Const::ContentAttr> constsToExtract;
    for (const auto& split : wsAnalysis.getCollectedSplits()) {
        constsToExtract.insert(split.getContentAttr());
    }
    auto tree = VPU::getOutliningRepresentation(mainFuncOp);

    auto [extractedResults, splits] = extractWeightsToMainFunction(mainFuncOp, wsAnalysis);
    if (extractedResults.empty() || splits.empty()) {
        _log.trace("No weights extracted to main function, skipping updating kernel functions");
        return;
    }
    _log.debug("Extracted the following weights to main function:");
    for (const auto& [arg, value] : extractedResults) {
        _log.nest().debug("arg: {0}; value: {1}", arg, value);
    }

    mlir::DenseMap<ConstArg, mlir::Value> constArgToConstValue;
    for (const auto& [arg, value] : extractedResults) {
        constArgToConstValue.insert({/*VPU::ConstArg=*/arg, /*mlir::Value=*/value});
    }

    const auto isWorthy = [&](Const::DeclareOp constOp) {
        return constsToExtract.contains(constOp.getContentAttr());
    };
    KernelFuncsUpdater kernelUpdater(_log, moduleOp, std::move(constArgToConstValue), isWorthy);
    tree.apply(kernelUpdater);
}

//
// createExtractWeightsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createExtractWeightsPass(Logger log) {
    return std::make_unique<ExtractWeightsPass>(log);
}
