//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Visitors.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_VERTICALFUSIONOUTLINING
#define GEN_PASS_DEF_VERTICALFUSIONOUTLINING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace outliner {

//
// VerticalFusionOutliner
//

class VerticalFusionOutliner final : public OutlinerBase {
public:
    VerticalFusionOutliner(size_t numInstanceThreshold, size_t verticalFusionTileThreshold, const Logger& log)
            : OutlinerBase(std::make_unique<FunctionOutlinerVerticalFusion>(numInstanceThreshold,
                                                                            verticalFusionTileThreshold, log),
                           log) {
    }

    static constexpr StringRef name() {
        return "vertical-fusion";
    }

private:
    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override;

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override;
};

void VerticalFusionOutliner::buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                          ArrayRef<OutliningInstance> outlinedTargets) {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    auto builder = mlir::OpBuilder(moduleOp.getBodyRegion());
    builder.setInsertionPoint(netFunc);

    auto* ctx = moduleOp.getContext();
    auto isPureVFRegion = [](const IRSlice& slice) {
        const auto vfNum = llvm::count_if(slice.operations, [](auto* op) {
            return mlir::isa_and_nonnull<VPU::VerticalFusionOp>(op);
        });
        const auto clusteredOpNum = llvm::count_if(slice.operations, [](auto* op) {
            return mlir::isa_and_nonnull<VPU::ClusteredOpInterface>(op);
        });
        return vfNum == 1 && clusteredOpNum == 0;
    };
    for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
        const auto& slice = slices.front();
        const size_t sliceIdx = 0;
        const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                      ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
        const auto funcLoc = appendLoc(netFunc.getLoc(), "_part{0}", targetIdx + 1);
        auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcName, funcType);
        if (config::getWeightsTableReuseMode(func) == WeightsTableReuseMode::VF_ENABLED && isPureVFRegion(slice)) {
            func->setAttr(VPU::PureVerticalFusionRegionAttrName, mlir::UnitAttr::get(ctx));
        }
        func.setPrivate();
        OpBuilderLogger builderLog(getLogger().nest());
        auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

        mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
        for (size_t i = 0; i < slice.inputs.size(); i++) {
            oldToNewMap[slice.inputs[i]] = func.getArgument(i);
        }
        for (const auto op : slice.operations) {
            mlir::IRMapping mapper;
            for (auto operand : op->getOperands()) {
                mapper.map(operand, oldToNewMap[operand]);
            }
            auto clonedOp = builder.clone(*op, mapper);

            for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
                oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
            }
        }
        SmallVector<mlir::Value> funcOutputFromSlices;
        for (const auto output : slice.outputs) {
            funcOutputFromSlices.push_back(oldToNewMap[output]);
        }
        const auto returnLoc = appendLoc(netFunc.getLoc(), "_part{0}_return", targetIdx + 1);
        builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
    }
}

void VerticalFusionOutliner::buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                          ArrayRef<OutliningInstance> outlinedTargets) {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    OpBuilderLogger builderLog(getLogger().nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&netFunc.getBody().front(), &builderLog);
    DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;
    for (const auto& arg : netFunc.getArguments()) {
        oldToNewArgMap[arg] = arg;
    }

    for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
        const auto& slice = slices.front();
        const size_t sliceIdx = 0;

        SmallVector<mlir::Value> newInputs;
        for (const auto input : slice.inputs) {
            newInputs.push_back(oldToNewArgMap[input]);
        }

        const auto callLoc = appendLoc(netFunc.getLoc(), "_part{0}_call", targetIdx + 1);
        auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcName,
                                                          funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
        for (const auto& res : newCall.getResults()) {
            size_t idx = res.getResultNumber();
            oldToNewArgMap[slice.outputs[idx]] = res;
        }
    }
    netFunc.walk([&](mlir::func::ReturnOp ret) {
        for (auto i : irange(ret.getNumOperands())) {
            ret.setOperand(i, oldToNewArgMap[ret.getOperand(i)]);
        }
    });
}

}  // namespace outliner

namespace {

//
// VerticalFusionOutliningPass
//

class VerticalFusionOutliningPass final : public VPU::impl::VerticalFusionOutliningBase<VerticalFusionOutliningPass> {
public:
    VerticalFusionOutliningPass() = default;
    VerticalFusionOutliningPass(const VPU::TilingOptions& TilingOptions, Logger log);

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnModule() final;

private:
    // Initialize fields from pass options
    void initializeFromOptions();

private:
    size_t _numInstanceThreshold = 0;
    size_t _verticalFusionTileThreshold = 0;
};

VerticalFusionOutliningPass::VerticalFusionOutliningPass(const VPU::TilingOptions& TilingOptions, Logger log) {
    Base::initLogger(log, Base::getArgumentName());
    Base::copyOptionValuesFrom(TilingOptions);

    initializeFromOptions();
}

mlir::LogicalResult VerticalFusionOutliningPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    initializeFromOptions();

    return mlir::success();
}

void VerticalFusionOutliningPass::initializeFromOptions() {
    _verticalFusionTileThreshold = verticalFusionTileThreshold.getValue();
    _numInstanceThreshold = numInstanceThreshold.getValue();
}

//
// safeRunOnModule
//

void VerticalFusionOutliningPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

    // TODO E#150569: remove this condition when this pass is compatible with pre-outlined functions
    bool containsCallOps = false;
    netFunc.walk([&](mlir::Operation* op) {
        if (mlir::isa<mlir::func::CallOp>(op)) {
            containsCallOps = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    if (containsCallOps) {
        _log.info("The main function already contains call operations. Skipping pass");
        return;
    }

    outliner::VerticalFusionOutliner outliner(_numInstanceThreshold, _verticalFusionTileThreshold, _log);
    outliner.outline(moduleOp, "vf");
}

}  // namespace

//
// createVerticalFusionOutliningPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createVerticalFusionOutliningPass() {
    return std::make_unique<VerticalFusionOutliningPass>();
}

std::unique_ptr<mlir::Pass> vpux::VPU::createVerticalFusionOutliningPass(const TilingOptions& TilingOptions,
                                                                         Logger log) {
    return std::make_unique<VerticalFusionOutliningPass>(TilingOptions, log);
}
