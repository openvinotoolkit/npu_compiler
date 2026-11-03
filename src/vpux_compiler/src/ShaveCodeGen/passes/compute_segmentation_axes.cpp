//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/SmallPtrSet.h>

#include <optional>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_COMPUTESEGMENTATIONAXES
#define GEN_PASS_DEF_COMPUTESEGMENTATIONAXES
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

// Sizes and offsets returned by the tilingInfoFunc for one sliced input.
struct SliceReturnInfo {
    SmallVector<mlir::Value> sizes;
    SmallVector<mlir::Value> offsets;
};

// Returns the set of all values transitively reachable forward through uses from `arg`.
static llvm::SmallPtrSet<mlir::Value, 16> collectReachable(mlir::BlockArgument arg) {
    llvm::SmallPtrSet<mlir::Value, 16> reachable;
    SmallVector<mlir::Value> worklist = {arg};
    while (!worklist.empty()) {
        auto val = worklist.pop_back_val();
        if (!reachable.insert(val).second) {
            continue;
        }
        for (auto* user : val.getUsers()) {
            for (auto result : user->getResults()) {
                worklist.push_back(result);
            }
        }
    }
    return reachable;
}

// Builds SliceReturnInfo entries for the sliced inputs and aux buffers
// by mapping tilingInfoFunc return values to per-input sizes/offsets.
// The returned vector has length numSlicedInputs + numAuxBuffers
static SmallVector<SliceReturnInfo> collectTilingInfoFuncReturnInfos(mlir::func::FuncOp kernelFunc,
                                                                     mlir::func::FuncOp tilingInfoFunc,
                                                                     int64_t numSlicedInputs, int64_t numTensorInputs,
                                                                     int64_t numAuxBuffers) {
    auto& entryBlock = tilingInfoFunc.getBody().front();
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(entryBlock.getTerminator());

    SmallVector<SliceReturnInfo> infos(numSlicedInputs + numAuxBuffers);

    int64_t retIdx = 0;
    auto fillInfo = [&](SliceReturnInfo& info, int64_t kernelArgIdx) {
        const int64_t rank =
                mlir::cast<mlir::ShapedType>(kernelFunc.getFunctionType().getInput(kernelArgIdx)).getRank();

        for (int64_t dim = 0; dim < rank; ++dim) {
            info.sizes.push_back(returnOp.getOperand(retIdx + dim));
            info.offsets.push_back(returnOp.getOperand(retIdx + rank + dim));
        }
        retIdx += 2 * rank;
    };

    for (int64_t argIdx = 0; argIdx < numTensorInputs + numAuxBuffers; ++argIdx) {
        if (argIdx >= numSlicedInputs && argIdx < numTensorInputs) {
            // Skip non-sliced inputs
            continue;
        }
        const int64_t infoIdx = (argIdx < numSlicedInputs) ? argIdx : numSlicedInputs + (argIdx - numTensorInputs);
        fillInfo(infos[infoIdx], argIdx);
    }
    return infos;
}

// Classifies how a single sliced input relates to one tiling axis (segmented or broadcast)
// or std::nullopt if the relationship cannot be classified.
static std::optional<int64_t> classifyInputForAxis(const SliceReturnInfo& info, mlir::BlockArgument sizeArg,
                                                   mlir::BlockArgument offsetArg,
                                                   const llvm::SmallPtrSet<mlir::Value, 16>& reachableFromSize,
                                                   const llvm::SmallPtrSet<mlir::Value, 16>& reachableFromOffset) {
    const int64_t rank = static_cast<int64_t>(info.sizes.size());
    int64_t segmentationDim = VPU::KernelInfoAttr::kBroadcastDim;

    for (int64_t dim = 0; dim < rank; ++dim) {
        if (info.sizes[dim] == sizeArg && info.offsets[dim] == offsetArg) {
            if (segmentationDim != VPU::KernelInfoAttr::kBroadcastDim) {
                return std::nullopt;
            }
            segmentationDim = dim;
            continue;
        }
        if (reachableFromSize.count(info.sizes[dim]) || reachableFromOffset.count(info.sizes[dim]) ||
            reachableFromSize.count(info.offsets[dim]) || reachableFromOffset.count(info.offsets[dim])) {
            return std::nullopt;
        }
    }
    return segmentationDim;
}

//
// ComputeSegmentationAxesPass
//

class ComputeSegmentationAxesPass final :
        public vpux::ShaveCodeGen::impl::ComputeSegmentationAxesBase<ComputeSegmentationAxesPass> {
public:
    explicit ComputeSegmentationAxesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void processKernelFunc(mlir::ModuleOp swModule, mlir::func::FuncOp funcOp, VPU::KernelInfoAttr kernelInfo);
};

void ComputeSegmentationAxesPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

    for (auto funcOp : swModule.getOps<mlir::func::FuncOp>()) {
        const auto kernelInfo = funcOp->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
        if (!kernelInfo) {
            continue;
        }

        processKernelFunc(swModule, funcOp, kernelInfo);
    }
}

void ComputeSegmentationAxesPass::processKernelFunc(mlir::ModuleOp swModule, mlir::func::FuncOp funcOp,
                                                    VPU::KernelInfoAttr kernelInfo) {
    auto* ctx = funcOp.getContext();

    const auto tilingAxes = kernelInfo.getTilingAxes();
    const int64_t numAxes = static_cast<int64_t>(tilingAxes.size());
    const int64_t numSlicedInputs = kernelInfo.getNumSlicedInputs().getInt();
    const int64_t numTensorInputs = kernelInfo.getNumTensorInputs().getInt();
    const int64_t numAuxBuffers = kernelInfo.getNumAuxBuffers().getInt();

    auto tilingInfoFunc = swModule.lookupSymbol<mlir::func::FuncOp>(kernelInfo.getTilingInfoFunc());
    if (!tilingInfoFunc) {
        _log.warning("Cannot find tilingInfoFunc for kernel '{0}'", funcOp.getName());
        return;
    }
    if (tilingInfoFunc.isExternal()) {
        _log.warning("tilingInfoFunc '{0}' for kernel '{1}' has no body; skipping classification",
                     kernelInfo.getTilingInfoFunc(), funcOp.getName());
        return;
    }

    const auto inputInfos =
            collectTilingInfoFuncReturnInfos(funcOp, tilingInfoFunc, numSlicedInputs, numTensorInputs, numAuxBuffers);

    // The last 2*numAxes args of tilingInfoFunc are the iteration space values.
    const int64_t iterationSpaceStartPos = tilingInfoFunc.getNumArguments() - 2 * numAxes;

    SmallVector<mlir::Attribute> inputSegmentationAxes;

    for (int64_t axis = 0; axis < numAxes; ++axis) {
        auto sizeArg = tilingInfoFunc.getArgument(iterationSpaceStartPos + axis);
        auto offsetArg = tilingInfoFunc.getArgument(iterationSpaceStartPos + numAxes + axis);
        const auto reachableFromSize = collectReachable(sizeArg);
        const auto reachableFromOffset = collectReachable(offsetArg);

        SmallVector<int64_t> splitDims;

        // Iterate over all inputs:
        //   [sliced inputs] [non-sliced tensor inputs: broadcast] [aux buffers]
        for (int64_t argIdx = 0; argIdx < numTensorInputs + numAuxBuffers; ++argIdx) {
            if (argIdx >= numSlicedInputs && argIdx < numTensorInputs) {
                // Non-sliced tensor input: always broadcast.
                splitDims.push_back(VPU::KernelInfoAttr::kBroadcastDim);
                continue;
            }

            const int64_t infoIdx = (argIdx < numSlicedInputs) ? argIdx : numSlicedInputs + (argIdx - numTensorInputs);
            const auto splitDim = classifyInputForAxis(inputInfos[infoIdx], sizeArg, offsetArg, reachableFromSize,
                                                       reachableFromOffset);
            splitDims.push_back(splitDim.value_or(VPU::KernelInfoAttr::kUnknownDim));
        }

        inputSegmentationAxes.push_back(mlir::DenseI64ArrayAttr::get(ctx, splitDims));
    }

    auto updatedKernelInfo =
            VPU::KernelInfoAttr::get(ctx, kernelInfo.getTilingInfoFunc(), kernelInfo.getTilingAxes(),
                                     kernelInfo.getNumSlicedInputs(), kernelInfo.getNumTensorInputs(),
                                     kernelInfo.getNumAuxBuffers(), mlir::ArrayAttr::get(ctx, inputSegmentationAxes));
    funcOp->setAttr(VPU::KernelInfoAttr::kFuncAttrName, updatedKernelInfo);
}

}  // namespace

//
// createComputeSegmentationAxesPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createComputeSegmentationAxesPass(Logger log) {
    return std::make_unique<ComputeSegmentationAxesPass>(log);
}
