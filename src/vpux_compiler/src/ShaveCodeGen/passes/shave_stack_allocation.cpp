//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/logger/logger.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <iterator>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_SHAVESTACKALLOCATION
#define GEN_PASS_DEF_SHAVESTACKALLOCATION
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

static constexpr unsigned MAX_ALLOC_BYTES = 64;

mlir::FailureOr<mlir::Value> materializeValueInInfoFunc(mlir::Value value, mlir::Block& sourceBlock,
                                                        mlir::OpBuilder& builder, mlir::IRMapping& mapper) {
    if (mapper.contains(value)) {
        return mapper.lookup(value);
    }

    if (mlir::isa<mlir::BlockArgument>(value)) {
        return mlir::failure();
    }

    auto* defOp = value.getDefiningOp();
    if (defOp == nullptr || defOp->getBlock() != &sourceBlock) {
        return mlir::failure();
    }

    for (auto operand : defOp->getOperands()) {
        if (mlir::failed(materializeValueInInfoFunc(operand, sourceBlock, builder, mapper))) {
            return mlir::failure();
        }
    }

    builder.clone(*defOp, mapper);
    if (!mapper.contains(value)) {
        return mlir::failure();
    }

    return mapper.lookup(value);
}

mlir::LogicalResult appendScratchInfoToTilingInfoFunc(mlir::func::FuncOp kernelFunc, mlir::func::FuncOp infoFunc,
                                                      int64_t numTilingAxes, mlir::MemRefType scratchMemRefType,
                                                      mlir::ValueRange scratchDynamicSizes) {
    auto& kernelBody = kernelFunc.getBody().front();
    auto& infoBody = infoFunc.getBody().front();
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(infoBody.getTerminator());
    mlir::OpBuilder builder(returnOp);
    mlir::IRMapping mapper;

    const int64_t numIterationSpaceArgs = numTilingAxes * 2;
    if (kernelFunc.getNumArguments() < static_cast<size_t>(numIterationSpaceArgs) ||
        infoFunc.getNumArguments() < static_cast<size_t>(numIterationSpaceArgs)) {
        return mlir::failure();
    }

    const auto kernelStart = kernelFunc.getNumArguments() - numIterationSpaceArgs;
    const auto infoStart = infoFunc.getNumArguments() - numIterationSpaceArgs;
    for (int64_t i = 0; i < numIterationSpaceArgs; ++i) {
        mapper.map(kernelFunc.getArgument(kernelStart + i), infoFunc.getArgument(infoStart + i));
    }

    auto zeroIdx = builder.create<mlir::arith::ConstantIndexOp>(returnOp.getLoc(), 0);

    // Map a scratch-buffer size value to its corresponding scratch-buffer offset. When the size
    // directly maps to an iteration-space size argument, the paired offset argument is used,
    // enabling the axis to be identified as multiclusterable (segmented).
    //
    // Scratch is uninitialized working memory, so the exact offset value does not affect
    // correctness. The only real constraint is that size + offset must stay within the bounds of
    // the original scratch buffer so the access is not out of bounds. Zero always satisfies this.
    // When the size is an iteration space size we can also map the offset to the corresponding
    // iteration space offset and maintain the required property. This allows the scratch buffer
    // to possibly (but not necessarily) be identified as being segmented.
    const auto getScratchOffsetFromScratchSize = [&](mlir::Value size) -> mlir::Value {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(size);
        if (blockArg == nullptr) {
            return zeroIdx;
        }
        const auto argNum = static_cast<int64_t>(blockArg.getArgNumber());
        if (blockArg.getOwner() == &infoFunc.getBody().front() && argNum >= infoStart &&
            argNum < infoStart + numTilingAxes) {
            return infoFunc.getArgument(argNum + numTilingAxes);
        }
        return zeroIdx;
    };

    SmallVector<mlir::Value> appendedResults;
    SmallVector<mlir::Value> appendedOffsets;
    size_t dynIdx = 0;
    for (auto dim : scratchMemRefType.getShape()) {
        if (dim == mlir::ShapedType::kDynamic) {
            if (dynIdx >= scratchDynamicSizes.size()) {
                return mlir::failure();
            }
            auto dynVal = scratchDynamicSizes[dynIdx++];
            auto mappedVal = materializeValueInInfoFunc(dynVal, kernelBody, builder, mapper);
            if (mlir::failed(mappedVal)) {
                return mlir::failure();
            }
            appendedResults.push_back(*mappedVal);
            appendedOffsets.push_back(getScratchOffsetFromScratchSize(*mappedVal));
        } else {
            appendedResults.push_back(builder.create<mlir::arith::ConstantIndexOp>(returnOp.getLoc(), dim));
            appendedOffsets.push_back(zeroIdx);
        }
    }

    appendedResults.append(appendedOffsets);
    returnOp.getOperandsMutable().append(appendedResults);
    auto newFuncType = mlir::FunctionType::get(infoFunc.getContext(), infoFunc.getArgumentTypes(),
                                               returnOp.getOperands().getTypes());
    infoFunc.setFunctionType(newFuncType);

    return mlir::success();
}

class ShaveStackAllocationPass final : public ShaveCodeGen::impl::ShaveStackAllocationBase<ShaveStackAllocationPass> {
public:
    using SwKernelUses = SmallVector<vpux::VPU::GenericSwLayerOp>;
    explicit ShaveStackAllocationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    mlir::LogicalResult addScratchBuffers(mlir::func::FuncOp func, SwKernelUses& uses,
                                          llvm::DenseMap<mlir::Operation*, size_t>& infoFuncUseCount);

    void safeRunOnModule() final {
        auto moduleOp = getOperation();
        auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);

        mlir::PassManager pm(&getContext());
        pm.addPass(mlir::bufferization::createBufferHoistingPass());
        pm.addPass(mlir::bufferization::createBufferLoopHoistingPass());

        mlir::bufferization::PromoteBuffersToStackPassOptions opts;
        opts.maxAllocSizeInBytes = MAX_ALLOC_BYTES;
        opts.maxRankOfAllocatedMemRef = 1;
        pm.addPass(mlir::bufferization::createPromoteBuffersToStackPass(opts));

        // Create a cache of uses of every FuncOp by SwKernelOps.
        llvm::DenseMap<mlir::func::FuncOp, SwKernelUses> funcUseMap;
        moduleOp.walk([&](vpux::VPU::GenericSwLayerOp swLayerOp) {
            auto kernelFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(swLayerOp.getCallee());
            funcUseMap[kernelFunc].push_back(swLayerOp);
        });

        // Count how many kernels reference each tiling info function up front, so
        // we can decide when a shared info func must be cloned.
        // Decremented when an info func is cloned, so the remaining sharers modify the
        // original in place instead of cloning again.
        llvm::DenseMap<mlir::Operation*, size_t> infoFuncUseCount;
        for (auto kernelFunc : swModule.getOps<mlir::func::FuncOp>()) {
            const auto kernelInfo = kernelFunc->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
            if (kernelInfo == nullptr) {
                continue;
            }
            if (auto infoFunc = swModule.lookupSymbol<mlir::func::FuncOp>(kernelInfo.getTilingInfoFunc())) {
                ++infoFuncUseCount[infoFunc.getOperation()];
            }
        }

        auto funcOps = vpux::to_small_vector(swModule.getOps<mlir::func::FuncOp>());

        for (auto func : funcOps) {
            if (mlir::failed(pm.run(func))) {
                _log.trace("Failed to promote buffers to stack");
                return signalPassFailure();
            }

            if (mlir::failed(addScratchBuffers(func, funcUseMap[func], infoFuncUseCount))) {
                _log.trace("Failed to add scratch buffers");
                return signalPassFailure();
            }
        }
    }
};

// Clones the tiling info function under a unique name and repoints the kernel's KernelInfo to
// the clone. Used when the original info function is shared between several kernels, so that
// appending scratch buffer info for one kernel does not corrupt the others.
mlir::func::FuncOp cloneInfoFuncForKernel(mlir::ModuleOp swModule, mlir::func::FuncOp kernelFunc,
                                          VPU::KernelInfoAttr kernelInfo, mlir::func::FuncOp infoFunc) {
    auto clonedInfoFunc = mlir::cast<mlir::func::FuncOp>(infoFunc->clone());
    mlir::SymbolTable symbolTable(swModule);
    symbolTable.insert(clonedInfoFunc);

    const auto newTilingInfoFunc = mlir::FlatSymbolRefAttr::get(clonedInfoFunc.getSymNameAttr());
    const auto newKernelInfo = VPU::KernelInfoAttr::get(kernelFunc.getContext(), newTilingInfoFunc,
                                                        kernelInfo.getTilingAxes(), kernelInfo.getNumSlicedInputs());
    kernelFunc->setAttr(VPU::KernelInfoAttr::kFuncAttrName, newKernelInfo);
    return clonedInfoFunc;
}

mlir::LogicalResult ShaveStackAllocationPass::addScratchBuffers(
        mlir::func::FuncOp func, SwKernelUses& uses, llvm::DenseMap<mlir::Operation*, size_t>& infoFuncUseCount) {
    auto allocOps = vpux::to_small_vector(func.getOps<mlir::memref::AllocOp>());
    if (allocOps.empty() || uses.empty()) {
        return mlir::success();
    }

    auto& ctx = getContext();
    mlir::OpBuilder builder(&ctx);

    auto kernelInfo = func->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);

    // Collect memref.alloc ops and replace them with kernel arguments.
    auto inputTys = vpux::to_small_vector(func.getArgumentTypes());
    auto outputTys = vpux::to_small_vector(func.getResultTypes());
    auto& funcBlock = func.getFunctionBody();

    const size_t argInsertionPoint = llvm::count_if(uses.front().getInputs(), [](mlir::Value v) {
        return mlir::isa<mlir::RankedTensorType>(v.getType());
    });

    // Resolve the tiling info function from the KernelInfo when present.
    mlir::func::FuncOp infoFunc = nullptr;
    int64_t numTilingAxes = 0;
    if (kernelInfo) {
        auto swModule = func->getParentOfType<mlir::ModuleOp>();
        infoFunc = swModule.lookupSymbol<mlir::func::FuncOp>(kernelInfo.getTilingInfoFunc());
        if (!infoFunc) {
            mlir::emitError(func.getLoc())
                    << "Failed to find tiling info function for kernel '" << func.getSymName() << "'";
            return mlir::failure();
        }
        // If the info function is shared with other kernels, clone it so that appending
        // scratch buffer info for this kernel does not corrupt the others. Decrement the
        // original's use count so the remaining sharers can modify it in place.
        auto* const infoFuncOp = infoFunc.getOperation();
        if (infoFuncUseCount.lookup(infoFuncOp) > 1) {
            infoFunc = cloneInfoFuncForKernel(swModule, func, kernelInfo, infoFunc);
            --infoFuncUseCount[infoFuncOp];
            // The local kernelInfo is no longer valid after the repoint, so re-read it.
            kernelInfo = func->getAttrOfType<VPU::KernelInfoAttr>(VPU::KernelInfoAttr::kFuncAttrName);
        }
        numTilingAxes = static_cast<int64_t>(kernelInfo.getTilingAxes().size());
    }

    SmallVector<mlir::Type> scratchMemRefTypes;
    bool hasInvalidAllocOps = false;
    size_t currentInsertPos = argInsertionPoint;
    for (auto allocOp : allocOps) {
        auto allocTy = allocOp.getType();
        if (!allocTy.areTrailingDimsContiguous(allocTy.getRank())) {
            // Non-contiguous allocations are not supported.
            mlir::emitError(allocOp.getLoc()) << "Unexpected non-contiguous allocation in shave kernel.";
            hasInvalidAllocOps = true;
            continue;
        }
        if (!allocTy.hasStaticShape()) {
            if (!kernelInfo) {
                // TODO: E#192657 convert dynamic allocations to scratch buffers
                mlir::emitError(allocOp.getLoc())
                        << "Unexpected dynamic allocation in shave kernel without KernelInfo.";
                hasInvalidAllocOps = true;
            }
        }

        if (kernelInfo) {
            if (mlir::failed(appendScratchInfoToTilingInfoFunc(func, infoFunc, numTilingAxes, allocTy,
                                                               allocOp.getDynamicSizes()))) {
                mlir::emitError(allocOp.getLoc()) << "Failed to append scratch sizes to tiling info function";
                return mlir::failure();
            }
        }

        auto arg = funcBlock.insertArgument(currentInsertPos, allocOp.getType(), allocOp.getLoc());
        allocOp->getResult(0).replaceAllUsesWith(arg);
        allocOp->erase();
        inputTys.insert(inputTys.begin() + currentInsertPos, arg.getType());
        scratchMemRefTypes.push_back(allocTy);
        currentInsertPos++;
    }

    // The calling convention passes each scratch buffer twice: once as an input arg (the
    // alloc-replacement arg above, used by the function body) and once as an output arg
    // (matching the scratch_inputs operand of GenericSwLayerOp). Insert the output-side args
    // directly after the last MemRefType arg, before any scalar args.
    const auto lastMemrefRevIt = std::find_if(inputTys.rbegin(), inputTys.rend(), [](mlir::Type t) {
        return mlir::isa<mlir::MemRefType>(t);
    });
    const size_t afterLastMemref =
            lastMemrefRevIt == inputTys.rend()
                    ? 0
                    : static_cast<size_t>(std::distance(inputTys.begin(), lastMemrefRevIt.base()));
    size_t extraArgPos = afterLastMemref;
    for (const auto scratchTy : scratchMemRefTypes) {
        funcBlock.insertArgument(extraArgPos, scratchTy, func.getLoc());
        inputTys.insert(inputTys.begin() + extraArgPos, scratchTy);
        ++extraArgPos;
    }

    auto funcType = mlir::FunctionType::get(&ctx, inputTys, outputTys);
    func.setFunctionType(funcType);

    // Add the new scratch buffers to every call of the kernel.
    for (auto swLayerOp : uses) {
        builder.setInsertionPoint(swLayerOp);

        // Create empty auxiliary buffers for the new scratch allocations.
        SmallVector<mlir::Value> newScratchValues;
        newScratchValues.reserve(scratchMemRefTypes.size());
        for (auto memrefTy : scratchMemRefTypes) {
            auto tensorTy = mlir::memref::getTensorTypeFromMemRefType(memrefTy);
            newScratchValues.push_back(VPU::createEmptyAuxiliaryBuffer(builder, swLayerOp->getLoc(), tensorTy));
        }

        // Append new scratch operands in-place; MutableOperandRange updates operandSegmentSizes automatically.
        swLayerOp.getScratchInputsMutable().append(newScratchValues);

        if (swLayerOp.getTilingProperties().has_value()) {
            // The tensor types could be dynamic in this case. We use backInferTileInfo
            // to infer the actual type.
            const auto resultNdType = mlir::cast<vpux::NDTypeInterface>(swLayerOp.getResult(0).getType());
            const TileInfo outputTile(resultNdType.getShape());
            const auto inputTiling = swLayerOp.backInferTileInfo(outputTile, _log);
            const auto numInputs = llvm::count_if(swLayerOp.getInputs(), [](mlir::Value v) {
                return mlir::isa<mlir::RankedTensorType>(v.getType());
            });
            const auto scratchInputs = swLayerOp.getScratchInputs();
            VPUX_THROW_UNLESS(inputTiling.tiles.size() == numInputs + scratchInputs.size(),
                              "Expected {0} tiles from backInferTileInfo, got {1}", numInputs + scratchInputs.size(),
                              inputTiling.tiles.size());

            // Only update the newly appended scratch inputs; pre-existing ones already have
            // correct types.
            const auto firstNewScratch = scratchInputs.size() - newScratchValues.size();
            for (const auto scratchIdx : llvm::seq<size_t>(firstNewScratch, scratchInputs.size())) {
                const auto& scratchTile = inputTiling.tiles[numInputs + scratchIdx];
                for (auto dim : scratchTile.shape.raw()) {
                    VPUX_THROW_UNLESS(dim != mlir::ShapedType::kDynamic,
                                      "Dynamic scratch buffer shapes are not yet supported (E#192657)");
                }
                const auto newScratchIdx = scratchIdx - firstNewScratch;
                const auto memrefTy = mlir::cast<mlir::MemRefType>(scratchMemRefTypes[newScratchIdx]);
                const auto scratchTensorTy =
                        mlir::RankedTensorType::get(scratchTile.shape.raw(), memrefTy.getElementType());
                scratchInputs[scratchIdx].setType(scratchTensorTy);
            }
        }
    }

    if (hasInvalidAllocOps) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createShaveStackAllocationPass(Logger log) {
    return std::make_unique<ShaveStackAllocationPass>(log);
}
