//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/RegionUtils.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_OUTLINECODEGENCAPSULES
#define GEN_PASS_DEF_OUTLINECODEGENCAPSULES
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

// Returns the indices of dimensions of the first result along which the scf::ForOp
// can be tiled, or failure if the loop does not conform to the expected tiled form.
static mlir::FailureOr<SmallVector<int64_t>> getTilablePositions(mlir::scf::ForOp forOp) {
    // Note that this pre-checks some assumptions about the tiled form as well:
    // - the top level scf.for op has a scf.yield for which all operands are insert_slice ops,
    //   and the first one takes values produced by OutSliceInfo.
    // - the presence of the OutSliceInfo op constrains the form of the loop as it
    //   provides slices of the iteration space and ties it to the 0th result of the
    //   loop (the tiled dimensions actually constitute our full iteration space)
    auto yieldOp = mlir::cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());

    if (!llvm::all_of(yieldOp.getOperands(), [](mlir::Value operand) {
            return mlir::isa_and_nonnull<mlir::tensor::InsertSliceOp>(operand.getDefiningOp());
        })) {
        return mlir::failure();
    }

    auto insertSliceOp = mlir::cast<mlir::tensor::InsertSliceOp>(yieldOp.getOperand(0).getDefiningOp());

    SmallVector<int64_t> tilablePositions;
    for (auto [idx, size] : llvm::enumerate(insertSliceOp.getMixedSizes())) {
        auto sizeValue = mlir::dyn_cast<mlir::Value>(size);
        if (sizeValue && mlir::isa<Shave::OutSliceInfoOp>(sizeValue.getDefiningOp())) {
            tilablePositions.push_back(static_cast<int64_t>(idx));
        }
    }

    // This path only supports tiled loops where OutSliceInfo marks at least one
    // tilable dimension in insert_slice sizes.
    if (tilablePositions.empty()) {
        return mlir::failure();
    }

    return tilablePositions;
}

static mlir::FailureOr<mlir::scf::ForOp> getTiledOp(IE::CodeGenCapsuleOp capsuleOp) {
    auto capsuleBlock = capsuleOp.getBody();
    auto capsuleTerminator = mlir::cast<IE::CGCYieldOp>(capsuleBlock->getTerminator());
    auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(capsuleTerminator->getOperand(0).getDefiningOp());
    if (!forOp) {
        return mlir::failure();
    }

    auto upperBound = forOp.getUpperBound();
    if (!mlir::isa_and_nonnull<Shave::LoopTripCountOp>(upperBound.getDefiningOp())) {
        return mlir::failure();
    }

    // Verify that the forOp is the only tensor-producing op in the capsule block,
    // besides tensor.empty ops used as forOp iter_args and tensor arith.constants
    // that are only used from within the loop body and will be sunk there later.
    for (auto& op : *capsuleBlock) {
        if (&op == forOp.getOperation() || &op == capsuleTerminator.getOperation()) {
            continue;
        }
        const auto producesTensor = llvm::any_of(op.getResultTypes(), [](mlir::Type ty) {
            return mlir::isa<mlir::TensorType>(ty);
        });
        if (!producesTensor) {
            continue;
        }

        if (mlir::isa<mlir::tensor::EmptyOp>(op) && llvm::all_of(op.getUsers(), [&](mlir::Operation* user) {
                return user == forOp.getOperation();
            })) {
            continue;
        }

        if (mlir::isa<mlir::arith::ConstantOp>(op) && llvm::all_of(op.getUsers(), [&](mlir::Operation* user) {
                return forOp.getOperation()->isAncestor(user);
            })) {
            continue;
        }

        return mlir::failure();
    }

    return forOp;
}

static mlir::func::FuncOp outlineSwLayer(mlir::MLIRContext* ctx, mlir::ModuleOp module, IE::CodeGenCapsuleOp capsuleOp,
                                         size_t counter) {
    auto capsuleBlock = capsuleOp.getBody();
    auto dpsInputs = vpux::to_small_vector(capsuleBlock->getArgumentTypes());

    auto capsuleTerminator = mlir::cast<IE::CGCYieldOp>(capsuleBlock->getTerminator());
    auto outputTypes = vpux::to_small_vector(capsuleTerminator->getOperandTypes());

    auto builder = mlir::OpBuilder::atBlockBegin(module.getBody());

    builder.setInsertionPointToEnd(capsuleBlock);
    auto loc = capsuleTerminator->getLoc();
    builder.create<mlir::func::ReturnOp>(loc, capsuleTerminator->getOperands());
    capsuleTerminator->erase();

    // Actually create the function
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = mlir::FunctionType::get(ctx, dpsInputs, outputTypes);
    auto funcName = printToString("generated_{0}", counter);
    auto funcOp = builder.create<mlir::func::FuncOp>(loc, funcName, funcType);
    auto funcOpBody = funcOp.addEntryBlock();
    capsuleBlock->dropAllUses();
    capsuleBlock->moveBefore(funcOpBody);
    funcOpBody->erase();

    return funcOp;
}

static VPU::GenericSwLayerOp createSwLayerOp(mlir::OpBuilder& builder, IE::CodeGenCapsuleOp op,
                                             mlir::SymbolRefAttr fullSymRef) {
    return builder.create<VPU::GenericSwLayerOp>(op->getLoc(), op->getResultTypes(), fullSymRef, op->getOperands(),
                                                 VPU::JITTilingAttr{}, mlir::ValueRange{}, mlir::ValueRange{});
}

//
// OutlineCodeGenCapsulesPass
//

class OutlineCodeGenCapsulesPass final :
        public ShaveCodeGen::impl::OutlineCodeGenCapsulesBase<OutlineCodeGenCapsulesPass> {
public:
    explicit OutlineCodeGenCapsulesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnModule() final;
};

// Rewrites a non-tilable CodeGenCapsule by outlining its body into a generated
// SW function and replacing the capsule with a VPU::GenericSwLayerOp call.
struct OutlineCodeGenCapsule : mlir::OpRewritePattern<IE::CodeGenCapsuleOp> {
    using OpRewritePattern::OpRewritePattern;

    explicit OutlineCodeGenCapsule(mlir::MLIRContext* ctx, const mlir::ModuleOp swModule, size_t& counter,
                                   mlir::StringAttr swModuleRoot)
            : OpRewritePattern<IE::CodeGenCapsuleOp>(ctx),
              _swModule(swModule),
              _counter(counter),
              _swModuleRoot(swModuleRoot) {
    }

    mlir::LogicalResult matchAndRewrite(IE::CodeGenCapsuleOp op, mlir::PatternRewriter& rewriter) const override {
        auto& ctx = *rewriter.getContext();

        auto* capsuleBlock = op.getBody();
        auto hasDisallowedOp = capsuleBlock->walk([](mlir::Operation* innerOp) {
            if (mlir::isa<Shave::LoopTripCountOp, Shave::OutSliceInfoOp>(innerOp)) {
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
        });
        if (hasDisallowedOp.wasInterrupted()) {
            return mlir::failure();
        }

        auto outlinedFunc = outlineSwLayer(&ctx, _swModule, op, _counter);
        rewriter.setInsertionPointAfter(op);
        auto fullSymRef = mlir::SymbolRefAttr::get(
                _swModuleRoot, llvm::ArrayRef<mlir::FlatSymbolRefAttr>(mlir::FlatSymbolRefAttr::get(outlinedFunc)));

        auto genericSwLayerOp = createSwLayerOp(rewriter, op, fullSymRef);
        rewriter.replaceOp(op, genericSwLayerOp->getResults());

        _counter++;
        return mlir::success();
    }

private:
    mlir::ModuleOp _swModule;
    size_t& _counter;
    mlir::StringAttr _swModuleRoot;
};

// Helper struct to track the calling convention information for the kernel
// and slice info functions and map values from the original loop to the various
// arguments.
// Note that the kernel argument mapping is:
//   <slices..., tensor operands..., scalar operands..., iteration space values...>
//  -> returnedValues
// The slice info mapping is:
//   <scalar operands..., iteration space values...> -> [sizes..., offsets...] for each slice in order
struct KernelCCInfo {
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> slices;
    llvm::SmallSetVector<mlir::Value, 4> tensorOps;
    llvm::SmallSetVector<mlir::Value, 4> scalarOps;
    llvm::SmallVector<mlir::Value, 4> iterationSpaceValues;
    llvm::SmallVector<mlir::Value> returnValues;

    llvm::SmallVector<mlir::Value> getAllInputValues(bool includeTensors) const {
        llvm::SmallVector<mlir::Value> result;
        if (includeTensors) {
            for (auto slice : slices) {
                result.push_back(slice.getResult());
            }
            result.append(tensorOps.begin(), tensorOps.end());
        }
        result.append(scalarOps.begin(), scalarOps.end());
        result.append(iterationSpaceValues.begin(), iterationSpaceValues.end());
        return result;
    }
};

// Outlines a function that computes slice sizes/offsets from the iteration space
// and returns them as index values in [sizes..., offsets...] order per slice.
static mlir::func::FuncOp outlineSliceInfoFunc(KernelCCInfo& ccInfo, mlir::Block* sourceBlock, mlir::ModuleOp module,
                                               mlir::Location loc, size_t counter) {
    auto builder = mlir::OpBuilder(module.getBodyRegion());
    auto inputValues = ccInfo.getAllInputValues(false);
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Type> returnTypes;
    for (auto val : inputValues) {
        inputTypes.push_back(val.getType());
    }
    for (auto slice : ccInfo.slices) {
        auto rank = mlir::cast<mlir::RankedTensorType>(slice.getType()).getRank();
        for (int64_t i = 0; i < rank; ++i) {
            returnTypes.push_back(builder.getIndexType());
            returnTypes.push_back(builder.getIndexType());
        }
    }

    const auto funcType = mlir::FunctionType::get(module.getContext(), inputTypes, returnTypes);
    auto funcName = printToString("generated_info_{0}", counter);
    auto newFunc = builder.create<mlir::func::FuncOp>(loc, funcName, funcType);

    auto funcBuilder = mlir::OpBuilder::atBlockEnd(newFunc.addEntryBlock());
    mlir::IRMapping mapper;
    for (size_t i = 0; i < inputValues.size(); ++i) {
        mapper.map(inputValues[i], newFunc.getArgument(i));
    }

    // Collect values that the function must return, materializing constants for static sizes/offsets.
    SmallVector<mlir::Value> returnValues;
    for (auto slice : ccInfo.slices) {
        for (auto size : slice.getMixedSizes()) {
            returnValues.push_back(mlir::getValueOrCreateConstantIndexOp(funcBuilder, slice.getLoc(), size));
        }
        for (auto offset : slice.getMixedOffsets()) {
            returnValues.push_back(mlir::getValueOrCreateConstantIndexOp(funcBuilder, slice.getLoc(), offset));
        }
    }

    // Walk up def-use chains from return values to find all ops that need cloning.
    llvm::SmallSetVector<mlir::Operation*, 16> opsToClone;
    SmallVector<mlir::Operation*> worklist;
    for (auto val : returnValues) {
        if (mapper.contains(val) || mlir::isa<mlir::BlockArgument>(val)) {
            continue;
        }
        auto* defOp = val.getDefiningOp();
        if (defOp && opsToClone.insert(defOp)) {
            worklist.push_back(defOp);
        }
    }
    while (!worklist.empty()) {
        auto* op = worklist.pop_back_val();
        for (auto operand : op->getOperands()) {
            if (mapper.contains(operand) || mlir::isa<mlir::BlockArgument>(operand)) {
                continue;
            }
            auto* defOp = operand.getDefiningOp();
            if (defOp && opsToClone.insert(defOp)) {
                worklist.push_back(defOp);
            }
        }
    }

    // Walk pre-order to ensure that each operand is mapped before its uses are cloned.
    sourceBlock->walk([&](mlir::Operation* op) {
        if (opsToClone.contains(op)) {
            funcBuilder.clone(*op, mapper);
        }
    });

    SmallVector<mlir::Value> returnedVals;
    for (auto val : returnValues) {
        returnedVals.push_back(mapper.contains(val) ? mapper.lookup(val) : val);
    }

    funcBuilder.create<mlir::func::ReturnOp>(loc, returnedVals);
    return newFunc;
}

// Outlines the loop body into a kernel function by cloning the executable
// body ops and returning the produced slice values.
static mlir::func::FuncOp outlineTiledBody(KernelCCInfo& ccInfo, mlir::Block* body, mlir::ModuleOp module,
                                           mlir::Location loc, size_t counter) {
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Value> inputValues = ccInfo.getAllInputValues(true);
    SmallVector<mlir::Type> returnTypes;

    // Construct the function type.
    for (auto val : inputValues) {
        inputTypes.push_back(val.getType());
    }
    for (auto val : ccInfo.returnValues) {
        returnTypes.push_back(val.getType());
    }
    const auto funcType = mlir::FunctionType::get(module.getContext(), inputTypes, returnTypes);
    auto builder = mlir::OpBuilder(module.getBodyRegion());
    auto funcName = printToString("generated_{0}", counter);
    auto newFunc = builder.create<mlir::func::FuncOp>(loc, funcName, funcType);
    auto funcBuilder = mlir::OpBuilder::atBlockEnd(newFunc.addEntryBlock());

    mlir::IRMapping mapper;
    for (size_t i = 0; i < inputValues.size(); ++i) {
        mapper.map(inputValues[i], newFunc.getArgument(i));
    }

    auto yield = mlir::cast<mlir::scf::YieldOp>(body->getTerminator());
    for (auto& op : *body) {
        if (mlir::isa<Shave::OutSliceInfoOp>(op) || &op == yield.getOperation()) {
            continue;
        }
        if (auto insertSlice = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op)) {
            if (llvm::any_of(yield->getOperands(), [&](mlir::Value operand) {
                    return operand.getDefiningOp() == insertSlice;
                })) {
                continue;
            }
        }
        funcBuilder.clone(op, mapper);
    }

    // Replace the extract slices as the mapping won't do this for us
    mlir::IRRewriter funcRewriter(newFunc);
    for (auto [idx, slice] : llvm::enumerate(ccInfo.slices)) {
        // We need the 'if' below because of differences caused by how we map slices
        // for the region case and non-region case (in the non-region case we
        // don't map anything so no cloning happens). Ideally this should be
        // handled uniformly (with the non-region case being the correct
        // behaviour.
        if (mapper.lookup(slice->getResult(0)).getDefiningOp()) {
            funcRewriter.replaceOp(mapper.lookup(slice->getResult(0)).getDefiningOp(),
                                   mlir::ValueRange{newFunc.getArgument(idx)});
        }
    }

    SmallVector<mlir::Value> returnedValues;
    for (auto operand : ccInfo.returnValues) {
        returnedValues.push_back(mapper.lookup(operand));
    }
    funcBuilder.create<mlir::func::ReturnOp>(yield->getLoc(), returnedValues);
    return newFunc;
}

// Inlines the slice-info function body at the current insertion point using
// provided call values and returns the mapped return operands.
static SmallVector<mlir::Value> inlineInputSliceInfoCall(mlir::func::FuncOp infoFunc,
                                                         llvm::SmallVector<mlir::Value>& infoCallValues,
                                                         mlir::PatternRewriter& rewriter) {
    mlir::IRMapping inlineMapper;
    auto& infoFuncBody = infoFunc.getBody().front();
    for (auto [arg, val] : llvm::zip(infoFuncBody.getArguments(), infoCallValues)) {
        inlineMapper.map(arg, val);
    }
    auto infoReturn = mlir::cast<mlir::func::ReturnOp>(infoFuncBody.getTerminator());
    for (auto& op : infoFuncBody) {
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            continue;
        }
        auto* clonedOp = rewriter.clone(op, inlineMapper);
        SmallVector<mlir::OpFoldResult> foldResults;
        if (succeeded(clonedOp->fold(foldResults)) && !foldResults.empty()) {
            for (auto [origResult, foldResult] : llvm::zip(op.getResults(), foldResults)) {
                auto replacement = mlir::dyn_cast<mlir::Value>(foldResult);
                if (!replacement) {
                    replacement = rewriter.create<mlir::arith::ConstantOp>(
                            clonedOp->getLoc(), mlir::cast<mlir::TypedAttr>(mlir::cast<mlir::Attribute>(foldResult)));
                }
                inlineMapper.map(origResult, replacement);
            }
            rewriter.eraseOp(clonedOp);
        }
    }
    SmallVector<mlir::Value> inlinedResults;
    for (auto retVal : infoReturn.getOperands()) {
        inlinedResults.push_back(inlineMapper.lookup(retVal));
    }
    return inlinedResults;
}

// Create a slice op from inlined slice info results. Returns failure if the slice could not be constructed.
static mlir::FailureOr<mlir::Value> createSliceFromInlinedInfo(mlir::tensor::ExtractSliceOp slice,
                                                               ArrayRef<mlir::Value> inlinedResults,
                                                               int64_t callValuePos, mlir::PatternRewriter& rewriter) {
    // IE::SliceOp does not support non-unit strides.
    if (!llvm::all_of(slice.getMixedStrides(), [](mlir::OpFoldResult stride) {
            auto val = mlir::getConstantIntValue(stride);
            return val.has_value() && *val == 1;
        })) {
        return mlir::failure();
    }

    auto sliceRank = mlir::cast<mlir::RankedTensorType>(slice.getType()).getRank();
    SmallVector<mlir::OpFoldResult> newSizes;
    SmallVector<mlir::OpFoldResult> newOffsets;
    for (int64_t i = 0; i < sliceRank; ++i) {
        newSizes.push_back(mlir::getAsOpFoldResult(inlinedResults[callValuePos + i]));
        newOffsets.push_back(mlir::getAsOpFoldResult(inlinedResults[callValuePos + sliceRank + i]));
    }

    // Create and fold a tensor::ExtractSliceOp with the sizes/offsets. This may
    // fold, in which case no slice was needed.
    SmallVector<mlir::OpFoldResult> newStrides(newSizes.size(), rewriter.getIndexAttr(1));
    auto newSlice = rewriter.createOrFold<mlir::tensor::ExtractSliceOp>(slice.getLoc(), slice.getSource(), newOffsets,
                                                                        newSizes, newStrides);

    if (!newSlice.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
        return newSlice;
    }

    // Otherwise convert to IE::SliceOp which requires static offsets/sizes.
    SmallVector<int64_t> staticOffsets;
    SmallVector<int64_t> staticSizes;
    for (auto ofr : newOffsets) {
        auto val = mlir::getConstantIntValue(ofr);
        if (!val) {
            rewriter.eraseOp(newSlice.getDefiningOp());
            return mlir::failure();
        }
        staticOffsets.push_back(*val);
    }
    for (auto ofr : newSizes) {
        auto val = mlir::getConstantIntValue(ofr);
        if (!val) {
            rewriter.eraseOp(newSlice.getDefiningOp());
            return mlir::failure();
        }
        staticSizes.push_back(*val);
    }
    rewriter.eraseOp(newSlice.getDefiningOp());
    auto* ctx = rewriter.getContext();
    return rewriter
            .create<IE::SliceOp>(slice.getLoc(), slice.getSource(), getIntArrayAttr(ctx, staticOffsets),
                                 getIntArrayAttr(ctx, staticSizes))
            .getResult();
}

// Sink ops from the capsule block into the forOp body so that
// values captured by the forOp region are only capsule block arguments.
static void sinkCapsuleOpsIntoForBody(IE::CodeGenCapsuleOp capsule, mlir::scf::ForOp forOp,
                                      mlir::PatternRewriter& rewriter) {
    auto* capsuleBlock = capsule.getBody();
    rewriter.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
    mlir::DominanceInfo domInfo(capsule.getOperation());
    mlir::IRMapping mapper;
    for (auto& op : *capsuleBlock) {
        if (&op == forOp.getOperation() || mlir::isa<IE::CGCYieldOp>(op) || mlir::isa<mlir::tensor::EmptyOp>(op) ||
            mlir::isa<Shave::LoopTripCountOp>(op)) {
            continue;
        }
        if (mlir::isa<mlir::tensor::DimOp>(op)) {
            continue;
        }
        auto* clonedOp = rewriter.clone(op, mapper);
        for (auto [origResult, clonedResult] : llvm::zip(op.getResults(), clonedOp->getResults())) {
            rewriter.replaceUsesWithIf(origResult, clonedResult, [&](mlir::OpOperand& use) {
                return domInfo.properlyDominates(clonedOp, use.getOwner());
            });
        }
    }

    if (mlir::failed(mlir::runRegionDCE(rewriter, capsule.getBodyRegion()))) {
        VPUX_THROW("Failed to run region DCE after sinking capsule ops into for body");
    }
}

// Collect calling convention info for the kernel function by classifying
// values used in the loop body into slices, tensor/scalar operands,
// iteration space values, and return values.
static KernelCCInfo collectKernelCCInfo(mlir::Block* body, const llvm::SetVector<mlir::Value>& capturedValues) {
    KernelCCInfo ccInfo;
    auto loopYielded = mlir::cast<mlir::scf::YieldOp>(body->getTerminator());

    body->walk([&](mlir::Operation* op) {
        if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(op)) {
            if (capturedValues.contains(slice.getSource())) {
                ccInfo.slices.push_back(slice);
            }
            return;
        }

        if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
            if (yield == loopYielded) {
                auto yielded = op->getOperand(0).getDefiningOp();
                // Always safe, otherwise getTilablePositions would have failed.
                // Note that we currently shouldn't get here with multiple results.
                auto slice = mlir::cast<mlir::tensor::InsertSliceOp>(yielded);
                ccInfo.returnValues.push_back(slice.getOperand(0));
                return;
            }
        }

        if (auto outInfo = mlir::dyn_cast<Shave::OutSliceInfoOp>(op)) {
            for (auto size : outInfo.getSizes()) {
                ccInfo.iterationSpaceValues.push_back(size);
            }
            for (auto offset : outInfo.getOffsets()) {
                ccInfo.iterationSpaceValues.push_back(offset);
            }
            return;
        }
        for (auto operand : op->getOperands()) {
            if (capturedValues.contains(operand)) {
                if (mlir::isa<mlir::TensorType>(operand.getType())) {
                    ccInfo.tensorOps.insert(operand);
                } else {
                    ccInfo.scalarOps.insert(operand);
                }
            }
        }
    });

    return ccInfo;
}

// Outlines a tiled capsule loop into generated kernel and slice-info functions,
// then materializes a VPU::GenericSwLayerOp that carries the tiling metadata.
// Returns failure if we can't materialize valid input slices.
static mlir::FailureOr<VPU::GenericSwLayerOp> outlineLoop(IE::CodeGenCapsuleOp capsule, mlir::scf::ForOp forOp,
                                                          mlir::ModuleOp swModule, mlir::StringAttr swModuleRoot,
                                                          size_t counter, const SmallVector<int64_t>& tilablePositions,
                                                          mlir::PatternRewriter& rewriter) {
    auto body = forOp.getBody();

    if (forOp.getNumResults() > 1) {
        // E#219995: Back out here if we have more than one result until support for more than one
        // result is implemented. The loop structure is still okay for unrolling.
        return mlir::failure();
    }

    sinkCapsuleOpsIntoForBody(capsule, forOp, rewriter);

    llvm::SetVector<mlir::Value> capturedValues;
    mlir::getUsedValuesDefinedAbove(forOp.getRegion(), capturedValues);

    auto ccInfo = collectKernelCCInfo(body, capturedValues);

    auto kernelFunc = outlineTiledBody(ccInfo, body, swModule, forOp.getLoc(), counter);
    auto infoFunc = outlineSliceInfoFunc(ccInfo, body, swModule, forOp.getLoc(), counter);

    auto infoFuncSymRef = mlir::FlatSymbolRefAttr::get(infoFunc);
    auto kernelFuncSymRef = mlir::SymbolRefAttr::get(
            swModuleRoot, llvm::ArrayRef<mlir::FlatSymbolRefAttr>(mlir::FlatSymbolRefAttr::get(kernelFunc)));

    // Attach the kernel info attribute to the kernel function.
    auto* ctx = forOp.getContext();
    auto i64Ty = mlir::IntegerType::get(ctx, 64);
    auto kernelInfoAttr = VPU::KernelInfoAttr::get(
            ctx,
            /*tilingInfoFunc=*/infoFuncSymRef,
            /*tilingAxes=*/mlir::DenseI64ArrayAttr::get(ctx, tilablePositions),
            /*numSlicedInputs=*/mlir::IntegerAttr::get(i64Ty, static_cast<int64_t>(ccInfo.slices.size())));
    kernelFunc->setAttr(VPU::KernelInfoAttr::kFuncAttrName, kernelInfoAttr);

    // Construct the iteration space from the first result sizes.
    auto resultSizes = mlir::tensor::getMixedSizes(rewriter, forOp.getLoc(), forOp.getInitsMutable()[0].get());
    SmallVector<mlir::OpFoldResult> tilableSizes;
    for (auto pos : tilablePositions) {
        tilableSizes.push_back(resultSizes[pos]);
    }

    SmallVector<int64_t> staticOffsets(tilableSizes.size(), 0);
    SmallVector<int64_t> staticSizes;
    SmallVector<mlir::Value> dynamicSizes;
    std::tie(staticSizes, dynamicSizes) = mlir::decomposeMixedValues(tilableSizes);

    auto tilingAttr = vpux::VPU::JITTilingAttr::get(forOp.getContext(),
                                                    mlir::DenseI64ArrayAttr::get(forOp.getContext(), staticSizes),
                                                    mlir::DenseI64ArrayAttr::get(forOp.getContext(), staticOffsets));
    rewriter.setInsertionPointAfter(forOp);

    SmallVector<mlir::Value> infoCallValues;
    for (auto scalar : ccInfo.scalarOps) {
        infoCallValues.push_back(scalar);
    }
    for (auto size : tilableSizes) {
        infoCallValues.push_back(mlir::getValueOrCreateConstantIndexOp(rewriter, forOp.getLoc(), size));
    }
    auto zeroIdx = rewriter.create<mlir::arith::ConstantIndexOp>(forOp.getLoc(), 0);
    for (size_t i = 0; i < tilableSizes.size(); ++i) {
        infoCallValues.push_back(zeroIdx);
    }

    auto inlinedResults = inlineInputSliceInfoCall(infoFunc, infoCallValues, rewriter);

    SmallVector<mlir::Value> callValues;
    SmallVector<IE::SliceOp> createdSlices;
    auto cleanupCreatedSlices = [&]() {
        for (auto createdSlice : llvm::reverse(createdSlices)) {
            rewriter.eraseOp(createdSlice);
        }
        createdSlices.clear();
    };
    int64_t callValuePos = 0;
    for (auto slice : ccInfo.slices) {
        auto sliceRank = mlir::cast<mlir::RankedTensorType>(slice.getType()).getRank();
        auto newSlice = createSliceFromInlinedInfo(slice, inlinedResults, callValuePos, rewriter);
        if (mlir::failed(newSlice)) {
            // Do some cleanup and return failure. We'll unroll after this so the transformation
            // still succeeds in all cases. Any transformations that we've done so far and didn't
            // back out are compatible with unrolling and will be removed later (we'll need
            // canonicalization anyway after this). Note this is cheaper than alternatives like
            // checkpointing/cloning the entire capsule before doing the transformation.
            cleanupCreatedSlices();
            kernelFunc.erase();
            infoFunc.erase();

            return mlir::failure();
        }
        if (auto createdSlice = (*newSlice).getDefiningOp<IE::SliceOp>()) {
            createdSlices.push_back(createdSlice);
        }
        callValues.push_back(*newSlice);
        callValuePos += 2 * sliceRank;
    }
    // Further append the tensor and scalar operands. We don't need to append iteration space values
    // since these get passed through the dynamic sizes and in the tiling attribute (offsets are all zeros).
    callValues.append(ccInfo.tensorOps.begin(), ccInfo.tensorOps.end());
    callValues.append(ccInfo.scalarOps.begin(), ccInfo.scalarOps.end());
    auto swOp = rewriter.create<VPU::GenericSwLayerOp>(forOp.getLoc(), capsule.getResultTypes(), kernelFuncSymRef,
                                                       callValues, tilingAttr, dynamicSizes, mlir::ValueRange{});
    auto oldYieldOp = capsule.getBody()->getTerminator();
    rewriter.create<IE::CGCYieldOp>(forOp.getLoc(), swOp.getResults());
    rewriter.eraseOp(oldYieldOp);

    // Ensure we don't have any dead ops in order to avoid cloning them further.
    if (mlir::failed(mlir::runRegionDCE(rewriter, capsule.getBodyRegion()))) {
        // DCE only fails when erasing an op that still has uses, indicating an IR invariant violation.
        VPUX_THROW("Failed to run region DCE after outlining capsule loop");
    }

    return swOp;
}

// Unrolls a tiled capsule into a single-iteration form by replacing
// LoopTripCount with 1, OutSliceInfo results with the full init tensor sizes,
// and unrolling the single-iteration for loop.
static void unrollTiledCapsule(mlir::scf::ForOp forOp, const SmallVector<int64_t>& tilablePositions,
                               mlir::PatternRewriter& rewriter) {
    auto loc = forOp.getLoc();

    // Materialize the full init tensor sizes before the for loop.
    rewriter.setInsertionPoint(forOp);

    // Get the full iteration space, which is by construction the sizes of the first result
    // at the tilable positions. Note that we rely here on the fact that we've collected
    // tilable positions in ascending order.
    auto initSizes = mlir::tensor::getMixedSizes(rewriter, loc, forOp.getInitsMutable()[0].get());
    SmallVector<mlir::Value> infoValues;

    for (auto pos : tilablePositions) {
        infoValues.push_back(mlir::getValueOrCreateConstantIndexOp(rewriter, loc, initSizes[pos]));
    }

    // Append the offset values (all zeros).
    auto zeroIdx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    infoValues.append(tilablePositions.size(), zeroIdx);

    // Replace Shave::OutSliceInfoOp results with the full init tensor sizes.
    forOp.getBody()->walk([&](Shave::OutSliceInfoOp outInfo) {
        rewriter.replaceOp(outInfo, infoValues);
    });

    // Replace Shave::LoopTripCountOp with constant 1 and fully unroll the loop.
    auto tripCountOp = forOp.getUpperBound().getDefiningOp<Shave::LoopTripCountOp>();
    rewriter.setInsertionPoint(tripCountOp);
    auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    rewriter.replaceOp(tripCountOp, one.getResult());

    if (mlir::failed(forOp.promoteIfSingleIteration(rewriter))) {
        // This cannot fail since we've just changed the loop to have a single iteration.
        VPUX_THROW("Failed to unroll tiled capsule loop");
    }
}

// Inline the capsule body in place of the capsule op, inserting PermuteCasts
// to normalize non-identity dim orders on capsule operands.
static void inlineCapsuleBody(IE::CodeGenCapsuleOp op, mlir::PatternRewriter& rewriter) {
    auto* capsuleBlock = op.getBody();
    mlir::IRMapping mapper;
    rewriter.setInsertionPoint(op);
    for (auto operand : op->getOperands() | indexed) {
        auto val = operand.value();
        if (auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(val.getType())) {
            auto order = ndType.getDimsOrder();
            if (!order.isIdentity()) {
                auto* ctx = rewriter.getContext();
                auto identityOrder = DimsOrder::fromNumDims(order.numDims());
                auto dstOrderMap = identityOrder.toAffineMap(ctx);
                auto memPerm = identityOrder.toAffineMap(ctx);
                val = rewriter.create<IE::PermuteCastOp>(op.getLoc(), val, dstOrderMap, memPerm);
            }
        }
        mapper.map(capsuleBlock->getArguments()[operand.index()], val);
    }

    SmallVector<mlir::Value> replacements;
    for (auto& innerOp : *capsuleBlock) {
        if (auto yield = mlir::dyn_cast<IE::CGCYieldOp>(innerOp)) {
            for (auto yielded : yield->getOperands()) {
                replacements.push_back(mapper.lookup(yielded));
            }
            break;
        }
        if (auto sliceOp = mlir::dyn_cast<IE::SliceOp>(innerOp)) {
            auto mappedInput = mapper.lookupOrDefault(sliceOp.getInput());
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
            const auto outputTileType = mlir::cast<vpux::NDTypeInterface>(mappedInput.getType())
                                                .extractDenseTile(ShapeRef(sliceOffsets), ShapeRef(sliceSizes));
            auto newSlice = rewriter.create<IE::SliceOp>(sliceOp.getLoc(), outputTileType, mappedInput,
                                                         sliceOp.getStaticOffsetsAttr(), sliceOp.getStaticSizesAttr());
            mapper.map(sliceOp.getOutput(), newSlice.getOutput());
            continue;
        }
        rewriter.clone(innerOp, mapper);
    }
    rewriter.replaceOp(op, replacements);
}

// Rewrites a tilable CodeGenCapsule by outlining it to GenericSwLayer with
// tiling metadata, or falling back to single-iteration unrolling when needed.
struct OutlineTilableCodeGenCapsule : mlir::OpRewritePattern<IE::CodeGenCapsuleOp> {
    using OpRewritePattern::OpRewritePattern;

    explicit OutlineTilableCodeGenCapsule(mlir::MLIRContext* ctx, const mlir::ModuleOp swModule, size_t& counter,
                                          mlir::StringAttr swModuleRoot, bool forceUnroll)
            : OpRewritePattern<IE::CodeGenCapsuleOp>(ctx),
              _swModule(swModule),
              _counter(counter),
              _swModuleRoot(swModuleRoot),
              _forceUnroll(forceUnroll) {
    }

    mlir::LogicalResult matchAndRewrite(IE::CodeGenCapsuleOp op, mlir::PatternRewriter& rewriter) const override {
        auto tiledOp = getTiledOp(op);
        if (mlir::failed(tiledOp)) {
            return mlir::failure();
        }
        auto tilablePositions = getTilablePositions(*tiledOp);
        if (mlir::failed(tilablePositions)) {
            return mlir::failure();
        }

        if (_forceUnroll) {
            unrollTiledCapsule(*tiledOp, *tilablePositions, rewriter);
            return mlir::success();
        }

        if (mlir::failed(outlineLoop(op, *tiledOp, _swModule, _swModuleRoot, _counter, *tilablePositions, rewriter))) {
            unrollTiledCapsule(*tiledOp, *tilablePositions, rewriter);
            return mlir::success();
        }

        _counter++;

        inlineCapsuleBody(op, rewriter);
        return mlir::success();
    }

private:
    mlir::ModuleOp _swModule;
    size_t& _counter;
    mlir::StringAttr _swModuleRoot;
    bool _forceUnroll;
};

// Cleanup pattern for VPU::GenericSwLayerOp operands: removes identity
// IE::PermuteCastOps and folds IE::SliceOp(IE::PermuteCastOp(x)) into IE::SliceOp(x).
struct FoldSwLayerOperands : mlir::OpRewritePattern<VPU::GenericSwLayerOp> {
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::GenericSwLayerOp op, mlir::PatternRewriter& rewriter) const override {
        bool changed = false;
        SmallVector<mlir::Value> newOperands(op->getOperands().begin(), op->getOperands().end());
        for (auto [idx, operand] : llvm::enumerate(newOperands)) {
            auto permuteCast = operand.getDefiningOp<IE::PermuteCastOp>();
            if (permuteCast) {
                if (permuteCast.getDstOrder().isIdentity() && permuteCast.getMemPerm().isIdentity()) {
                    newOperands[idx] = permuteCast.getInput();
                    changed = true;
                }
                continue;
            }

            // Fold IE::SliceOp(IE::PermuteCastOp(x)) into IE::SliceOp(x) with permuted offsets/sizes.
            auto sliceOp = operand.getDefiningOp<IE::SliceOp>();
            if (!sliceOp) {
                continue;
            }
            auto sliceInputPermuteCast = sliceOp.getInput().getDefiningOp<IE::PermuteCastOp>();
            if (!sliceInputPermuteCast || !sliceInputPermuteCast.getDstOrder().isIdentity() ||
                !sliceInputPermuteCast.getMemPerm().isIdentity()) {
                continue;
            }

            auto permuteCastInput = sliceInputPermuteCast.getInput();
            auto inputNdType = mlir::cast<vpux::NDTypeInterface>(permuteCastInput.getType());
            auto* ctx = rewriter.getContext();
            auto inputMemMap = inputNdType.getDimsOrder().toAffineMap(ctx);
            auto inverseMap = mlir::inversePermutation(inputMemMap);

            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            const auto sliceSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
            auto permutedOffsets = mlir::applyPermutationMap<int64_t>(inverseMap, sliceOffsets);
            auto permutedSizes = mlir::applyPermutationMap<int64_t>(inverseMap, sliceSizes);
            const auto outputTileType =
                    inputNdType.extractDenseTile(ShapeRef(permutedOffsets), ShapeRef(permutedSizes));

            auto newSlice = rewriter.create<IE::SliceOp>(sliceOp.getLoc(), outputTileType, permuteCastInput,
                                                         getIntArrayAttr(ctx, permutedOffsets),
                                                         getIntArrayAttr(ctx, permutedSizes));
            newOperands[idx] = newSlice.getOutput();
            changed = true;
        }
        if (!changed) {
            return mlir::failure();
        }
        rewriter.modifyOpInPlace(op, [&]() {
            op->setOperands(newOperands);
        });
        return mlir::success();
    }
};

void OutlineCodeGenCapsulesPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto moduleOp = getOperation();
    auto func = net::getMainFunc(moduleOp);

    auto swModule = VPUIP::getVPUSWModule(moduleOp, _log);
    size_t counter = 0;

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<OutlineCodeGenCapsule>(&ctx, swModule, counter, swModule.getSymNameAttr());
    patterns.insert<OutlineTilableCodeGenCapsule>(&ctx, swModule, counter, swModule.getSymNameAttr(), forceUnroll);
    patterns.insert<FoldSwLayerOperands>(&ctx);
    if (failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
    }
}

}  // namespace

//
// createOutlineCodeGenCapsulesPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createOutlineCodeGenCapsulesPass(Logger log) {
    return std::make_unique<OutlineCodeGenCapsulesPass>(log);
}
