//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_multicluster_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_unroll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Dialect/Affine/Analysis/Utils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_FULLUNROLLSCFLOOP
#define GEN_PASS_DEF_FULLUNROLLSCFLOOP
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

class ConvertSlice final : public mlir::OpInterfaceRewritePattern<mlir::OffsetSizeAndStrideOpInterface> {
public:
    ConvertSlice(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::OffsetSizeAndStrideOpInterface>(ctx, vpux::benefitLow), _log(log) {
        setDebugName("ConvertSlice");
    }

    mlir::LogicalResult matchAndRewrite(mlir::OffsetSizeAndStrideOpInterface convOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertSlice::matchAndRewrite(mlir::OffsetSizeAndStrideOpInterface sliceOp,
                                                  mlir::PatternRewriter& rewriter) const {
    if (sliceOp->getParentOfType<mlir::scf::ForOp>() != nullptr ||
        sliceOp->getName().getDialectNamespace() != mlir::tensor::TensorDialect::getDialectNamespace()) {
        return mlir::failure();
    }

    auto offsets = mlir::getConstantIntValues(sliceOp.getMixedOffsets());
    auto sizes = mlir::getConstantIntValues(sliceOp.getMixedSizes());
    auto strides = mlir::getConstantIntValues(sliceOp.getMixedStrides());

    if (!offsets.has_value() || !sizes.has_value()) {
        return mlir::failure();
    }

    auto insertOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(sliceOp.getOperation());

    if (insertOp != nullptr) {
        auto destinations = insertOp.getDpsInits();
        if (destinations.size() > 1) {
            return mlir::failure();
        }

        auto destOperation = destinations.front().getDefiningOp();

        if (getShape(insertOp.getDpsInputOperand(0)->get()).isDynamic()) {
            return mlir::failure();
        }

        auto input = insertOp.getDpsInputOperand(0)->get();

        if (auto emptyOp = mlir::dyn_cast<mlir::tensor::EmptyOp>(destOperation)) {
            /*
                For situation when insert_slice writes directly into result buffer,

                result_buffer = tensor.empty
                operation = VPU.Operation
                slice = tensor.insert_slice operation into result_buffer[offsets][sizes]

                just replace insert slice to operation
            */
            rewriter.replaceOp(sliceOp, input);
        } else if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(destOperation)) {
            /*
                For situation when insert_slice writes its result to the Concat,
                extend the concat instead of the slice
                result_buffer = VPU.Concat operations [offsets0][sizes0]
                operation0 = VPU.Operation
                slice = tensor.insert_slice operation into result_buffer[offsets1][sizes1]

                ->

                operation0 = VPU.Operation
                VPU.Concat operations + operation0 [offsets0 + offsets1][sizes0 + sizes1]

            */
            auto inputs = to_small_vector(concatOp.getInputs());
            inputs.emplace_back(input);

            auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.getStaticOffsets().value());
            concatOffsets.emplace_back(offsets.value());

            rewriter.replaceOpWithNewOp<VPU::ConcatOp>(sliceOp, inputs, /*per_axis=*/nullptr,
                                                       getIntArrayOfArray(rewriter.getContext(), concatOffsets));
        } else {
            /*
               For situation when insert_slice writes its result to operation
               substituted on the previous steps,
               create the concat instead
               result_buffer = VPU.Operation
               operation1 = VPU.Operation
               slice = tensor.insert_slice operation into result_buffer[offsets1][sizes1]

               ->

               operation0 = VPU.Operation
               operation1 = VPU.Operation
               VPU.Concat operation0 + operation1 [offsets][sizes]

           */
            SmallVector<mlir::Value> inputs;
            inputs.emplace_back(destinations.front());
            inputs.emplace_back(input);

            SmallVector<SmallVector<int64_t>> concatOffsets;
            concatOffsets.emplace_back(SmallVector<int64_t>(offsets.value().size(), 0));
            concatOffsets.emplace_back(offsets.value());

            rewriter.replaceOpWithNewOp<VPU::ConcatOp>(sliceOp, inputs, /*per_axis=*/nullptr,
                                                       getIntArrayOfArray(rewriter.getContext(), concatOffsets));
        }

    } else {
        /*
             extract_slice -> VPU.Slice
        */

        const auto notEqualOne = [](int64_t stride) {
            return stride != 1;
        };
        if (strides.has_value() && std::any_of(strides.value().begin(), strides.value().end(), notEqualOne)) {
            return mlir::failure();
        }

        auto source = sliceOp->getOperand(0);
        const auto oneShape = Shape(sizes.value().size(), 1);
        const auto tileInfo = TileInfo(ShapeRef(sizes.value()), ShapeRef(offsets.value()), oneShape);

        auto newSliceOp = VPU::makeTile(rewriter, sliceOp.getLoc(), source, tileInfo, "converted_from_extract_slice");
        rewriter.replaceOp(sliceOp, newSliceOp);
    }

    return mlir::success();
}

class SimplifyDynamicCast final : public mlir::OpRewritePattern<mlir::tensor::CastOp> {
public:
    SimplifyDynamicCast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::tensor::CastOp>(ctx, vpux::benefitHigh), _log(log) {
        setDebugName("SimplifyDynamicCast");
    }

    mlir::LogicalResult matchAndRewrite(mlir::tensor::CastOp castOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    mlir::Operation* adaptSourceOp(mlir::scf::ForallOp loop, mlir::Value castInput, mlir::ShapedType resultType,
                                   mlir::PatternRewriter& rewriter, Logger log) const;
    mlir::Operation* adaptSourceOp(mlir::InferTypeOpInterface inferRetTypeIfOp, mlir::PatternRewriter& rewriter,
                                   Logger log) const;
};

mlir::Operation* SimplifyDynamicCast::adaptSourceOp(mlir::scf::ForallOp loop, mlir::Value castInput,
                                                    mlir::ShapedType resultType, mlir::PatternRewriter& rewriter,
                                                    Logger log) const {
    if (loop == nullptr) {
        return nullptr;
    }

    auto sharedOutputs = loop.getOutputsMutable();

    mlir::IRMapping mapper;

    int64_t resultIdx = -1;
    for (auto& outOperand : sharedOutputs) {
        auto result = loop.getTiedOpResult(&outOperand);
        if (castInput != result) {
            // current shared_out is not connected to the tensor.cast input
            continue;
        }

        auto loopOutBuffer = mlir::dyn_cast<mlir::tensor::EmptyOp>(outOperand.get().getDefiningOp());
        if (loopOutBuffer == nullptr) {
            // current shared_out is not provided by a tensor.empty op
            continue;
        }

        rewriter.setInsertionPoint(loop);
        auto newEmptyOp = rewriter.create<mlir::tensor::EmptyOp>(loopOutBuffer->getLoc(), resultType.getShape(),
                                                                 resultType.getElementType());
        mapper.map(outOperand.get(), newEmptyOp);
        resultIdx = result.getResultNumber();
        break;
    }

    if (resultIdx < 0) {
        log.trace("Cannot simplify dynamic cast after forall op: tensor.cast input is not tied to a tensor.empty op.");
        return nullptr;
    }

    auto newOp = rewriter.clone(*loop, mapper);
    auto newLoop = mlir::cast<mlir::scf::ForallOp>(newOp);

    rewriter.modifyOpInPlace(newLoop, [&]() {
        // update result type of loop
        auto currentResult = newLoop->getOpResult(resultIdx);
        currentResult.setType(resultType);
        // update type of the OpOperand tied to the above shared_out
        auto tiedOpOperand = newLoop.getTiedOpOperand(currentResult);
        tiedOpOperand->get().setType(resultType);

        auto* terminator = newLoop.getBody()->getTerminator();
        if (auto inParallelOp = mlir::dyn_cast_if_present<mlir::scf::InParallelOp>(terminator)) {
            auto parallelInsertSliceOps = inParallelOp.getOps<mlir::tensor::ParallelInsertSliceOp>();

            auto tiedBlockArg = newLoop.getTiedBlockArgument(tiedOpOperand);
            // find the tensor.parallel_insert_slice op which inserts into the current result
            auto insertOpIt =
                    llvm::find_if(parallelInsertSliceOps, [&](mlir::tensor::ParallelInsertSliceOp insertSliceOp) {
                        auto dst = insertSliceOp.getDest();
                        return dst == tiedBlockArg;
                    });

            if (insertOpIt != parallelInsertSliceOps.end()) {
                log.trace("Found insertOp.");
                auto insertOp = *insertOpIt;

                // update type of parallel_insert_slice's destination
                insertOp.getDestMutable().get().setType(resultType);

                auto origOutType = mlir::cast<NDTypeInterface>(insertOp->getOperand(0).getType());
                auto newShape = Shape(origOutType.getShape());
                auto newBounds = Shape(getBoundedShape(origOutType));
                auto dstShape = resultType.getShape();

                for (auto idx : irange(newShape.size())) {
                    if (origOutType.getShape()[Dim(idx)] != mlir::ShapedType::kDynamic) {
                        newShape[Dim(idx)] = dstShape[idx];
                        newBounds[Dim(idx)] = dstShape[idx];
                    }
                }

                // set new static sizes to reflect the updated shapes after dynamic cast propagation
                insertOp.setStaticSizes(newShape.raw());

                // if there is a tensor.cast op inside the loop body, which goes into the parallel_insert_slice,
                // it should also be updated with the new shapes in mind
                auto insertedSliceCast = insertOp->getOperand(0).getDefiningOp<mlir::tensor::CastOp>();
                if (insertedSliceCast != nullptr) {
                    rewriter.setInsertionPoint(insertedSliceCast);
                    mlir::Type newType = origOutType.changeShape(newShape);
                    if (auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(newType)) {
                        const auto bounds = Bounds(newBounds.raw());
                        newType = boundedType.changeBounds(bounds);
                    }

                    log.trace("Replace tensor.cast found inside the forall loop; new cast output type: {0}", newType);
                    auto newCastOp = rewriter.create<mlir::tensor::CastOp>(
                            insertedSliceCast->getLoc(), mlir::TypeRange{newType}, insertedSliceCast->getOperands());

                    rewriter.replaceOp(insertedSliceCast, newCastOp);
                }
            }
        }
    });

    return newLoop;
}

mlir::Operation* SimplifyDynamicCast::adaptSourceOp(mlir::InferTypeOpInterface inferRetTypeIfOp,
                                                    mlir::PatternRewriter& rewriter, Logger log) const {
    if (inferRetTypeIfOp == nullptr) {
        return nullptr;
    }

    auto operands = inferRetTypeIfOp->getOperands();
    mlir::IRMapping mapper;

    for (auto operand : operands) {
        auto upperCast = operand.getDefiningOp<mlir::tensor::CastOp>();
        if (upperCast == nullptr) {
            continue;
        }
        log.trace("Found upper cast op.");

        auto upperSourceType = mlir::cast<mlir::ShapedType>(upperCast.getSource().getType());
        auto upperResultType = mlir::cast<mlir::ShapedType>(upperCast.getResult().getType());

        if (!upperSourceType.hasStaticShape() || upperResultType.hasStaticShape()) {
            log.trace("Cannot simplify dynamic cast: upper cast is not supported.");
            return nullptr;
        }

        mapper.map(operand, upperCast.getSource());
    }

    auto newOp = rewriter.clone(*inferRetTypeIfOp.getOperation(), mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    return newOp;
}

mlir::LogicalResult SimplifyDynamicCast::matchAndRewrite(mlir::tensor::CastOp castOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Simplify dynamic tensor.cast op @ {0}", castOp->getLoc());
    // check if cast is dynamic to static
    auto sourceType = mlir::cast<mlir::ShapedType>(castOp.getSource().getType());
    auto resultType = mlir::cast<mlir::ShapedType>(castOp.getResult().getType());

    const auto& nestedLog = _log.nest();

    if (sourceType.hasStaticShape() || !resultType.hasStaticShape()) {
        if (sourceType.hasStaticShape()) {
            // Collect users before replaceOp, because getUsers() returns a live view
            // of the use-list which becomes empty after replaceOp rewires all uses
            SmallVector<mlir::Operation*> userOps(castOp.getResult().getUsers().begin(),
                                                  castOp.getResult().getUsers().end());
            rewriter.replaceOp(castOp, castOp.getSource());
            if (!resultType.hasStaticShape()) {
                for (auto* user : userOps) {
                    if (mlir::isa<mlir::InferTypeOpInterface>(user)) {
                        vpux::inferReturnTypes(user, vpux::InferShapedTypeMode::SHAPE);
                    }
                }
            }
            return mlir::success();
        }
        return matchFailed(nestedLog, rewriter, castOp,
                           "Cannot simplify dynamic cast with static source shape and dynamic result shape.");
    }

    auto sourceOp = castOp.getSource().getDefiningOp();
    mlir::Operation* updatedOp = nullptr;
    if (auto inferRetTypeIfOp = mlir::dyn_cast_if_present<mlir::InferTypeOpInterface>(sourceOp)) {
        updatedOp = adaptSourceOp(inferRetTypeIfOp, rewriter, nestedLog);
    } else if (auto loopOp = mlir::dyn_cast_if_present<mlir::scf::ForallOp>(sourceOp)) {
        updatedOp = adaptSourceOp(loopOp, castOp.getSource(), resultType, rewriter, nestedLog);
    } else {
        return matchFailed(nestedLog, rewriter, castOp, "Cannot simplify dynamic cast with unsupported source op.");
    }

    if (updatedOp == nullptr) {
        return matchFailed(nestedLog, rewriter, castOp,
                           "Could not update source op in order to propagate the dynamic cast.");
    }

    rewriter.replaceOp(castOp, updatedOp->getResults());

    return mlir::success();
}

//
// FullUnrollSCFLoopPass
//
class FullUnrollSCFLoopPass final : public VPU::impl::FullUnrollSCFLoopBase<FullUnrollSCFLoopPass> {
public:
    explicit FullUnrollSCFLoopPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void unrollTiling(ArrayRef<mlir::scf::ForOp> loopVector, mlir::ModuleOp moduleOp);
    void unrollMulticlustering(mlir::ModuleOp moduleOp);
};

static std::optional<int64_t> getConstantTripCount(mlir::scf::ForOp forOp) {
    std::optional<int64_t> lbCstOp = getConstantIntValue(forOp.getLowerBound());
    std::optional<int64_t> ubCstOp = getConstantIntValue(forOp.getUpperBound());
    std::optional<int64_t> stepCstOp = getConstantIntValue(forOp.getStep());
    if (!lbCstOp.has_value() || !ubCstOp.has_value() || !stepCstOp.has_value()) {
        return {};
    }

    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst > 0 && "expected positive loop bounds and step");
    return llvm::divideCeilSigned(ubCst - lbCst, stepCst);
}

void FullUnrollSCFLoopPass::unrollTiling(ArrayRef<mlir::scf::ForOp> loopVector,
                                         [[maybe_unused]] mlir::ModuleOp moduleOp) {
    // full unrolling of tiling loops
    for (auto& loop : loopVector) {
        const auto tripCountOpt = getConstantTripCount(loop);

        VPUX_THROW_WHEN(!tripCountOpt.has_value(),
                        "Full unrolling is not supported for loop {0} with dynamic trip count", loop);

        (void)mlir::loopUnrollByFactor(loop, tripCountOpt.value());
    }

    // canonicalization patterns
    auto& ctx = *moduleOp.getContext();
    mlir::RewritePatternSet patterns(&ctx);
    ctx.getLoadedDialect<mlir::arith::ArithDialect>()->getCanonicalizationPatterns(patterns);
    ctx.getLoadedDialect<mlir::affine::AffineDialect>()->getCanonicalizationPatterns(patterns);
    ctx.getLoadedDialect<mlir::tensor::TensorDialect>()->getCanonicalizationPatterns(patterns);
    mlir::scf::IfOp::getCanonicalizationPatterns(patterns, &ctx);
    patterns.add<ConvertSlice>(&ctx, _log);
    patterns.add<SimplifyDynamicCast>(&ctx, _log);

    if (mlir::failed(applyPatternsGreedily(moduleOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    const auto inferShapeOps = [&]() {
        moduleOp->walk([&](mlir::Operation* operation) {
            if (operation->getNumResults() == 0 || !mlir::isa<mlir::InferTypeOpInterface>(operation)) {
                return;
            }

            auto type = mlir::dyn_cast<vpux::NDTypeInterface>(operation->getResult(0).getType());

            // skip interring Eltwise type one of its inputs has static shape and pther one has dynamic shape
            // due to having padded operation on one of Eltwise branches before padding is merged to the operation
            const auto isEltwiseWithPaddedDynamicInput = [&operation]() {
                if (!operation->hasTrait<VPU::EltwiseOp>() || operation->getNumOperands() != 2) {
                    return false;
                }

                auto input1 = operation->getOperand(0);
                auto input2 = operation->getOperand(1);
                auto in1Shape = getShape(input1);
                auto in2Shape = getShape(input2);

                if (in1Shape == in2Shape || (in1Shape.isStatic() && in2Shape.isStatic())) {
                    return false;
                }

                const auto isPaddedInput = [&in1Shape](auto value1, auto value2) {
                    auto opFirstInput = value1.getDefiningOp();
                    auto opSecondInput = value2.getDefiningOp();

                    if (opFirstInput == nullptr || opSecondInput == nullptr) {
                        return false;
                    }

                    auto dynamicOpInput = in1Shape.isDynamic() ? opFirstInput : opSecondInput;
                    auto staticOpInput = in1Shape.isStatic() ? opFirstInput : opSecondInput;

                    while (dynamicOpInput != nullptr && dynamicOpInput != staticOpInput) {
                        if (VPU::isNceOpWithPadAttr(dynamicOpInput)) {
                            return true;
                        }

                        dynamicOpInput = dynamicOpInput->getOperand(0).getDefiningOp();
                    }
                    return false;
                };

                return isPaddedInput(input1, input2);
            };
            if (type != nullptr && !isEltwiseWithPaddedDynamicInput()) {
                vpux::inferReturnTypes(operation, vpux::InferShapedTypeMode::SHAPE);
            }
        });
    };

    inferShapeOps();

    restorePaddingAttribute(moduleOp, _log.nest());

    inferShapeOps();

    // operations' transformation
    patterns.clear();
    patterns.add<ConvertSlice>(&ctx, _log);
    patterns.add<SimplifyDynamicCast>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(moduleOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    // Re-infer types after SimplifyDynamicCast removes tensor.cast ops, since
    // inferReturnTypes in that pattern only updates direct users, not the full chain
    inferShapeOps();

    // After type propagation, tensor.insert_slice/extract_slice ops may have stale
    // dynamic size declarations incompatible with now-static source types. Run
    // ConvertSlice again to convert them to VPU.Concat/VPU.Slice.
    mlir::RewritePatternSet finalPatterns(&ctx);
    finalPatterns.add<ConvertSlice>(&ctx, _log);
    if (mlir::failed(
                mlir::applyPatternsGreedily(moduleOp, std::move(finalPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

VPU::DistributionMode getInputNonDuplicatedMode(VPU::ClusteredOpInterface clusteredOp, mlir::OpOperand* input,
                                                NDTypeInterface inputType, const VPU::MultiClusterStrategy& strategy) {
    if (mlir::isa<SWOpInterface>(clusteredOp.getOperation())) {
        return VPU::getSWInputTensorDistributionMode(clusteredOp, strategy, input->get(), inputType);
    }

    if (auto gatherDMA = mlir::dyn_cast<VPU::GatherDMAOp>(clusteredOp.getOperation())) {
        return VPU::getActivationTensorDistributionMode(gatherDMA, strategy, input->get());
    }

    VPUX_THROW_WHEN(!mlir::isa<NCEOpInterface>(clusteredOp.getOperation()),
                    "Unsupported clustered operation @ {0}: not SW op, DPU op or GatherDMAOp", clusteredOp.getLoc());

    if (clusteredOp.getOperation()->hasTrait<VPU::EltwiseOp>() || input->get() == clusteredOp->getOperand(0)) {
        return VPU::getActivationTensorDistributionMode(clusteredOp, strategy);
    }

    return VPU::getWeightsTensorDistributionMode(strategy);
}

VPU::DistributedTensorType getDistributedTypeForInput(VPU::OpChainAnalysis& analysis, mlir::Operation* computeOp,
                                                      mlir::OpOperand* input,
                                                      mlir::tensor::ExtractSliceOp extractSliceOp,
                                                      [[maybe_unused]] mlir::tensor::PadOp padOp,
                                                      const VPU::MultiClusterStrategy& strategy,
                                                      mlir::IntegerAttr numClustersAttr, mlir::MLIRContext* ctx) {
    const auto memSpaceCMX = IndexedSymbolAttr::get(ctx, stringifyEnum(MemoryKind::CMX_NN));
    if (extractSliceOp == nullptr) {
        // No extract slice on input means we have a non-tiled input, which should be duplicated
        auto inputType = mlir::cast<NDTypeInterface>(input->get().getType());
        // Use bounded shape to resolve dynamic dims for the distributed type.
        const auto shape = getBoundedShape(inputType);
        auto distrModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::DUPLICATED);
        auto distribution =
                getNonOverlappedDistributedAttr(shape, distrModeAttr, /*numTiles=*/nullptr, numClustersAttr,
                                                /*alignment=*/nullptr, /*uniformDistributedSegments=*/nullptr, ctx);

        auto orderAttr = mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx));
        return VPU::DistributedTensorType::get(ctx, shape, inputType.getElementType(), orderAttr, memSpaceCMX,
                                               distribution);
    }

    const auto numClusters = numClustersAttr.getInt();
    auto inputType = mlir::cast<NDTypeInterface>(extractSliceOp.getSource().getType());

    VPU::DistributionInfo distribution;
    auto clusterOp = mlir::cast<VPU::ClusteredOpInterface>(computeOp);
    const auto mode = getInputNonDuplicatedMode(clusterOp, input, inputType, strategy);
    distribution.setDistributionMode(mode);

    fillInDistribution(analysis, extractSliceOp, inputType, numClusters, distribution);

    // Resolve dynamic dims to concrete values using the computed distribution.
    const auto resolvedShape = resolveShapeFromDistribution(inputType.getShape().raw(), distribution);

    if (mode == VPU::DistributionMode::SEGMENTED) {
        const auto isWeights = mlir::isa<NCEOpInterface>(computeOp) && !computeOp->hasTrait<VPU::EltwiseOp>() &&
                               input->get() != computeOp->getOperand(0);
        const auto channelSize = resolvedShape[Dims4D::Filter::OC.ind()];
        const auto alignment = isWeights ? getWeightsTensorAlignment(clusterOp, strategy, numClusters, channelSize)
                                                   .value_or(SmallVector<int64_t>{})
                                         : vpux::getAlignment(computeOp, ShapeRef(distribution.getNumTiles()),
                                                              ShapeRef(resolvedShape));

        const auto distributionAxis = VPU::getDistributedTilingAxis(distribution.getNumTiles());
        if (alignment[distributionAxis] != 1) {
            distribution.setAlignment(alignment);
        }
    }

    auto distributionAttr = VPU::DistributionInfo::getAttrFromClass(ctx, distribution);

    auto orderAttr = mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx));
    return VPU::DistributedTensorType::get(ctx, resolvedShape, inputType.getElementType(), orderAttr, memSpaceCMX,
                                           distributionAttr);
}

void FullUnrollSCFLoopPass::unrollMulticlustering(mlir::ModuleOp moduleOp) {
    auto analysis = VPU::OpChainAnalysis(_log.nest());
    auto ctx = moduleOp.getContext();
    mlir::OpBuilder builder(ctx);

    auto getPadAttribute = [&](mlir::tensor::PadOp padOp) {
        auto spatialDims = {Dims4D::Act::W, Dims4D::Act::H};
        llvm::SmallVector<int64_t> padValues;
        for (auto dim : spatialDims) {
            auto lowPad = padOp.getMixedLowPad()[dim.ind()];
            auto highPad = padOp.getMixedHighPad()[dim.ind()];

            ValueRangeMap emptyMap;
            auto lowValue = analysis.getOpFoldResultValue(lowPad, emptyMap);
            auto highValue = analysis.getOpFoldResultValue(highPad, emptyMap);
            VPUX_THROW_WHEN(!lowValue.has_value() || !highValue.has_value(),
                            "Failed to compute static padding values for {0} operation", padOp->getName());
            auto lv = lowValue.value();
            auto hv = highValue.value();

            _log.trace("Padding for dim {0}: low={1}, high={2}", dim, lv, hv);
            padValues.emplace_back(lowValue.value()[0]);
            padValues.emplace_back(highValue.value()[0]);
        }

        return VPU::getPaddingAttr(padOp.getContext(), padValues[0], padValues[1], padValues[2], padValues[3]);
    };

    auto getInputValue = [](mlir::Value input, mlir::tensor::ExtractSliceOp extractSliceOp) -> mlir::Value {
        if (extractSliceOp != nullptr) {
            return extractSliceOp.getSource();
        }

        if (auto copyOp = mlir::dyn_cast_if_present<VPU::CopyOp>(input.getDefiningOp())) {
            return copyOp.getInput();
        }

        return input;
    };

    SmallVector<mlir::Operation*> opsToErase;
    moduleOp->walk([&](mlir::scf::ForallOp forallOp) {
        _log.trace("Processing forall loop at {0}", forallOp->getLoc());
        auto clusteredOpIf = forallOp.getOps<VPU::ClusteredOpInterface>();
        const auto numClusteredIfOps = std::distance(clusteredOpIf.begin(), clusteredOpIf.end());

        // Multiple ops inside the scf.forall loop will be allowed after E#192457.
        VPUX_THROW_WHEN(numClusteredIfOps != 1, "Expected only one tiling interface op in forall loop at {0}",
                        forallOp->getLoc());

        auto inductionArgs = forallOp.getInductionVars();
        VPUX_THROW_WHEN(inductionArgs.size() != 1, "Expected multiclustering strategy with single axis, got {0}",
                        inductionArgs.size());

        ValueRangeMap emptyMap;
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(inductionArgs.front());
        VPUX_THROW_WHEN(blockArg == nullptr, "Induction arg for scf.forall loop is not a BlockArg");

        auto inductionDimRange = analysis.getForallInductionDimRange(forallOp, blockArg, emptyMap);
        const auto numClusters = static_cast<int64_t>(inductionDimRange.size());
        VPUX_THROW_WHEN(numClusters <= 1, "Cannot unroll forall with single iteration at {0}", forallOp->getLoc());

        auto numClustersAttr = getIntAttr(ctx, numClusters);

        const auto computeOp = *clusteredOpIf.begin();
        const auto& nestedLog = _log.nest();

        // Infer the multiclustering strategy from result 0. All results of the same
        // compute op share the same strategy because the tiling axis is identical.
        nestedLog.trace("Inferring multiclustering strategy from result 0.");
        auto strategy = inferMulticlusterStrategy(analysis, computeOp, computeOp->getOpResult(0));

        // Verify that every result produces the same strategy as result 0.
        // This guards against future ops where results might have different distributions.
        for (auto result : llvm::drop_begin(computeOp->getOpResults())) {
            const auto resultStrategy = inferMulticlusterStrategy(analysis, computeOp, result);
            VPUX_THROW_WHEN(resultStrategy != strategy,
                            "Result {0} of op at {1} infers a different multiclustering strategy than "
                            "result 0: expected {2}, got {3}",
                            result.getResultNumber(), computeOp->getLoc(), VPU::stringifyMultiClusterStrategy(strategy),
                            VPU::stringifyMultiClusterStrategy(resultStrategy));
        }

        // Compute the distributed type for each result.
        nestedLog.trace("Computing DistributedTensorType for {0} output(s).", computeOp->getNumResults());
        SmallVector<VPU::DistributedTensorType> outDistributedTypes;
        outDistributedTypes.reserve(computeOp->getNumResults());
        for (auto result : computeOp->getOpResults()) {
            outDistributedTypes.push_back(
                    getOutputDistributedType(analysis, computeOp, result, strategy, numClustersAttr, ctx));
        }

        mlir::IRMapping mapper;
        builder.setInsertionPointAfter(forallOp);
        mlir::tensor::PadOp padOp = nullptr;
        for (auto& input : computeOp->getOpOperands()) {
            auto extractSliceResult = input.get();

            auto maybePadOp = mlir::dyn_cast_if_present<mlir::tensor::PadOp>(input.get().getDefiningOp());
            if (maybePadOp != nullptr) {
                VPUX_THROW_WHEN(padOp != nullptr, "Expected only one tensor.pad op as input for compute op at {0}",
                                computeOp->getLoc());
                padOp = maybePadOp;

                extractSliceResult = padOp.getSource();
            }

            if (auto copyOp = mlir::dyn_cast_if_present<VPU::CopyOp>(extractSliceResult.getDefiningOp())) {
                extractSliceResult = copyOp.getInput();
            }

            auto extractSliceOp =
                    mlir::dyn_cast_if_present<mlir::tensor::ExtractSliceOp>(extractSliceResult.getDefiningOp());
            VPUX_THROW_WHEN(extractSliceOp == nullptr && maybePadOp != nullptr,
                            "Cannot have tensor.pad op on input without a tensor.extract_slice parent");

            auto distributedType = getDistributedTypeForInput(analysis, computeOp, &input, extractSliceOp, padOp,
                                                              strategy, numClustersAttr, ctx);

            const auto inputIdx = std::distance(computeOp->getOpOperands().begin(), &input);

            nestedLog.trace("Inserting distributed CopyOp for input {0}.", inputIdx);
            auto distributedCopyOp = builder.create<VPU::CopyOp>(
                    appendLoc(computeOp->getLoc(), "copy_in{0}", inputIdx), distributedType,
                    getInputValue(input.get(), extractSliceOp), distributedType.getMemSpace());

            // map current input to the output of the distributed Copy op
            mapper.map(input.get(), distributedCopyOp.getOutput());
        }

        nestedLog.trace("Clone compute op outside of loop.");
        auto clonedOp = builder.clone(*computeOp, mapper);
        if (padOp != nullptr) {
            auto padding = getPadAttribute(padOp);
            VPUX_THROW_WHEN(!clonedOp->hasAttr("pad"),
                            "tensor.pad op is producer to an op that does not support pad attr.");

            nestedLog.trace("tensor.pad op found; replacing with padding attribute.");
            clonedOp->setAttr("pad", padding);
        }

        if (auto permuteOp = mlir::dyn_cast<VPU::NCEPermuteOp>(clonedOp)) {
            const auto expandedChannels = outDistributedTypes[0].getShape()[Dims4D::Act::C];
            permuteOp.setExpandedChannels(expandedChannels);
        }

        // Set distributed types and create copy-out ops for every result.
        for (auto resultIdx : irange(computeOp->getNumResults())) {
            clonedOp->getResult(resultIdx).setType(outDistributedTypes[resultIdx]);

            nestedLog.trace("Inserting distributed CopyOp for output {0}.", resultIdx);
            auto valToReplace = forallOp.getResult(resultIdx);
            auto copyOutType = mlir::cast<NDTypeInterface>(valToReplace.getType());

            if (valToReplace.hasOneUse() && mlir::isa_and_present<VPU::CopyOp>(*(valToReplace.user_begin()))) {
                auto copyOp = mlir::cast<VPU::CopyOp>(*(valToReplace.user_begin()));
                copyOutType = mlir::cast<NDTypeInterface>(copyOp.getOutput().getType());
                valToReplace = copyOp.getOutput();

                // push copy op before loop, so it gets erased first
                opsToErase.push_back(copyOp);
            }

            // When the forall result type has dynamic dims, resolve the shape to match
            // the distributed type so that CopyOp type inference stays consistent.
            // Use the compact type from the distributed type (static shape, no bounds,
            // no distribution) for the CopyOp, then cast back to the original dynamic
            // type so that tensor.insert_slice consumers remain valid.
            NDTypeInterface resolvedCopyOutType = copyOutType;
            if (copyOutType.getShape().isDynamic()) {
                auto compactType = mlir::cast<NDTypeInterface>(outDistributedTypes[resultIdx].getCompactType());
                resolvedCopyOutType = compactType.changeMemSpace(copyOutType.getMemSpace());
            }

            auto copyOut = builder.create<VPU::CopyOp>(appendLoc(computeOp->getLoc(), "copy_out{0}", resultIdx),
                                                       resolvedCopyOutType, clonedOp->getResult(resultIdx),
                                                       resolvedCopyOutType.getMemSpace());

            mlir::Value replacement = copyOut.getOutput();
            if (copyOutType.getShape().isDynamic()) {
                replacement = builder.create<mlir::tensor::CastOp>(computeOp->getLoc(), copyOutType, replacement);
            }
            valToReplace.replaceAllUsesWith(replacement);
        }
        opsToErase.push_back(forallOp);
    });

    for (auto op : llvm::make_early_inc_range(opsToErase)) {
        if (op->use_empty()) {
            op->erase();
        }
    }
}

void FullUnrollSCFLoopPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    // full unrolling is not applicable for host pipeline
    if (config::isHostCompileMode(moduleOp)) {
        return;
    }

    mlir::OpBuilder builder(moduleOp);

    SmallVector<mlir::scf::ForOp> loopVector;
    collectLoops(moduleOp.getOperation(), loopVector);

    if (!loopVector.empty()) {
        unrollTiling(loopVector, moduleOp);
    }

    unrollMulticlustering(moduleOp);
}
}  // namespace

//
// createUnrollSCFLoopPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createFullUnrollSCFLoopPass(Logger log) {
    return std::make_unique<FullUnrollSCFLoopPass>(log);
}
