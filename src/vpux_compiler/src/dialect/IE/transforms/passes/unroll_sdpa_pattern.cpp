//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/check_shrink_matmul_groups.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/locations.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_UNROLLSDPAPATTERN
#define GEN_PASS_DEF_UNROLLSDPAPATTERN
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// Helper Structures
//

struct SDPAPattern {
    IE::MatMulOp matMulOp;
    IE::AddOp addOp;        // Optional: attn_mask Add
    IE::ConcatOp concatOp;  // Optional: Concat before Softmax
    IE::SoftMaxOp softmaxOp;
    IE::SliceOp sliceOp;  // Optional: Slice after Softmax
    IE::MatMulOp matMulV;

    // Collect the ops and unrolled dimension on q,k,v path because they might also deserve to be unrolled
    SmallVector<std::pair<mlir::Operation*, Dim>> opsOnPathQ;
    SmallVector<std::pair<mlir::Operation*, Dim>> opsOnPathK;
    SmallVector<std::pair<mlir::Operation*, Dim>> opsOnPathV;
};

//
// UnrollSDPAPattern Rewriter Pattern
//

class UnrollSDPAPattern final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    UnrollSDPAPattern(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SoftMaxOp>(ctx), _log(log) {
        setDebugName("UnrollSDPAPattern");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool detectSDPAPattern(IE::SoftMaxOp softmaxOp, SDPAPattern& pattern) const;
    bool isUnrollingBeneficial(SDPAPattern& pattern) const;
    bool isSupportedOpOnQKVPath(mlir::Operation* op) const;
    void trackBackQKVPaths(mlir::Value value, SmallVector<std::pair<mlir::Operation*, Dim>>& ops) const;
    mlir::LogicalResult unrollAndRearrangePattern(SDPAPattern& pattern, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

bool UnrollSDPAPattern::isSupportedOpOnQKVPath(mlir::Operation* op) const {
    if (!op->hasOneUse()) {
        return false;
    }

    if (mlir::isa_and_present<IE::TransposeOp, IE::AffineReshapeOp>(op)) {
        return true;
    }

    if (mlir::isa_and_present<IE::AddOp, IE::MultiplyOp, IE::FullyConnectedOp>(op)) {
        return vpux::Const::isConstValue(op->getOperand(1));
    }

    if (auto fq = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(op)) {
        return IE::isPerTensorFQ({fq});
    }

    return false;
}

void UnrollSDPAPattern::trackBackQKVPaths(mlir::Value value, SmallVector<std::pair<mlir::Operation*, Dim>>& ops) const {
    auto dim = Dims4D::Act::C;  // layout [batch, num_heads, seq_len, head_size]
    auto currentOp = value.getDefiningOp();
    while (currentOp != nullptr && isSupportedOpOnQKVPath(currentOp)) {
        auto inputDim = dim;
        if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(currentOp)) {
            auto maybeOrderValue = transposeOp.getOrderValue();
            if (!maybeOrderValue.has_value()) {
                break;
            }

            auto orderValue = maybeOrderValue.value();
            VPUX_THROW_UNLESS(checked_cast<size_t>(dim.ind()) < orderValue.getNumResults(),
                              "Invalid dim index {0} for order value with {1} results", dim.ind(),
                              orderValue.getNumResults());
            const auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(orderValue.getResult(dim.ind()));
            if (dimExpr == nullptr) {
                break;
            }
            inputDim = Dim(dimExpr.getPosition());
        } else if (auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(currentOp)) {
            const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());
            int64_t inputDimInd = -1;
            for (size_t mappedInputDimInd = 0; mappedInputDimInd < dimMapping.size(); ++mappedInputDimInd) {
                const auto& mappedOutputDims = dimMapping[mappedInputDimInd];
                if (std::find(mappedOutputDims.begin(), mappedOutputDims.end(), dim.ind()) != mappedOutputDims.end()) {
                    inputDimInd = checked_cast<int64_t>(mappedInputDimInd);
                    break;
                }
            }
            if (inputDimInd == -1) {
                break;
            }
            inputDim = Dim(inputDimInd);
        } else if (mlir::isa_and_present<IE::AddOp, IE::MultiplyOp>(currentOp)) {
            const auto weightsShape = getShape(currentOp->getOperand(1));
            const auto outputShape = getShape(currentOp->getResult(0));
            if (weightsShape.totalSize() != 1) {
                if (weightsShape.size() <= checked_cast<size_t>(dim.ind()) ||
                    (weightsShape[dim] != 1 && weightsShape[dim] != outputShape[dim])) {
                    break;
                }
            }
        } else if (auto fcOp = mlir::dyn_cast_if_present<IE::FullyConnectedOp>(currentOp)) {
            if (fcOp.getBias() != nullptr) {
                break;
            }

            const auto outShape = getShape(fcOp.getResult());
            if (outShape.size() != 2) {
                break;
            }

            if (dim.ind() != 1) {
                break;
            }
            // For FullyConnectedOp, unrolling output dim 1 maps to slicing weights dim 0 during path rebuilding.
            // `isSupportedOpOnQKVPath` already guarantees constant weights, so tracing stops at this boundary and
            // avoids generating slice copies for the shared activation input.
            inputDim = Dim(0);
        }

        ops.push_back({currentOp, inputDim});
        if (mlir::isa<IE::FullyConnectedOp>(currentOp)) {
            break;
        }
        dim = inputDim;
        currentOp = currentOp->getOperand(0).getDefiningOp();
    }
}

//
// SDPA Pattern Detection
// Currently support three SDPA patterns:
//
// Pattern 1 (Regular SDPA):
// InputQ --------------> MatMul ---> Add ---> Softmax ---> MatMul ---> Output
//                           ^         ^                       ^
//                           |         |                       |
// InputK ----> Multiply -----         |                       |
//                                     |                       |
// InputMask ---------------------------                       |
//                                                             |
// InputV ------------------------------------------------------
//
// Pattern 2 (SDPA with sink input: Concat/Slice around Softmax):
// InputQ --------------> MatMul ---> Add ---> Concat ---> Softmax ---> Slice ---> MatMul ---> Output
//                           ^         ^         ^                                     ^
//                           |         |         |                                     |
// InputK ----> Multiply -----         |         |                                     |
//                                     |         |                                     |
// InputMask ---------------------------         |                                     |
//                                               |                                     |
// ConcatInput -----------------------------------                                     |
//                                                                                     |
// InputV -----------------------------------------------------------------------------|
//
// Pattern 3 (SDPA with Transpose on Q/K/V paths):
// Input --> ops --> Transpose(Q) --> MatMul --> Softmax --> MatMul --> Output
//                                   ^                    ^
//   |                               |                    |
//   |                               |                    |
//   +-----> ops --> Transpose(K) ---|                    |
//   |                                                    | |
//   +-----> ops --> Transpose(V) ------------------------|
//

bool UnrollSDPAPattern::detectSDPAPattern(IE::SoftMaxOp softmaxOp, SDPAPattern& pattern) const {
    // Validate SoftMax
    if (!softmaxOp) {
        return false;
    }

    // Backward - check if softmax input is Concat (optional) or Add
    auto softmaxInput = softmaxOp.getInput();
    if (!softmaxInput || !softmaxInput.hasOneUse()) {
        return false;
    }

    auto softMaxInShape = getShape(softmaxInput);
    // Only support 4D input
    if (softMaxInShape.size() != 4) {
        return false;
    }

    IE::ConcatOp concatOp = nullptr;
    IE::AddOp addOp = nullptr;

    // Check if there's a Concat before Softmax
    if (auto concat = mlir::dyn_cast_or_null<IE::ConcatOp>(softmaxInput.getDefiningOp())) {
        concatOp = concat;
        // Concat must have exactly 2 inputs
        auto concatInputs = concat.getInputs();
        if (concatInputs.size() != 2) {
            return false;
        }

        auto addInputCount = llvm::count_if(concatInputs, [](mlir::Value input) {
            return mlir::isa_and_nonnull<IE::AddOp>(input.getDefiningOp());
        });
        if (addInputCount != 1) {
            return false;
        }

        // Find which input comes from Add (don't assume order)
        for (auto input : concatInputs) {
            if (!input.hasOneUse()) {
                continue;
            }
            if (auto maybeAddInput = mlir::dyn_cast_or_null<IE::AddOp>(input.getDefiningOp())) {
                addOp = maybeAddInput;
                break;
            }
        }
    } else {
        // No Concat, directly get Add from softmax input
        addOp = mlir::dyn_cast_or_null<IE::AddOp>(softmaxInput.getDefiningOp());
    }

    IE::MatMulOp matMulOp = nullptr;
    if (addOp == nullptr) {
        matMulOp = mlir::dyn_cast_or_null<IE::MatMulOp>(softmaxInput.getDefiningOp());
    } else {
        // Backward - check if add input[0] is MatMul1
        auto addInputs = addOp.getInputs();
        if (addInputs.size() < 2) {
            return false;
        }
        auto addInput0 = addInputs[0];
        if (!addInput0 || !addInput0.hasOneUse()) {
            return false;
        }
        matMulOp = mlir::dyn_cast_or_null<IE::MatMulOp>(addInput0.getDefiningOp());
    }

    if (matMulOp == nullptr) {
        return false;
    }

    // Forward - check if softmax output is Slice (optional) or MatMul2
    auto softmaxOutput = softmaxOp.getOutput();
    if (!softmaxOutput || !softmaxOutput.hasOneUse()) {
        return false;
    }

    IE::SliceOp sliceOp = nullptr;
    mlir::Value matMulVInput = softmaxOutput;

    // Check if there's a Slice after Softmax
    if (auto slice = mlir::dyn_cast_or_null<IE::SliceOp>(*softmaxOutput.getUsers().begin())) {
        sliceOp = slice;
        auto sliceOutput = slice.getResult();
        if (!sliceOutput || !sliceOutput.hasOneUse()) {
            return false;
        }
        matMulVInput = sliceOutput;
    }

    auto matMulV = mlir::dyn_cast_or_null<IE::MatMulOp>(*matMulVInput.getUsers().begin());
    if (!matMulV) {
        return false;
    }

    // Pattern 2: Concat and Slice must appear together
    if ((concatOp && !sliceOp) || (!concatOp && sliceOp)) {
        return false;
    }

    // Pattern 3: Track back the q,k,v paths to see if there are Transpose ops
    trackBackQKVPaths(matMulOp.getInput1(), pattern.opsOnPathQ);
    trackBackQKVPaths(matMulOp.getInput2(), pattern.opsOnPathK);
    trackBackQKVPaths(matMulV.getInput2(), pattern.opsOnPathV);

    // SDPA pattern detected successfully!
    pattern.matMulOp = matMulOp;
    pattern.addOp = addOp;
    pattern.concatOp = concatOp;
    pattern.softmaxOp = softmaxOp;
    pattern.sliceOp = sliceOp;
    pattern.matMulV = matMulV;

    return true;
}

bool UnrollSDPAPattern::isUnrollingBeneficial(SDPAPattern& pattern) const {
    auto matMulOp = pattern.matMulOp;
    const auto seqLen = getShape(matMulOp.getInput1())[Dims4D::Act::H];
    auto alignedChannelOpInterface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(matMulOp.getOperation());
    VPUX_THROW_UNLESS(alignedChannelOpInterface != nullptr,
                      "MatMulOp {0} does not implement AlignedChannelsOpInterface", matMulOp);
    auto inputChannelAlignment = alignedChannelOpInterface.getInputChannelAlignment();
    if (seqLen % inputChannelAlignment != 0) {
        // The DMA arising from expanding channels might affect scheduling. Although ExpandAttention exists to handle
        // this situation, it specifically matches IE.Attention, which is not supported on older platforms.
        return false;
    }

    auto softmax = pattern.softmaxOp;
    if (pattern.concatOp != nullptr && pattern.sliceOp != nullptr) {
        const auto softmaxShape = getShape(softmax.getOutput());
        if (softmaxShape[Dim(softmaxShape.size() - 2)] == 1) {
            // If the second-to-last dimension is 1, it's a decoding case where unrolling is not beneficial
            return false;
        }
        // For patterns with Concat and Slice, unrolling, it is not beneficial to unroll the ops on Q/K/V paths
        pattern.opsOnPathQ.clear();
        pattern.opsOnPathK.clear();
        pattern.opsOnPathV.clear();
        return true;
    }

    // Unrolling is beneficial when the two MatMuls in SDPA have different shrinking behavior.
    // If one MatMul gets shrunk while the other doesn't, they end up with different group counts, breaking pipeline
    // efficiency. Unrolling ensures consistent grouping.
    const auto hasDifferentMatMulShrink =
            IE::shouldShrinkMatmulGroups(pattern.matMulOp) != IE::shouldShrinkMatmulGroups(pattern.matMulV);

    // Unrolling is beneficial for getting rid of the Transposes in paths.
    auto addOp = pattern.addOp;
    auto isBeneficialToUnrollOpsOnQKVPath = [&](const SmallVector<std::pair<mlir::Operation*, Dim>>& opsOnPathQ,
                                                const SmallVector<std::pair<mlir::Operation*, Dim>>& opsOnPathK,
                                                const SmallVector<std::pair<mlir::Operation*, Dim>>& opsOnPathV) {
        if (addOp != nullptr || opsOnPathQ.empty() || opsOnPathK.empty() || opsOnPathV.empty()) {
            return false;
        }

        auto lastOpOnQPath = opsOnPathQ.back().first;
        auto lastOpOnKPath = opsOnPathK.back().first;
        auto lastOpOnVPath = opsOnPathV.back().first;
        // If Q/K/V don't share one root buffer, unrolling is not beneficial.
        if (lastOpOnQPath->getOperand(0) != lastOpOnKPath->getOperand(0) ||
            lastOpOnQPath->getOperand(0) != lastOpOnVPath->getOperand(0)) {
            return false;
        }

        // if the last op is a FullyConnected with weights as constant, it will not generate slice copies
        // because the unrolled dim is mapped to the first dim of weights, which is constant.
        auto generateSliceCopy = [&] {
            return !mlir::isa_and_present<IE::FullyConnectedOp>(lastOpOnQPath) ||
                   !mlir::isa_and_present<IE::FullyConnectedOp>(lastOpOnKPath) ||
                   !mlir::isa_and_present<IE::FullyConnectedOp>(lastOpOnVPath);
        }();
        if (generateSliceCopy) {
            return false;
        }

        auto hasRemovableTransposeOnPath = [](const SmallVector<std::pair<mlir::Operation*, Dim>>& opsOnPath) {
            if (opsOnPath.empty()) {
                return false;
            }

            // Only support the Transpose is at the beginning of the path because it is easier to catch the unrolling
            // dim
            auto transposeOp = mlir::dyn_cast_if_present<IE::TransposeOp>(opsOnPath.front().first);
            if (transposeOp == nullptr) {
                return false;
            }

            auto inShape = getShape(transposeOp.getInput());
            auto outShape = getShape(transposeOp.getOutput());
            if (inShape.size() != 4 || outShape.size() != 4) {
                return false;
            }

            // Q/K/V hold the layout [batch, num_heads, seq_len, head_dim] and it is unrolled along the num_heads
            // dimension
            auto inShapeVec = SmallVector<int64_t>(inShape.raw().begin(), inShape.raw().end());
            const auto inUnrolledDim = opsOnPath.front().second;
            // Set the unrolled dimension to 1 for checking trivial permute. For example,
            // AffineMap [d0, d1, d2, d3] -> [d0, d2, d1, d3],
            // after dim d1 is unrolled, the permute becomes trivial: [d0, 1, d2, d3] -> [d0, d2, 1, d3]
            inShapeVec[inUnrolledDim.ind()] = 1;
            auto newInShape = ShapeRef(inShapeVec);
            auto newInMemShape = MemShape(newInShape);
            const auto affineMap = transposeOp.getOrderValue().value();
            if (vpux::isTrivialPermute(newInMemShape, affineMap)) {
                return true;
            }

            // It is also beneficial if there is a op that support ODU permute
            if (opsOnPath.size() < 2) {
                return false;
            }

            auto currentOp = opsOnPath[1].first;
            while (mlir::isa_and_present<IE::AffineReshapeOp>(currentOp)) {
                currentOp = currentOp->getOperand(0).getDefiningOp();
            }

            return mlir::isa<IE::LayerWithPermuteInterface, IE::FullyConnectedOp>(currentOp);
        };
        if (!hasRemovableTransposeOnPath(opsOnPathQ) && !hasRemovableTransposeOnPath(opsOnPathK) &&
            !hasRemovableTransposeOnPath(opsOnPathV)) {
            return false;
        }

        // Estimate the required CMX size for each op; Unrolling can be treated as tiling along the C-axis. If the tile
        // size still exceeds the threshold after unrolling, further tiling along the H-axis may occur, transforming it
        // into a two-axis tiling scenario and thereby breaking input buffer sharing.
        auto checkRequiredCMXSize = [&](const SmallVector<std::pair<mlir::Operation*, Dim>>& opsOnPath) {
            auto numHeads = getShape(matMulOp.getInput1())[Dims4D::Act::C];
            auto module = softmax->getParentOfType<mlir::ModuleOp>();
            const int64_t numClusters = config::getTileExecutor(module).getCount();
            // Ensure the pipelining can work after unrolling.
            const auto thresholdCMXSize = VPU::getTotalCMXSize(softmax).count() * numClusters * 0.5;
            auto getRequiredCMXSize = [&](mlir::Operation* op) {
                auto requiredCMXSize = Byte(0);
                for (auto operand : op->getOperands().drop_front()) {
                    auto operandType = mlir::cast<NDTypeInterface>(operand.getType());
                    auto operandElemType = operandType.getElementType();
                    auto allocSize = operandType.getTotalAllocSize();
                    // Assuming that the precision has been adjusted from FP32 to FP16 via the AdjustPrecisionPipeline.
                    requiredCMXSize += operandElemType.isF32() ? allocSize / 2 : allocSize;
                }
                for (auto result : op->getResults()) {
                    auto resultType = mlir::cast<NDTypeInterface>(result.getType());
                    auto resultElemType = resultType.getElementType();
                    auto allocSize = resultType.getTotalAllocSize();
                    requiredCMXSize += resultElemType.isF32() ? allocSize / 2 : allocSize;
                }
                requiredCMXSize = requiredCMXSize / numHeads;

                // Since unrolling is treated as tiling along the C-axis, it requires a complete input buffer.
                auto firstOperandType = mlir::cast<NDTypeInterface>(op->getOperand(0).getType());
                auto elemType = firstOperandType.getElementType();
                auto allocSize = firstOperandType.getTotalAllocSize();
                requiredCMXSize += elemType.isF32() ? allocSize / 2 : allocSize;

                return requiredCMXSize.count();
            };
            for (auto op : opsOnPath) {
                if (mlir::isa<IE::ViewLikeOpInterface, IE::TransposeOp>(op.first)) {
                    continue;
                }

                if (getRequiredCMXSize(op.first) > thresholdCMXSize) {
                    return false;
                }
            }

            return true;
        };

        return checkRequiredCMXSize(opsOnPathQ) && checkRequiredCMXSize(opsOnPathK) && checkRequiredCMXSize(opsOnPathV);
    };
    if (isBeneficialToUnrollOpsOnQKVPath(pattern.opsOnPathQ, pattern.opsOnPathK, pattern.opsOnPathV)) {
        return true;
    }

    pattern.opsOnPathQ.clear();
    pattern.opsOnPathK.clear();
    pattern.opsOnPathV.clear();
    return addOp != nullptr && hasDifferentMatMulShrink;
}

static SmallVector<mlir::Value> splitTensor(mlir::Value input, int64_t batch, Dim channelDim,
                                            mlir::PatternRewriter& rewriter, mlir::Location origLoc,
                                            const std::string& prefix) {
    SmallVector<mlir::Value> results;
    auto inputType = mlir::cast<NDTypeInterface>(input.getType());
    auto inputShape = inputType.getShape().raw();

    if (batch == 1) {
        return {input};
    }

    // split tensor if batch > 1
    assert(batch > 0 && "Batch must be positive");
    for (int64_t i = 0; i < batch; i++) {
        Shape sliceOffsets = Shape(inputShape.size(), 0);
        sliceOffsets[channelDim] = checked_cast<int64_t>(i);

        Shape sliceSizes = inputShape;
        sliceSizes[channelDim] = 1;
        auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(origLoc, "{0}_slice_{1}", prefix, i), input,
                                                    getIntArrayAttr(rewriter.getContext(), sliceOffsets),
                                                    getIntArrayAttr(rewriter.getContext(), sliceSizes));
        results.push_back(sliceOp.getOutput());
    }

    return results;
}

static SmallVector<mlir::Value> rebuildOpsOnQKVPath(mlir::PatternRewriter& rewriter,
                                                    const SmallVector<std::pair<mlir::Operation*, Dim>>& opsOnPath,
                                                    int64_t batchSize, const std::string& prefix) {
    const auto ctx = rewriter.getContext();

    auto cloneFQ = [&](IE::FakeQuantizeOp fqOp, mlir::Value input, int64_t offset, int64_t size) -> mlir::Value {
        auto createConstParams = [&](mlir::Value parameter, mlir::Location loc) -> mlir::Value {
            auto constOp = parameter.getDefiningOp<Const::DeclareOp>();
            VPUX_THROW_UNLESS(constOp != nullptr, "Expected parameter to be a constant");
            if (getShape(parameter).totalSize() == 1) {
                const Shape targetShape({1});
                const auto newContentAttr = constOp.transformContentAttr().reshape(targetShape).get();
                return rewriter.create<Const::DeclareOp>(loc, newContentAttr.getType(), std::move(newContentAttr))
                        .getOutput();
            }

            auto paramShape = getShape(parameter);
            auto nonOneDims = vpux::getNonOneDim(paramShape);
            VPUX_THROW_UNLESS(nonOneDims.size() == 1, "Expected parameter to have only one non-one dimension, got {0}",
                              nonOneDims.size());
            auto sliceDim = nonOneDims[0];
            auto sliceOffset = SmallVector<int64_t>(paramShape.size(), 0);
            sliceOffset[sliceDim.ind()] = offset * size;
            auto sliceSize = SmallVector<int64_t>(paramShape.begin(), paramShape.end());
            sliceSize[sliceDim.ind()] = size;
            const auto newContentAttr =
                    constOp.transformContentAttr().subview(ShapeRef(sliceOffset), ShapeRef(sliceSize)).get();
            return rewriter.create<Const::DeclareOp>(loc, newContentAttr.getType(), std::move(newContentAttr))
                    .getOutput();
        };

        auto inputLow = createConstParams(fqOp.getInputLow(), appendLoc(fqOp.getLoc(), "input_low"));
        auto inputHigh = createConstParams(fqOp.getInputHigh(), appendLoc(fqOp.getLoc(), "input_high"));
        auto outputLow = createConstParams(fqOp.getOutputLow(), appendLoc(fqOp.getLoc(), "output_low"));
        auto outputHigh = createConstParams(fqOp.getOutputHigh(), appendLoc(fqOp.getLoc(), "output_high"));

        return rewriter
                .create<IE::FakeQuantizeOp>(appendLoc(fqOp.getLoc(), "_{}", offset), input, inputLow, inputHigh,
                                            outputLow, outputHigh, fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(),
                                            fqOp.getAutoBroadcastAttr())
                .getOutput();
    };

    SmallVector<mlir::Value> rebuiltValues;
    VPUX_THROW_UNLESS(!opsOnPath.empty(), "Ops on Q/K/V path is empty, cannot rebuild");
    auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(opsOnPath.back().first);
    VPUX_THROW_UNLESS(fcOp != nullptr, "Last op on Q/K/V path must be a FullyConnectedOp");

    auto input = fcOp.getWeights();
    auto unrolledDim = opsOnPath.back().second;
    auto inputShape = getShape(input);
    VPUX_THROW_UNLESS(inputShape.size() > static_cast<size_t>(unrolledDim.ind()),
                      "Input shape rank {0} is not greater than unrolled dim {1}", inputShape.size(),
                      unrolledDim.ind());

    auto unrolledDimSize = inputShape[unrolledDim];
    VPUX_THROW_UNLESS(unrolledDimSize % batchSize == 0, "Unrolled dim size {0} is not divisible by batch size {1}",
                      unrolledDimSize, batchSize);

    for (int64_t index = 0; index < batchSize; ++index) {
        // Create input slice
        auto sliceOffsetsVec = SmallVector<int64_t>(inputShape.size(), 0);
        sliceOffsetsVec[unrolledDim.ind()] = index * (unrolledDimSize / batchSize);
        auto sliceSizesVec = SmallVector<int64_t>(inputShape.raw().begin(), inputShape.raw().end());
        sliceSizesVec[unrolledDim.ind()] = unrolledDimSize / batchSize;

        mlir::Value slicedWeights;
        if (auto fq = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(input.getDefiningOp())) {
            slicedWeights = rewriter.createOrFold<IE::SliceOp>(appendLoc(input.getLoc(), "{0}_{1}", prefix, index),
                                                               fq.getInput(), getIntArrayAttr(ctx, sliceOffsetsVec),
                                                               getIntArrayAttr(ctx, sliceSizesVec));
            slicedWeights = cloneFQ(fq, slicedWeights, index, unrolledDimSize / batchSize);
        } else {
            slicedWeights = rewriter.createOrFold<IE::SliceOp>(appendLoc(input.getLoc(), "{0}_{1}", prefix, index),
                                                               input, getIntArrayAttr(ctx, sliceOffsetsVec),
                                                               getIntArrayAttr(ctx, sliceSizesVec));
        }

        mlir::Value outputVal =
                rewriter.create<IE::FullyConnectedOp>(appendLoc(fcOp.getLoc(), "{0}_{1}", prefix, index),
                                                      fcOp.getInput(), slicedWeights, fcOp.getBias())
                        .getOutput();

        for (auto opsOnPathIt = opsOnPath.rbegin() + 1; opsOnPathIt != opsOnPath.rend(); ++opsOnPathIt) {
            const auto [op, dim] = *opsOnPathIt;
            if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(op)) {
                outputVal =
                        rewriter.create<IE::TransposeOp>(appendLoc(transposeOp.getLoc(), "{0}_{1}", prefix, index),
                                                         outputVal, /*order*/ nullptr, transposeOp.getOrderValueAttr())
                                .getOutput();
            } else if (mlir::isa_and_present<IE::FakeQuantizeOp>(op)) {
                mlir::IRMapping mapper;
                auto operands = llvm::to_vector(op->getOperands());
                operands[0] = outputVal;
                mapper.map(op->getOperands(), operands);
                auto* newOp = rewriter.clone(*op, mapper);
                inferReturnTypes(newOp, InferShapedTypeMode::SHAPE);
                newOp->setLoc(appendLoc(op->getLoc(), "{0}_{1}", prefix, index));
                outputVal = newOp->getResult(0);
            } else if (auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(op)) {
                auto inShape = getShape(outputVal);
                auto newOutShape = vpux::IE::computeShapeValueFromAffineReshape(affineReshapeOp, inShape);
                VPUX_THROW_UNLESS(newOutShape.has_value(), "Failed to compute output shape for AffineReshapeOp");
                auto newOutShapeAttr = getIntArrayAttr(rewriter.getContext(), newOutShape.value());
                outputVal =
                        rewriter.create<IE::AffineReshapeOp>(
                                        appendLoc(affineReshapeOp.getLoc(), "{0}_affine_reshape_{1}", prefix, index),
                                        outputVal, affineReshapeOp.getDimMapping(), newOutShapeAttr)
                                .getOutput();
            } else if (mlir::isa_and_present<IE::AddOp, IE::MultiplyOp>(op)) {
                mlir::IRMapping mapper;
                auto weightsOperand = op->getOperand(1);
                const auto weightsShape = getShape(weightsOperand);
                if (weightsShape.totalSize() != 1) {
                    VPUX_THROW_UNLESS(weightsShape.size() > checked_cast<size_t>(dim.ind()),
                                      "Weights shape rank {0} is not greater than unrolled dim {1}",
                                      weightsShape.size(), dim.ind());
                    VPUX_THROW_UNLESS(weightsShape[dim] % batchSize == 0,
                                      "Weights unrolled dim size {0} is not divisible by batch size {1}",
                                      weightsShape[dim], batchSize);
                    if (weightsShape[dim] != 1) {
                        auto weightsSliceOffsets = SmallVector<int64_t>(weightsShape.size(), 0);
                        weightsSliceOffsets[dim.ind()] = index * (weightsShape[dim] / batchSize);
                        auto weightsSliceSizes =
                                SmallVector<int64_t>(weightsShape.raw().begin(), weightsShape.raw().end());
                        weightsSliceSizes[dim.ind()] = weightsShape[dim] / batchSize;
                        weightsOperand = rewriter.createOrFold<IE::SliceOp>(
                                appendLoc(op->getLoc(), "{0}_weights_slice_{1}", prefix, index), weightsOperand,
                                getIntArrayAttr(rewriter.getContext(), weightsSliceOffsets),
                                getIntArrayAttr(rewriter.getContext(), weightsSliceSizes));
                    }
                }

                SmallVector<mlir::Value, 2> newOperands{outputVal, weightsOperand};
                mapper.map(op->getOperands(), newOperands);
                auto* newOp = rewriter.clone(*op, mapper);
                inferReturnTypes(newOp, InferShapedTypeMode::SHAPE);
                newOp->setLoc(appendLoc(op->getLoc(), "{0}_{1}", prefix, index));
                outputVal = newOp->getResult(0);
            } else {
                VPUX_THROW("Unsupported op {0} on Q/K/V path", op->getName());
            }
        }

        rebuiltValues.push_back(outputVal);
    }

    return rebuiltValues;
}

mlir::LogicalResult UnrollSDPAPattern::unrollAndRearrangePattern(SDPAPattern& pattern,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto loc = pattern.matMulOp->getLoc();

    if (!isUnrollingBeneficial(pattern)) {
        _log.trace("UnrollingSDPA is not beneficial for this pattern.");
        return mlir::failure();
    }

    // Step 1: Get input tensors
    auto matMulOp = const_cast<IE::MatMulOp&>(pattern.matMulOp);
    auto addOp = const_cast<IE::AddOp&>(pattern.addOp);
    auto concatOp = const_cast<IE::ConcatOp&>(pattern.concatOp);
    auto softmaxOp = const_cast<IE::SoftMaxOp&>(pattern.softmaxOp);
    auto sliceOp = const_cast<IE::SliceOp&>(pattern.sliceOp);
    auto matMulV = const_cast<IE::MatMulOp&>(pattern.matMulV);

    auto opsOnPathQ = pattern.opsOnPathQ;
    auto opsOnPathK = pattern.opsOnPathK;
    auto opsOnPathV = pattern.opsOnPathV;

    auto inputQ = matMulOp.getInput1();
    auto inputK = matMulOp.getInput2();
    auto maskValue = addOp == nullptr ? nullptr : addOp.getInputs()[1];
    auto inputV = matMulV.getInput2();

    // compute unroll batch
    auto inputType = mlir::cast<NDTypeInterface>(inputQ.getType());
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 3 && inputShape.size() != 4) {
        _log.trace("Unsupported shape rank for Q MatMul input.");
        return mlir::failure();
    }

    // 3D: [B, H, W]
    // 4D: [1, B, H, W]
    const auto channelDim = Dim(inputShape.size() - 3);
    const int64_t batch = inputShape[channelDim];

    // Ensure batch dimension is static
    if (mlir::ShapedType::isDynamic(batch)) {
        _log.trace("Batch dimension is dynamic, cannot unroll SDPA pattern");
        return mlir::failure();
    }

    // Step 2: get Q/K/V inputs
    SmallVector<mlir::Value> matmulInputQs =
            opsOnPathQ.empty() ? splitTensor(inputQ, batch, channelDim, rewriter, loc, "matmul_inputQ_slice")
                               : rebuildOpsOnQKVPath(rewriter, opsOnPathQ, batch, "matmul_inputQ_slice");
    SmallVector<mlir::Value> matmulInputKs =
            opsOnPathK.empty() ? splitTensor(inputK, batch, channelDim, rewriter, loc, "matmul_inputK_slice")
                               : rebuildOpsOnQKVPath(rewriter, opsOnPathK, batch, "matmul_inputK_slice");
    SmallVector<mlir::Value> vParts = opsOnPathV.empty()
                                              ? splitTensor(inputV, batch, channelDim, rewriter, loc, "v_slice")
                                              : rebuildOpsOnQKVPath(rewriter, opsOnPathV, batch, "v_slice");

    // Step 3: create  MatMul1_split -> [Add_split] -> [Concat_split] -> Softmax_split -> [Slice_split] -> MatMul2_split
    // chain
    SmallVector<mlir::Value> finalResults;

    // Get concat additional input and determine input order if exists
    mlir::Value concatAdditionalInput = nullptr;
    int addInputIndex = -1;  // Track which index the Add output is at
    if (concatOp != nullptr && addOp != nullptr) {
        auto concatInputs = concatOp.getInputs();
        // Concat is guaranteed to have exactly 2 inputs from detection
        VPUX_THROW_UNLESS(concatInputs.size() == 2, "Concat must have exactly 2 inputs");

        // Find the input that is NOT from Add (the additional input) and track Add's position
        for (size_t idx = 0; idx < concatInputs.size(); ++idx) {
            if (concatInputs[idx].getDefiningOp() == addOp.getOperation()) {
                addInputIndex = idx;
            } else {
                concatAdditionalInput = concatInputs[idx];
            }
        }
        VPUX_THROW_UNLESS(concatAdditionalInput, "Failed to find concat additional input");
        VPUX_THROW_UNLESS(addInputIndex >= 0, "Failed to find Add output in concat inputs");
    }

    for (int64_t i = 0; i < batch; ++i) {
        // Matmul1_split operation
        auto matMul1 = cloneMatMulOp(rewriter, matMulOp, matmulInputQs[i], matmulInputKs[i]);
        matMul1->setLoc(appendLoc(loc, "_matmul_1_" + std::to_string(i)));

        // Add_split operation
        mlir::Value softmaxInput = matMul1->getResult(0);
        if (addOp != nullptr) {
            auto addNewOp = rewriter.create<IE::AddOp>(
                    appendLoc(addOp->getLoc(), "_add_" + std::to_string(i)), matMul1->getResult(0), maskValue,
                    /*auto_broadcast=*/
                    IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), IE::AutoBroadcastType::NUMPY), nullptr,
                    nullptr, nullptr, nullptr);
            softmaxInput = addNewOp.getOutput();
        }

        // Concat_split operation (if exists)
        if (concatOp && concatAdditionalInput) {
            // Split the concat additional input if needed
            SmallVector<mlir::Value> concatAdditionalParts = splitTensor(
                    concatAdditionalInput, batch, channelDim, rewriter, concatOp->getLoc(), "concat_additional");

            // Preserve the original input order
            SmallVector<mlir::Value> concatInputsOrdered(2);
            concatInputsOrdered[addInputIndex] = softmaxInput;
            concatInputsOrdered[1 - addInputIndex] = concatAdditionalParts[i];

            auto concatNewOp = rewriter.create<IE::ConcatOp>(
                    appendLoc(concatOp->getLoc(), "_concat_" + std::to_string(i)),
                    mlir::ValueRange(concatInputsOrdered), concatOp.getPerAxisAttr(), concatOp.getStaticOffsetsAttr());
            softmaxInput = concatNewOp.getOutput();
        }

        // Softmax_split operation
        auto softmaxNewOp =
                rewriter.create<IE::SoftMaxOp>(appendLoc(softmaxOp->getLoc(), "_softmax_" + std::to_string(i)),
                                               softmaxInput, softmaxOp.getAxisIndAttr(), softmaxOp.getPadSizeAttr(),
                                               softmaxOp.getDstElemTypeAttr(), softmaxOp.getMaskAwareAttr());

        mlir::Value matMulVInput = softmaxNewOp.getOutput();

        // Slice_split operation (if exists)
        if (sliceOp) {
            // Adjust slice offsets and sizes for the unrolled dimension
            auto origOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr());
            auto origSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr());

            // Update the batch dimension (channelDim) to be 0 offset and size 1
            SmallVector<int64_t> newOffsets(origOffsets.begin(), origOffsets.end());
            SmallVector<int64_t> newSizes(origSizes.begin(), origSizes.end());

            newOffsets[channelDim.ind()] = 0;
            newSizes[channelDim.ind()] = 1;

            auto sliceNewOp = rewriter.create<IE::SliceOp>(appendLoc(sliceOp->getLoc(), "_slice_" + std::to_string(i)),
                                                           softmaxNewOp.getOutput(),
                                                           getIntArrayAttr(rewriter.getContext(), newOffsets),
                                                           getIntArrayAttr(rewriter.getContext(), newSizes));
            matMulVInput = sliceNewOp.getResult();
        }

        // MatMul2_split operation
        auto matMulVOp = cloneMatMulOp(rewriter, matMulV, matMulVInput, vParts[i]);
        matMulVOp->setLoc(appendLoc(matMulV->getLoc(), "_matmul_v_" + std::to_string(i)));

        finalResults.push_back(matMulVOp->getResult(0));
    }
    VPUX_THROW_WHEN(finalResults.empty(), "finalResults should not be empty");

    // Step 4: Concat final results if needed and replace the original matMulV
    mlir::Value finalValue = finalResults.size() != 1
                                     ? rewriter.create<IE::ConcatOp>(takeOpLoc(pattern.matMulV, "slice_gather"),
                                                                     finalResults, channelDim.ind())
                                               .getOutput()
                                     : finalResults.front();

    // Step 5: Replace the original SDPA chain's last output (matMulV)
    rewriter.replaceOp(matMulV, finalValue);

    // Step 6: Delete the original SDPA chain operations
    // After replacing matMulV, the chain matMulOp -> addOp -> [concatOp] -> softmaxOp -> [sliceOp] has no more use
    // Delete in reverse order to avoid breaking dependencies
    if (sliceOp && sliceOp->use_empty()) {
        rewriter.eraseOp(sliceOp);
    }
    if (softmaxOp->use_empty()) {
        rewriter.eraseOp(softmaxOp);
    }
    if (concatOp && concatOp->use_empty()) {
        rewriter.eraseOp(concatOp);
    }
    if (addOp && addOp->use_empty()) {
        rewriter.eraseOp(addOp);
    }
    if (matMulOp->use_empty()) {
        rewriter.eraseOp(matMulOp);
    }
    for (auto op : opsOnPathQ) {
        if (op.first->use_empty()) {
            rewriter.eraseOp(op.first);
        }
    }
    for (auto op : opsOnPathK) {
        if (op.first->use_empty()) {
            rewriter.eraseOp(op.first);
        }
    }
    for (auto op : opsOnPathV) {
        if (op.first->use_empty()) {
            rewriter.eraseOp(op.first);
        }
    }

    _log.trace("Successfully unrolled and rearranged SDPA pattern");
    return mlir::success();
}

mlir::LogicalResult UnrollSDPAPattern::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    auto opLoc = origOp->getLoc();
    _log.debug("Found SoftMaxOp at loc: {0}", opLoc);

    SDPAPattern pattern;
    if (!detectSDPAPattern(origOp, pattern)) {
        _log.debug("SDPA pattern not detected at {0}", opLoc);
        return mlir::failure();
    }

    // Set insertion point before matMulV to ensure all new operations come before it
    rewriter.setInsertionPoint(pattern.matMulV);

    if (mlir::succeeded(unrollAndRearrangePattern(pattern, rewriter))) {
        _log.info("Successfully unrolled SDPA pattern at {0}", opLoc);
        return mlir::success();
    }

    return mlir::failure();
}

//
// UnrollSDPAPatternPass Implementation
//

class UnrollSDPAPatternPass final : public IE::impl::UnrollSDPAPatternBase<UnrollSDPAPatternPass> {
public:
    explicit UnrollSDPAPatternPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSDPAPatternPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnrollSDPAPattern>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createUnrollSDPAPatternPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollSDPAPatternPass(Logger log) {
    return std::make_unique<UnrollSDPAPatternPass>(log);
}
