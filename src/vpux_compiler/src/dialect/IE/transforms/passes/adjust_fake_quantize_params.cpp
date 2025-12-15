//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ADJUSTFAKEQUANTIZEPARAMS
#define GEN_PASS_DEF_ADJUSTFAKEQUANTIZEPARAMS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

const float FP16_MAX = checked_cast<float>(std::numeric_limits<vpux::type::float16>::max());
const float FP16_MIN = checked_cast<float>(std::numeric_limits<vpux::type::float16>::lowest());

namespace {

//
// FQParamsRewriter
//
class FQParamsRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    FQParamsRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp multiplyOp, mlir::PatternRewriter& rewriter) const final;

private:
    struct Metadata {
        float inputScale;
        float outputScale;
        bool isStartNode;
    };

    bool hasExceededFp16Range(float low, float high) const {
        return ((high > FP16_MAX) || (low < FP16_MIN));
    };

    bool checkForSupportedOperations(mlir::Operation* op) const {
        return op && mlir::isa<IE::MultiplyOp, IE::ReduceMeanOp, IE::FakeQuantizeOp>(op);
    };

    bool isFqRangeOutOfBounds(IE::FakeQuantizeOp fqOp, float inScale, float outScale) const;

    float getScale(mlir::Operation* operation, float scale, bool backPropagate) const;
    float getScale(IE::FakeQuantizeOp fakeQuantOp, float inScale = 1.0f, float outScale = 1.0f) const;

    mlir::LogicalResult traverseAndScaleSubgraph(llvm::SmallVector<mlir::Operation*>& subgraph,
                                                 llvm::DenseMap<mlir::Operation*, Metadata>& opsMetadata) const;
    mlir::LogicalResult scaleInputOperand(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                                          llvm::DenseMap<mlir::Operation*, Metadata>& opsMetadata) const;

    Logger _log;
};

std::tuple<float, float, float, float> getFqValues(IE::FakeQuantizeOp fq) {
    return std::make_tuple(IE::getConst(fq.getInputLow().getDefiningOp<Const::DeclareOp>())[0],
                           IE::getConst(fq.getInputHigh().getDefiningOp<Const::DeclareOp>())[0],
                           IE::getConst(fq.getOutputLow().getDefiningOp<Const::DeclareOp>())[0],
                           IE::getConst(fq.getOutputHigh().getDefiningOp<Const::DeclareOp>())[0]);
}

bool FQParamsRewriter::isFqRangeOutOfBounds(IE::FakeQuantizeOp fqOp, float inScale = 1.0f,
                                            float outScale = 1.0f) const {
    auto [inLow, inHigh, outLow, outHigh] = getFqValues(fqOp);
    return (hasExceededFp16Range(inLow * inScale, inHigh * inScale) ||
            hasExceededFp16Range(outLow * outScale, outHigh * outScale));
}

auto createMultiplyOp(mlir::Operation* op, mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, float scale) {
    rewriter.setInsertionPointAfter(op);

    auto tensorType = mlir::RankedTensorType::get({1}, mlir::Float32Type::get(ctx));
    const auto newScaleConst = Const::createFloatConst(rewriter, op->getLoc(), tensorType, {scale});
    auto multiplyOp = rewriter.create<IE::MultiplyOp>(takeOpLoc(op, "as_mul"), op->getResult(0).getType(),
                                                      op->getResult(0), newScaleConst, IE::AutoBroadcastType::NUMPY,
                                                      /*post_op=*/nullptr,
                                                      /*clamp=*/nullptr,
                                                      /*output_channels=*/nullptr,
                                                      /*input_channels=*/nullptr);
    return multiplyOp;
};

float FQParamsRewriter::getScale(IE::FakeQuantizeOp fakeQuantOp, float inScale, float outScale) const {
    float scale = 1.0f;
    auto getScaleForFqRange = [&](float low, float high) {
        double suggestedScale = 1.0f;
        auto maxVal = std::max(std::abs(low), std::abs(high));
        if (maxVal > FP16_MAX) {
            int p = std::ceil(std::log10(maxVal / FP16_MAX));
            suggestedScale = 1.0f / std::pow(10, p);
            return suggestedScale;
        }

        return suggestedScale;
    };

    auto [inLow, inHigh, outLow, outHigh] = getFqValues(fakeQuantOp);
    auto inputScale = getScaleForFqRange(inLow * inScale, inHigh * inScale);
    auto outputScale = getScaleForFqRange(outLow * outScale, outHigh * outScale);

    if (inputScale <= 1.0f && outputScale <= 1.0f) {
        scale = std::min(inputScale, outputScale);
    } else {
        scale = std::max(inputScale, outputScale);
    }

    return scale;
}

float FQParamsRewriter::getScale(mlir::Operation* operation, float scale, bool backPropagate = false) const {
    auto isSquareOp = [](IE::MultiplyOp op) {
        return (op.getInput1() == op.getInput2());
    };

    if (auto mulOp = mlir::dyn_cast_or_null<IE::MultiplyOp>(operation)) {
        if (isSquareOp(mulOp)) {
            return backPropagate ? std::sqrt(scale) : (scale * scale);
        }
    }

    return scale;
}

/*
 *  Starting with the parent of FQ operation whose range exceeds FP16 bounds, traverse the graph
 *  till we reach FQ layer with range in FP16 bounds and capture all the nodes.
 *  Once a scale is applied on Starting node of subgraph, value of scale propagates down the
 *  subgraph till we reach the end of subgraph.
 */
mlir::LogicalResult FQParamsRewriter::traverseAndScaleSubgraph(
        llvm::SmallVector<mlir::Operation*>& subgraph, llvm::DenseMap<mlir::Operation*, Metadata>& opsMetadata) const {
    llvm::SetVector<mlir::Operation*> operationsToProcess;
    llvm::DenseSet<mlir::Operation*> visitedOps;

    operationsToProcess.insert(subgraph.begin(), subgraph.end());

    auto getOpUsers = [&](mlir::Operation* op) {
        llvm::SmallVector<mlir::Operation*> userVec;
        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                userVec.push_back(user);
            }
        }
        return userVec;
    };

    llvm::SmallVector<mlir::Operation*> outputVec;
    while (!operationsToProcess.empty()) {
        auto currentOp = operationsToProcess.back();
        if (!checkForSupportedOperations(currentOp)) {
            return mlir::failure();
        }

        if (visitedOps.contains(currentOp)) {
            operationsToProcess.pop_back();
            if (!opsMetadata[currentOp].isStartNode) {
                subgraph.push_back(currentOp);
            }
            continue;
        }
        visitedOps.insert(currentOp);

        if (!opsMetadata[currentOp].isStartNode) {
            opsMetadata[currentOp].outputScale = getScale(currentOp, opsMetadata[currentOp].inputScale);
        }

        // User of currentOp can be a FQ or a non FQ operation
        // If FQ is user of current op, Check if FQ rang eis in FP16 bounds which marks the end of subgraph
        // else validate that FQ layer with new scale adjusted range is in FP16 range.
        for (auto user : currentOp->getResult(0).getUsers()) {
            if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(user)) {
                if (!isFqRangeOutOfBounds(fqOp)) {
                    opsMetadata[currentOp].outputScale = 1.0f;
                    continue;
                } else if (!opsMetadata[currentOp].isStartNode &&
                           isFqRangeOutOfBounds(fqOp, opsMetadata[currentOp].inputScale,
                                                opsMetadata[currentOp].outputScale)) {
                    // Requires backpropogation of a new scale across the nodes we visited so far
                    return mlir::failure();
                }
            }

            outputVec.push_back(user);
        }

        while (!outputVec.empty()) {
            auto outputOp = outputVec.pop_back_val();
            if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(outputOp)) {
                auto fqOutputUsers = getOpUsers(fqOp);
                outputVec.append(fqOutputUsers.begin(), fqOutputUsers.end());
                continue;
            }

            if (!mlir::isa<mlir::func::ReturnOp>(outputOp)) {
                opsMetadata[outputOp].inputScale = opsMetadata[currentOp].outputScale;
                operationsToProcess.insert(outputOp);
            } else {
                opsMetadata[currentOp].outputScale = 1.0f;
            }
        }
    }

    std::reverse(subgraph.begin(), subgraph.end());
    return mlir::success();
}

mlir::LogicalResult FQParamsRewriter::scaleInputOperand(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                                                        llvm::DenseMap<mlir::Operation*, Metadata>& opsMetadata) const {
    auto hasSingleUser = [&](mlir::Operation* op) {
        llvm::SetVector<mlir::Operation*> userSet;
        for (auto user : op->getUsers()) {
            userSet.insert(user);
        }

        if (userSet.size() == 1) {
            return true;
        }

        return false;
    };

    auto updateFqParams = [&](IE::FakeQuantizeOp origFq, float inScale, float outScale) {
        rewriter.setInsertionPoint(origFq);
        auto [inLow, inHigh, outLow, outHigh] = getFqValues(origFq);

        auto newInputLo =
                Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getInputLow().getType(), inLow * inScale);
        auto newInputHi =
                Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getInputHigh().getType(), inHigh * inScale);
        auto newOutputLo =
                Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getOutputLow().getType(), outLow * outScale);
        auto newOutputHi = Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getOutputHigh().getType(),
                                                   outHigh * outScale);

        return rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
                origFq, origFq.getInput(), newInputLo, newInputHi, newOutputLo, newOutputHi, origFq.getLevelsAttr(),
                origFq.getLowFpTypeAttr(), origFq.getAutoBroadcastAttr());
    };

    llvm::SetVector<mlir::Operation*> inputs;
    for (auto operand : op->getOperands()) {
        inputs.insert(operand.getDefiningOp());
    }

    auto handleEltwiseAddSub = [&](mlir::Operation* op) {
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
            auto input = op->getOperand(i).getDefiningOp();
            if (!opsMetadata.contains(input)) {
                if (auto cstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(op->getOperand(i).getDefiningOp())) {
                    auto cstType = mlir::cast<vpux::NDTypeInterface>(cstOp.getOutput().getType());
                    auto newContentAttr = cstOp.getContentAttr().transform().rescale(opsMetadata[op].inputScale).get();
                    mlir::OpBuilder builder(cstOp);
                    auto newCstOp =
                            builder.create<vpux::Const::DeclareOp>(cstOp.getLoc(), cstType, std::move(newContentAttr));
                    op->setOperand(i, newCstOp);
                } else {
                    auto mulOp = createMultiplyOp(op->getOperand(i).getDefiningOp(), rewriter, getContext(),
                                                  opsMetadata[op].inputScale);
                    op->setOperand(i, mulOp);
                }
            }
        }

        return mlir::success();
    };

    return llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<IE::MultiplyOp>([&](auto mulOp) {
                // In case of MUL - FQ - ADD , input to MUL Needs to be scaled to bring the
                // output in FP16 range. If MUL OP is in the middle of subgraph, i/p and o/p are
                // scaled
                if (opsMetadata[op].isStartNode) {
                    // SquareOp
                    if (mulOp.getInput1() == mulOp.getInput2()) {
                        if (auto fq = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(inputs[0])) {
                            if (hasSingleUser(inputs[0])) {
                                updateFqParams(fq, 1.0f, opsMetadata[op].inputScale);
                                return mlir::success();
                            }
                        }
                    }

                    auto newMulOp = createMultiplyOp(inputs[0], rewriter, getContext(), opsMetadata[op].inputScale);
                    op->setOperand(0, newMulOp);
                }
                return mlir::success();
            })
            .Case<IE::SubtractOp>([&](auto op) {
                return handleEltwiseAddSub(op.getOperation());
            })
            .Case<IE::AddOp>([&](auto op) {
                return handleEltwiseAddSub(op.getOperation());
            })
            .Default([&](mlir::Operation* op) -> mlir::LogicalResult {
                if (inputs.size() != 1) {
                    return mlir::failure();
                }

                if (auto fq = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(inputs[0])) {
                    if (hasSingleUser(inputs[0])) {
                        updateFqParams(fq, 1.0f, opsMetadata[op].inputScale);
                        return mlir::success();
                    }
                }

                auto newMulOp = createMultiplyOp(inputs[0], rewriter, getContext(), opsMetadata[op].inputScale);
                op->setOperand(0, newMulOp);
                return mlir::success();
            });
}

/*
      data  in_L in_H out_L out_H           data  in_L in_H  out_L * C  out_H * C
        |    |    |     |     |               |    |    |        |          |
        |    |    |     |     |               |    |    |        |          |
        v    v    v     v     v               v    v    v        v          v
      +-------------------------+             +-----------------------------------+
      |       FakeQuantize      |             |            FakeQuantize           |
      +-------------------------+             +-----------------------------------+
                   |                =====>                     |
                   v                                           v
              +----------+                               +----------+
              | Multiply | <--- Const                    | Multiply | <--- Const
              +----+-----+                               +----+-----+
                   |                                          |
                   v                                          v
       +-------------------------+            +-----------------------------------+
       |       FakeQuantize      |            |            FakeQuantize           |
       +-------------------------+            +-----------------------------------+
       data in_L in_H out_L out_H            data  in_L*C in_H*C  out_L*C  out_H*C
        |    |    |     |     |               |     |      |        |         |
        |    |    |     |     |               |     |      |        |         |
        v    v    v     v     v               v     v      v        v         v
             +------------+                              +------------+
             | ReduceMean |                              | ReduceMean |
             +------------+                              +------------+
                   |                                          |
                   v                                          v
       +-------------------------+            +-----------------------------------+
       |       FakeQuantize      |            |            FakeQuantize           |
       +-------------------------+            +-----------------------------------+
       data in_L in_H out_L out_H            data  in_L*C  in_H*C   out_L    out_H
        |    |    |     |     |               |     |       |         |        |
        |    |    |     |     |               |     |       |         |        |
        v    v    v     v     v               v     v       v         v        v

*/
mlir::LogicalResult FQParamsRewriter::matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp,
                                                      mlir::PatternRewriter& rewriter) const {
    auto levels = fakeQuantizeOp.getLevels();

    // Maximum number of levels that don't exceeds I8/U8 storage type . TODO: E#169024 adjust logic for int16 quant
    // levels.
    if (!levels.has_value() || *levels <= QuantizationLevels::QUANT_LEVELS_8BIT) {
        return matchFailed(rewriter, fakeQuantizeOp,
                           "Skipping AdjustFQParams pass for quantization range < i8 {0} at {1}",
                           fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());
    }

    if (!IE::isPerTensorFQ({fakeQuantizeOp}) || !isFqRangeOutOfBounds(fakeQuantizeOp)) {
        return matchFailed(rewriter, fakeQuantizeOp, "Skipping AdjustFQParams pass as FQ {0} at {1} is in range",
                           fakeQuantizeOp->getName(), fakeQuantizeOp->getLoc());
    }

    auto parentOp = fakeQuantizeOp.getOperand(0).getDefiningOp();
    if (!checkForSupportedOperations(parentOp)) {
        return matchFailed(rewriter, fakeQuantizeOp, "Encountered unsupported operation {0}",
                           fakeQuantizeOp->getName());
    }

    llvm::SmallVector<mlir::Operation*> subgraph;
    llvm::DenseMap<mlir::Operation*, Metadata> subgraphMetaData;
    subgraphMetaData[parentOp].outputScale = getScale(fakeQuantizeOp, 1.0f, 1.0f);
    subgraphMetaData[parentOp].inputScale = getScale(parentOp, subgraphMetaData[parentOp].outputScale, true);
    subgraphMetaData[parentOp].isStartNode = true;
    subgraph.push_back(parentOp);

    if (mlir::failed(traverseAndScaleSubgraph(subgraph, subgraphMetaData))) {
        return mlir::failure();
    }

    auto updateFqParams = [&](IE::FakeQuantizeOp origFq, float inScale, float outScale) {
        rewriter.setInsertionPoint(origFq);
        auto [inLow, inHigh, outLow, outHigh] = getFqValues(origFq);

        auto newInLo =
                Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getInputLow().getType(), inLow * inScale);
        auto newInHi =
                Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getInputHigh().getType(), inHigh * inScale);
        auto newOutLo =
                Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getOutputLow().getType(), outLow * outScale);
        auto newOutHi = Const::createFloatConst(rewriter, origFq->getLoc(), origFq.getOutputHigh().getType(),
                                                outHigh * outScale);

        return rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
                origFq, origFq.getInput(), newInLo, newInHi, newOutLo, newOutHi, origFq.getLevelsAttr(),
                origFq.getLowFpTypeAttr(), origFq.getAutoBroadcastAttr());
    };

    for (auto subgraphOp : subgraph) {
        auto metadata = subgraphMetaData[subgraphOp];

        // Restricting the pass to handle few ops based on current models (ex Model C4).
        // Some of the ops require special handling of inputs. For example, Add / Subtract op requires both inputs to be
        // multiplied with same scale. For ops like convolution, bias values needs to be adjusted to get the right
        // output.
        if (subgraphMetaData[subgraphOp].isStartNode || (mlir::isa<IE::AddOp, IE::SubtractOp>(subgraphOp))) {
            if (mlir::failed(scaleInputOperand(subgraphOp, rewriter, subgraphMetaData))) {
                _log.warning("Unsupported Op found in subgraph");
                return mlir::failure();
            }
        }

        for (auto user : llvm::make_early_inc_range(subgraphOp->getResult(0).getUsers())) {
            if (auto fqOut = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(user)) {
                updateFqParams(fqOut,
                               subgraphMetaData[subgraphOp].isStartNode ? metadata.outputScale : metadata.inputScale,
                               metadata.outputScale);
            } else {
                if (metadata.outputScale == 1.0f) {
                    createMultiplyOp(subgraphOp, rewriter, getContext(), 1.0f / metadata.inputScale);
                }
            }
        }
    }

    return mlir::success();
}

//
// AdjustFakeQuantizeParams
//

class AdjustFakeQuantizeParamsPass final : public IE::impl::AdjustFakeQuantizeParamsBase<AdjustFakeQuantizeParamsPass> {
public:
    explicit AdjustFakeQuantizeParamsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AdjustFakeQuantizeParamsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FQParamsRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustFakeQuantizeParamsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustFakeQuantizeParamsPass(Logger log) {
    return std::make_unique<AdjustFakeQuantizeParamsPass>(log);
}
