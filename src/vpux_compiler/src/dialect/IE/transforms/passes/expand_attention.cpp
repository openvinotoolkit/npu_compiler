//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/attention_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/locations.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/SmallPtrSet.h>

#include <algorithm>
#include <limits>
#include <tuple>

namespace vpux::IE {
#define GEN_PASS_DECL_EXPANDATTENTION
#define GEN_PASS_DEF_EXPANDATTENTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

struct PathOp {
    mlir::Operation* op = nullptr;
    Dim inputDim;
    Dim outputDim;
};

struct PathInput {
    mlir::Value value;
    Dim dim;
};

struct PatternOps {
    SmallVector<PathOp> rootOps;
    SmallVector<PathOp> opsOnPathQ;
    SmallVector<PathOp> opsOnPathK;
    SmallVector<PathOp> opsOnPathV;
};

int64_t getAlignment(IE::AttentionOp attentionOp) {
    auto inputQType = mlir::cast<vpux::NDTypeInterface>(attentionOp.getInputQ().getType());
    return VPU::NCEInvariant::getAlignment(inputQType.getElementType());
}

bool isTransposedV(const ShapeRef& qShape, const ShapeRef& vShape) {
    return qShape[Dims4D::Act::H] == vShape[Dims4D::Act::W] && qShape[Dims4D::Act::W] == vShape[Dims4D::Act::H];
}

int64_t getPaddingSize(IE::AttentionOp attentionOp) {
    const auto qShape = getShape(attentionOp.getInputQ());
    const auto H = qShape[Dims4D::Act::H];
    // The sequence length is mapped to the input and output channels of the
    // Convolution after MatMul lowering which requires channel alignment.
    const auto alignment = getAlignment(attentionOp);
    return alignValUp(H, alignment) - H;
}

bool doesAttentionNeedExpansion(IE::AttentionOp attentionOp) {
    if (attentionOp.getInputSink() != nullptr || attentionOp.getInputMask() != nullptr ||
        attentionOp.getPadSizeS().has_value()) {
        return false;
    }

    auto hasExpectedSplatVal = [](mlir::Value value, float expectedVal) {
        if (value == nullptr) {
            return true;
        }
        auto constOp = value.getDefiningOp<Const::DeclareOp>();
        if (constOp == nullptr) {
            return false;
        }
        auto content = constOp.getContent();
        return content.isSplat() && content.getSplatValue<float>() == expectedVal;
    };

    auto attentionScale = attentionOp.getInputScale();
    if (attentionScale != nullptr) {
        auto scaleConst = attentionScale.getDefiningOp<Const::DeclareOp>();
        if (scaleConst == nullptr || !scaleConst.getContent().isSplat()) {
            return false;
        }
    }

    auto biasConst = attentionOp.getInputBias();
    if (!hasExpectedSplatVal(biasConst, 0.0f)) {
        return false;
    }

    const auto qShape = getShape(attentionOp.getInputQ());
    const auto kShape = getShape(attentionOp.getInputK());
    const auto vShape = getShape(attentionOp.getInputV());
    if (qShape.size() != 4 || kShape.size() != 4 || vShape.size() != 4) {
        return false;
    }

    if (qShape != kShape || (qShape != vShape && !isTransposedV(qShape, vShape))) {
        return false;
    }

    const auto qHeadSize = qShape[Dim(0)] * qShape[Dim(1)];
    const auto tSL = qShape[Dim(2)];
    const auto sSL = kShape[Dim(2)];
    auto matches = [&](const IE::AttentionConfig& config) {
        return (config.qHeadSize == qHeadSize) && (config.sSL == sSL) && (config.tSL == tSL);
    };
    // For those configurations that are in the LEGAL_ATTENTION_CONFIGS, the Attention will not be decomposed.
    // Therefore, expansion is not needed.
    if (llvm::any_of(IE::LEGAL_ATTENTION_CONFIGS, matches)) {
        return false;
    }

    return getPaddingSize(attentionOp) > 0;
}

// Walk through supported ops and find common rootOp for matrix Q,K,V.
void trackBackRootOp(IE::AttentionOp attentionOp, PatternOps& patternOps) {
    // Only support the Reshape operation that preserves the highest non-trivial dimension, which is the sequence length
    // because dim mapping makes it difficult to track the root dim
    auto getReshapeInputDim = [](mlir::Operation* op, Dim outDim) -> std::optional<Dim> {
        auto inNDType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
        auto outNDType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
        const auto inShape = inNDType.getShape();
        const auto inOrder = inNDType.getDimsOrder();
        const auto outShape = outNDType.getShape();
        const auto outOrder = outNDType.getDimsOrder();
        const auto inHighestNonTrivialDim = getHighestNonTrivialDim(inShape, inOrder);
        const auto outHighestNonTrivialDim = getHighestNonTrivialDim(outShape, outOrder);
        if (!inHighestNonTrivialDim.has_value() || !outHighestNonTrivialDim.has_value()) {
            return std::nullopt;
        }

        if (outHighestNonTrivialDim.value() != outDim || inShape[inHighestNonTrivialDim.value()] != outShape[outDim]) {
            return std::nullopt;
        }

        return inHighestNonTrivialDim.value();
    };

    auto getPathInput = [&](mlir::Operation* op, Dim outputDim,
                            bool allowMultiUses = false) -> std::optional<PathInput> {
        if (op == nullptr || (!allowMultiUses && !op->hasOneUse())) {
            return std::nullopt;
        }

        if (mlir::isa_and_present<IE::AffineReshapeOp, IE::ReshapeOp>(op)) {
            auto inputDim = getReshapeInputDim(op, outputDim);
            if (!inputDim.has_value()) {
                return std::nullopt;
            }
            return PathInput{op->getOperand(0), inputDim.value()};
        }

        if (auto transposeOp = mlir::dyn_cast_if_present<IE::TransposeOp>(op)) {
            if (!transposeOp.getOrderValue().has_value()) {
                return std::nullopt;
            }
            const auto dimExpr =
                    mlir::dyn_cast<mlir::AffineDimExpr>(transposeOp.getOrderValue().value().getResult(outputDim.ind()));
            if (dimExpr == nullptr) {
                return std::nullopt;
            }
            return PathInput{transposeOp.getInput(), Dim(dimExpr.getPosition())};
        }

        if (mlir::isa_and_present<IE::FullyConnectedOp>(op)) {
            // Temporarily we don't take weights padding into account.
            if (outputDim.ind() != 0 || !vpux::Const::isConstValue(op->getOperand(1))) {
                return std::nullopt;
            }

            return PathInput{op->getOperand(0), outputDim};
        }

        if (mlir::isa_and_present<IE::AddOp, IE::MultiplyOp>(op)) {
            // Temporarily we don't take weights padding into account.
            const auto lhsOperand = op->getOperand(0);
            const auto lhsOperandShape = getShape(lhsOperand);
            const auto rhsOperand = op->getOperand(1);
            const auto rhsOperandShape = getShape(rhsOperand);
            if (lhsOperandShape.size() != rhsOperandShape.size()) {
                return std::nullopt;
            }

            if (rhsOperandShape[outputDim] != 1 || !vpux::Const::isConstValue(rhsOperand)) {
                return std::nullopt;
            }

            return PathInput{lhsOperand, outputDim};
        }

        if (auto fqOp = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(op)) {
            if (!IE::isPerTensorFQ({fqOp})) {
                return std::nullopt;
            }

            return PathInput{fqOp.getInput(), outputDim};
        }

        return std::nullopt;
    };

    auto walkThroughSupportedOps = [&](mlir::Value value, SmallVector<PathOp>& opsOnPath, Dim& rootDim, Dim startDim,
                                       bool allowMultiUses = false) {
        rootDim = startDim;
        auto currentOp = value.getDefiningOp();
        while (auto input = getPathInput(currentOp, rootDim, allowMultiUses)) {
            opsOnPath.push_back(PathOp{currentOp, input->dim, rootDim});
            rootDim = input->dim;
            currentOp = input->value.getDefiningOp();
        }
    };

    Dim rootDimQ, rootDimK, rootDimV;
    const auto qShape = getShape(attentionOp.getInputQ());
    const auto vShape = getShape(attentionOp.getInputV());
    const auto vStartDim = isTransposedV(qShape, vShape) ? Dims4D::Act::W : Dims4D::Act::H;
    walkThroughSupportedOps(attentionOp.getInputQ(), patternOps.opsOnPathQ, rootDimQ, Dims4D::Act::H);
    walkThroughSupportedOps(attentionOp.getInputK(), patternOps.opsOnPathK, rootDimK, Dims4D::Act::H);
    walkThroughSupportedOps(attentionOp.getInputV(), patternOps.opsOnPathV, rootDimV, vStartDim);
    if (patternOps.opsOnPathQ.empty() || patternOps.opsOnPathK.empty() || patternOps.opsOnPathV.empty()) {
        return;
    }

    auto rootOpQ = patternOps.opsOnPathQ.back().op->getOperand(0).getDefiningOp();
    auto rootOpK = patternOps.opsOnPathK.back().op->getOperand(0).getDefiningOp();
    auto rootOpV = patternOps.opsOnPathV.back().op->getOperand(0).getDefiningOp();
    if (rootOpQ == nullptr || rootOpK == nullptr || rootOpV == nullptr) {
        return;
    }

    if (rootOpQ == rootOpK && rootOpQ == rootOpV && rootDimQ == rootDimK && rootDimQ == rootDimV) {
        // Continue tracking back root op for not interrupt VF or copy optimization
        walkThroughSupportedOps(patternOps.opsOnPathQ.back().op->getOperand(0), patternOps.rootOps, rootDimQ, rootDimQ,
                                /*allowMultiUses*/ true);
    }

    return;
}

bool isBeneficialForExpandAttention(IE::AttentionOp attentionOp) {
    // A slight regression occurs with the 1T configuration because inserting a concatOp before the rootOp might break
    // the VF. Disabling this optimization for 1T config serves as a workaround.
    // Track E#225307
    auto module = attentionOp->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = config::getTileExecutor(module).getCount();
    return numClusters != 1;
}

void expandAttention(IE::AttentionOp attentionOp) {
    PatternOps patternOps;
    trackBackRootOp(attentionOp, patternOps);
    if (patternOps.rootOps.empty()) {
        return;
    }

    if (!isBeneficialForExpandAttention(attentionOp)) {
        return;
    }

    auto rootOps = llvm::to_vector(llvm::reverse(patternOps.rootOps));
    auto opsOnQPath = llvm::to_vector(llvm::reverse(patternOps.opsOnPathQ));
    auto opsOnKPath = llvm::to_vector(llvm::reverse(patternOps.opsOnPathK));
    auto opsOnVPath = llvm::to_vector(llvm::reverse(patternOps.opsOnPathV));

    // Insert the Concat before rootOp and the Slice after Attention.
    mlir::IRRewriter rewriter(attentionOp->getContext());
    const auto padSize = getPaddingSize(attentionOp);
    const auto rootOp = rootOps.front();
    const mlir::Value rootInput = rootOp.op->getOperand(0);
    const auto rootShape = getShape(rootInput);
    SmallVector<int64_t> padShape(rootShape.raw().begin(), rootShape.raw().end());
    padShape[rootOp.inputDim.ind()] = padSize;

    const auto rootType = mlir::cast<vpux::NDTypeInterface>(rootInput.getType());
    const Shape padShapeObj(padShape);
    const auto padType = mlir::cast<mlir::RankedTensorType>(rootType.changeShape(padShapeObj));
    rewriter.setInsertionPoint(rootOp.op);
    const auto zeroPadding =
            Const::createDenseConst(rewriter, appendLoc(rootOp.op->getLoc(), "attention_pad"), padType, 0.0f);
    const SmallVector<mlir::Value> concatInputs{rootInput, zeroPadding};
    auto paddedRoot =
            rewriter.create<IE::ConcatOp>(appendLoc(rootOp.op->getLoc(), "expand_attention_root"), concatInputs,
                                          getIntAttr(rewriter.getContext(), rootOp.inputDim.ind()));

    auto rebuildOp = [&](PathOp pathOp, mlir::Value input, Dim inputPaddedDim) -> std::pair<mlir::Value, Dim> {
        VPUX_THROW_UNLESS(pathOp.inputDim == inputPaddedDim,
                          "Unexpected expanded attention dim '{0}' before '{1}', expected '{2}'", inputPaddedDim,
                          *pathOp.op, pathOp.inputDim);
        const auto outputPaddedDim = pathOp.outputDim;

        const auto origOutShape = getShape(pathOp.op->getResult(0));
        auto newShape = SmallVector<int64_t>(origOutShape.raw().begin(), origOutShape.raw().end());
        newShape[outputPaddedDim.ind()] += padSize;

        mlir::IRMapping mapper;
        auto operands = llvm::to_vector(pathOp.op->getOperands());
        operands[0] = input;

        mapper.map(pathOp.op->getOperands(), operands);
        rewriter.setInsertionPoint(pathOp.op);
        auto* newOp = rewriter.clone(*pathOp.op, mapper);
        if (mlir::isa<IE::AffineReshapeOp, IE::ReshapeOp>(pathOp.op)) {
            newOp->setAttr("shape_value", getIntArrayAttr(rewriter, newShape));
        }
        inferReturnTypes(newOp, InferShapedTypeMode::ALL);
        return {newOp->getResult(0), outputPaddedDim};
    };

    auto rebuildPath = [&](ArrayRef<PathOp> opsOnPath, mlir::Value input,
                           Dim inputPaddedDim) -> std::pair<mlir::Value, Dim> {
        auto newValue = input;
        auto paddedDim = inputPaddedDim;
        for (auto pathOp : opsOnPath) {
            std::tie(newValue, paddedDim) = rebuildOp(pathOp, newValue, paddedDim);
        }

        return {newValue, paddedDim};
    };

    mlir::Value sharedValue = paddedRoot.getOutput();
    auto sharedDim = rootOp.inputDim;
    std::tie(sharedValue, sharedDim) = rebuildPath(rootOps, sharedValue, sharedDim);

    auto rebuildAttentionInput = [&](ArrayRef<PathOp> opsOnPath) -> mlir::Value {
        mlir::Value newValue = sharedValue;
        auto paddedDim = sharedDim;
        std::tie(newValue, paddedDim) = rebuildPath(opsOnPath, newValue, paddedDim);
        return newValue;
    };

    const auto paddedQ = rebuildAttentionInput(opsOnQPath);
    const auto paddedK = rebuildAttentionInput(opsOnKPath);
    const auto paddedV = rebuildAttentionInput(opsOnVPath);

    rewriter.setInsertionPoint(attentionOp);
    auto padSizeAttr = getIntAttr(rewriter.getContext(), padSize);
    auto newAttentionOp = rewriter.create<IE::AttentionOp>(
            appendLoc(attentionOp.getLoc(), "expanded"), paddedQ, paddedK, paddedV, attentionOp.getInputMask(),
            attentionOp.getInputScale(), attentionOp.getInputSink(), attentionOp.getInputBias(), padSizeAttr);

    auto outputVal = newAttentionOp.getOutput();
    auto origShape = getShape(attentionOp.getOutput());
    mlir::Operation* opToReplace = attentionOp;
    if (attentionOp.getOutput().hasOneUse()) {
        auto postTranspose = mlir::dyn_cast_if_present<IE::TransposeOp>(*attentionOp.getOutput().getUsers().begin());
        if (postTranspose != nullptr) {
            opToReplace = postTranspose;
            origShape = getShape(postTranspose.getOutput());
            rewriter.setInsertionPointAfter(newAttentionOp);
            auto newPostTranspose =
                    rewriter.create<IE::TransposeOp>(appendLoc(postTranspose.getLoc(), "expanded"), outputVal,
                                                     postTranspose.getOrder(), postTranspose.getOrderValueAttr());
            outputVal = newPostTranspose.getOutput();
        }
    }

    const auto offsets = SmallVector<int64_t>(origShape.size(), 0);
    rewriter.setInsertionPointAfterValue(outputVal);
    auto sliceOp =
            rewriter.create<IE::SliceOp>(appendLoc(newAttentionOp->getLoc(), "_post_slice"), outputVal,
                                         getIntArrayAttr(rewriter, offsets), getIntArrayAttr(rewriter, origShape));
    rewriter.replaceAllOpUsesWith(opToReplace, sliceOp.getOutput());
    rewriter.eraseOp(opToReplace);
}

class ExpandAttentionPass final : public IE::impl::ExpandAttentionBase<ExpandAttentionPass> {
public:
    explicit ExpandAttentionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandAttentionPass::safeRunOnFunc() {
    auto func = getOperation();

    SmallVector<IE::AttentionOp> attentionOps;
    func->walk([&](IE::AttentionOp attentionOp) {
        if (doesAttentionNeedExpansion(attentionOp)) {
            attentionOps.push_back(attentionOp);
        }
    });

    for (auto attentionOp : attentionOps) {
        expandAttention(attentionOp);
    }
}

}  // namespace

//
// createExpandAttentionPass
//

namespace vpux::IE {

std::unique_ptr<mlir::Pass> createExpandAttentionPass(Logger log) {
    return std::make_unique<ExpandAttentionPass>(log);
}

}  // namespace vpux::IE
