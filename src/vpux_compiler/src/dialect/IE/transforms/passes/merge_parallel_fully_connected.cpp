//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MERGEPARALLELFULLYCONNECTED
#define GEN_PASS_DEF_MERGEPARALLELFULLYCONNECTED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int64_t rank3D = 3;

struct FQConstInputs {
    SmallVector<Const::DeclareOp> inputs;
    SmallVector<Const::DeclareOp> inLows;
    SmallVector<Const::DeclareOp> inHighs;
    SmallVector<Const::DeclareOp> outLows;
    SmallVector<Const::DeclareOp> outHighs;
};

// Convert
//       cst_set1(2x3x2,1x1x1,1x1x1,2x1x2,2x1x2)  cst_set2(2x3x3,1x1x1,1x1x1,2x1x3,2x1x3)
//               \     |     |    /     /                   \     |    |     /    /
//                IE.FakeQuantize (2x3x2)                 IE.FakeQuantize (2x3x3)
//                        |                                        |
//                IE.AffineReshape(6x2)                   IE.AffineReshape(6x3)
//                        |                                        |
//                IE.Transpose(2x6)                       IE.Transpose(3x6)
//                       \                                   /
//                        \         Input(1x6)              /
//                         \       /           \           /
//                   IE.FullyConnected(1x2)  IE.FullyConnected(1x3)

// To

//      cst_set_concat(2x3x5,1x1x1,1x1x1,2x1x5,2x1x5)
//                    \      |    |     /     /
//                IE.FakeQuantize (2x3x5)
//                        |
//                IE.AffineReshape(6x5)
//                        |
//                IE.Transpose(5x6)       Input(1x6)
//                        \               /
//                         \             /
//                       IE.FullyConnected(1x5)
//                            |         |
//                IE.SLice(1x2)     IE.slice(1x3)

class MergeParallelFullyConnected final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    MergeParallelFullyConnected(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(log) {
        setDebugName("mergeParallelFullyConnected");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp fullyConnectedOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

IE::ConcatOp concatConst(SmallVector<Const::DeclareOp>& constOps, mlir::PatternRewriter& rewriter) {
    SmallVector<mlir::Value> concats;
    for (auto constOp : constOps) {
        concats.push_back(constOp.getOutput());
    }
    const auto constShape = getShape(constOps.front().getOutput());
    const auto concatDim = constShape.size() - 1;
    auto concat = rewriter.create<IE::ConcatOp>(appendLoc(constOps.front()->getLoc(), "_concat_"),
                                                mlir::ValueRange(concats), Dim(concatDim));
    return concat;
}

bool isLastDimConcatable(ShapeRef shape1, ShapeRef shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }

    for (auto ind : irange(shape1.size())) {
        if (shape1[Dim(ind)] != shape2[Dim(ind)] && ind != (shape1.size() - 1)) {
            return false;
        }
    }

    return true;
};

template <class ConcreteOp>
bool isOutputCompatible(SmallVector<ConcreteOp>& ops) {
    auto refOp = ops.back();
    auto refElemType = mlir::cast<vpux::NDTypeInterface>(refOp->getResult(0).getType()).getElementType();
    auto refShape = getShape(refOp->getResult(0));
    for (auto ind : irange(ops.size() - 1)) {
        auto op = ops[ind];
        if (!isLastDimConcatable(refShape, getShape(op->getResult(0)))) {
            return false;
        }

        auto outElementType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
        if (outElementType != refElemType) {
            return false;
        }
    }
    return true;
}

std::optional<SmallVector<IE::FullyConnectedOp>> getFullyConnectedOpWithSameAttr(mlir::Value parent) {
    SmallVector<IE::FullyConnectedOp> fullyConnectedOps;
    for (auto user : parent.getUsers()) {
        auto fullConnected = mlir::dyn_cast<IE::FullyConnectedOp>(user);
        if (fullConnected == nullptr) {
            return std::nullopt;
        }
        if (fullConnected.getInput() != parent) {
            return std::nullopt;
        }
        fullyConnectedOps.push_back(fullConnected);
    }

    // need at least two fullyConnected
    if (fullyConnectedOps.size() < 2) {
        return std::nullopt;
    }

    if (!isOutputCompatible<IE::FullyConnectedOp>(fullyConnectedOps)) {
        return std::nullopt;
    }

    auto hasAllButLastDimOne = [](ShapeRef shape) {
        const auto isOne = [](auto dim) {
            return dim == 1;
        };
        return std::all_of(shape.begin(), shape.end() - 1, isOne);
    };

    for (auto fullConnected : fullyConnectedOps) {
        auto shape = getShape(fullConnected.getOutput());
        if (!hasAllButLastDimOne(shape)) {
            return std::nullopt;
        }
        // TODO: could support if the bias data is const.
        if (fullConnected.getBias() != nullptr) {
            return std::nullopt;
        }
    }

    // Sorting those ops is required for deterministic compilation as the order of getUsers
    // might be different between compilations depending on preceding passes
    llvm::sort(fullyConnectedOps, [](IE::FullyConnectedOp fc1, IE::FullyConnectedOp fc2) {
        return !fc1->isBeforeInBlock(fc2);
    });

    return fullyConnectedOps;
}

namespace AffineReshapeTranposeOrder {
std::optional<SmallVector<IE::TransposeOp>> getTransposeOpWithSameAttr(
        ArrayRef<IE::FullyConnectedOp> fullyConnectedOps) {
    SmallVector<IE::TransposeOp> transposeOps;
    for (auto fullyConnected : fullyConnectedOps) {
        auto transpose = mlir::dyn_cast<IE::TransposeOp>(fullyConnected.getWeights().getDefiningOp());
        if (transpose == nullptr || !transpose->hasOneUse() || !transpose.getOrderValue().has_value()) {
            return std::nullopt;
        }
        transposeOps.push_back(transpose);
    }

    // should have same order value
    auto refTranspose = transposeOps.back();
    auto refOrderValue = refTranspose.getOrderValue().value();
    for (auto transpose : transposeOps) {
        if (transpose.getOrderValue().value() != refOrderValue) {
            return std::nullopt;
        }
    }

    return transposeOps;
}

std::optional<SmallVector<IE::AffineReshapeOp>> getAffineReshapeOpWithSameAttr(ArrayRef<IE::TransposeOp> transposeOps) {
    SmallVector<IE::AffineReshapeOp> affineReshapeOps;
    for (auto transpose : transposeOps) {
        auto affineReshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(transpose.getInput().getDefiningOp());
        if (affineReshape == nullptr || !affineReshape->hasOneUse()) {
            return std::nullopt;
        }
        affineReshapeOps.push_back(affineReshape);
    }

    // should have the same dim mapping
    auto refAffineReshape = affineReshapeOps.back();
    auto refDimMapping = refAffineReshape.getDimMapping();
    for (auto affineReshape : affineReshapeOps) {
        if (getShape(affineReshape.getOutput()).size() != rank3D - 1 ||
            affineReshape.getDimMapping() != refDimMapping) {
            return std::nullopt;
        }
    }

    return affineReshapeOps;
}
}  // namespace AffineReshapeTranposeOrder

namespace TranposeAffineReshapeOrder {
std::optional<SmallVector<IE::AffineReshapeOp>> getAffineReshapeOpWithSameAttr(ArrayRef<IE::FullyConnectedOp> fcOps) {
    SmallVector<IE::AffineReshapeOp> affineReshapeOps;
    for (auto fc : fcOps) {
        auto affineReshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(fc.getWeights().getDefiningOp());
        if (affineReshape == nullptr || !affineReshape->hasOneUse()) {
            return std::nullopt;
        }
        affineReshapeOps.push_back(affineReshape);
    }

    // should have the same dim mapping
    auto refAffineReshape = affineReshapeOps.back();
    auto refDimMapping = refAffineReshape.getDimMapping();
    for (auto affineReshape : affineReshapeOps) {
        if (getShape(affineReshape.getOutput()).size() != rank3D - 1 ||
            affineReshape.getDimMapping() != refDimMapping) {
            return std::nullopt;
        }
    }

    return affineReshapeOps;
}

std::optional<SmallVector<IE::TransposeOp>> getTransposeOpWithSameAttr(ArrayRef<IE::AffineReshapeOp> affineReshapeOps) {
    SmallVector<IE::TransposeOp> transposeOps;
    for (auto reshape : affineReshapeOps) {
        auto transpose = mlir::dyn_cast_if_present<IE::TransposeOp>(reshape.getInput().getDefiningOp());
        if (transpose == nullptr || !transpose->hasOneUse() || !transpose.getOrderValue().has_value()) {
            return std::nullopt;
        }
        transposeOps.push_back(transpose);
    }

    // should have same order value
    auto refTranspose = transposeOps.back();
    auto refOrderValue = refTranspose.getOrderValue().value();
    for (auto transpose : transposeOps) {
        if (transpose.getOrderValue().value() != refOrderValue) {
            return std::nullopt;
        }
    }

    return transposeOps;
}
}  // namespace TranposeAffineReshapeOrder

FQConstInputs getFakeQuantizeConstInputs(SmallVector<IE::FakeQuantizeOp>& fakeQuantizeOps) {
    FQConstInputs fQConstInputs;
    for (auto fakeQuantize : fakeQuantizeOps) {
        auto input = fakeQuantize.getInput().getDefiningOp<Const::DeclareOp>();
        auto inLow = fakeQuantize.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHigh = fakeQuantize.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLow = fakeQuantize.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHigh = fakeQuantize.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        fQConstInputs.inputs.push_back(input);
        fQConstInputs.inLows.push_back(inLow);
        fQConstInputs.inHighs.push_back(inHigh);
        fQConstInputs.outLows.push_back(outLow);
        fQConstInputs.outHighs.push_back(outHigh);
    }
    return fQConstInputs;
}

bool doesFQHaveSameZeroPoint(SmallVector<IE::FakeQuantizeOp> fakeQuantizeOps) {
    SmallVector<int64_t> zeroPoints;
    for (auto fqOp : fakeQuantizeOps) {
        auto lowConstantOp = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto highConstantOp = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

        if (lowConstantOp == nullptr || highConstantOp == nullptr) {
            return false;
        }

        auto outputScalesAndZeroPoints = getScalesAndZeroPointsFromContentAttr(
                lowConstantOp.getContentAttr(), highConstantOp.getContentAttr(), fqOp.getAutoBroadcast(),
                fqOp.getLevels(), fqOp.getLowFpType(), /*isSigned=*/false);
        if (mlir::failed(outputScalesAndZeroPoints)) {
            return false;
        }
        const auto& outZeroPoints = std::get<1>(outputScalesAndZeroPoints.value());

        if (!std::equal(outZeroPoints.begin() + 1, outZeroPoints.end(), outZeroPoints.begin())) {
            return false;
        }
        zeroPoints.push_back(outZeroPoints.front());
    }

    return std::equal(zeroPoints.begin() + 1, zeroPoints.end(), zeroPoints.begin());
}

template <class ConcreteOp>
std::optional<SmallVector<IE::FakeQuantizeOp>> getFakeQuantizeOpWithSameAttr(ArrayRef<ConcreteOp> concreteOps) {
    SmallVector<IE::FakeQuantizeOp> fakeQuantizeOps;
    for (auto concreteOp : concreteOps) {
        if (!concreteOp->hasOneUse()) {
            return std::nullopt;
        }
        auto fakeQuantize = mlir::dyn_cast<IE::FakeQuantizeOp>(concreteOp.getInput().getDefiningOp());
        if (fakeQuantize == nullptr) {
            return std::nullopt;
        }
        fakeQuantizeOps.push_back(fakeQuantize);
    }

    if (!isOutputCompatible<IE::FakeQuantizeOp>(fakeQuantizeOps)) {
        return std::nullopt;
    }

    auto refFakeQuantizeOp = fakeQuantizeOps.front();
    auto refBroadcast = refFakeQuantizeOp.getAutoBroadcast();

    for (auto fakeQuantize : fakeQuantizeOps) {
        if (refBroadcast != fakeQuantize.getAutoBroadcast()) {
            return std::nullopt;
        }
        for (auto ind : irange(fakeQuantizeOps.front()->getOperands().size())) {
            auto constOp = mlir::dyn_cast<Const::DeclareOp>(fakeQuantize->getOperand(ind).getDefiningOp());
            if (constOp == nullptr) {
                return std::nullopt;
            }
        }
    }

    auto fQConstInputs = getFakeQuantizeConstInputs(fakeQuantizeOps);
    SmallVector<float> inLowData;
    SmallVector<float> inHighData;
    for (auto constInput : fQConstInputs.inLows) {
        if (!IE::isBaseContentSplat(constInput)) {
            return std::nullopt;
        }
        auto data = IE::getConst(constInput).front();
        inLowData.push_back(data);
    }
    if (!std::equal(inLowData.begin() + 1, inLowData.end(), inLowData.begin())) {
        return std::nullopt;
    }

    for (auto constInput : fQConstInputs.inHighs) {
        if (!IE::isBaseContentSplat(constInput)) {
            return std::nullopt;
        }
        auto data = IE::getConst(constInput).front();
        inHighData.push_back(data);
    }
    if (!std::equal(inHighData.begin() + 1, inHighData.end(), inHighData.begin())) {
        return std::nullopt;
    }

    auto isGPTQCase = [](ShapeRef shape) {
        const auto greaterThanOne = [](auto dim) {
            return dim > 1;
        };
        if (shape.size() != rank3D) {
            return false;
        }
        if (llvm::count_if(shape.raw(), greaterThanOne) == rank3D - 1) {
            return true;
        }
        return false;
    };

    for (auto constInput : fQConstInputs.outLows) {
        if (!isGPTQCase(getShape(constInput.getOutput()))) {
            return std::nullopt;
        }
    }
    for (auto constInput : fQConstInputs.outHighs) {
        if (!isGPTQCase(getShape(constInput.getOutput()))) {
            return std::nullopt;
        }
    }

    if (!doesFQHaveSameZeroPoint(fakeQuantizeOps)) {
        return std::nullopt;
    }

    return fakeQuantizeOps;
}

IE::FakeQuantizeOp createFakeQuantize(SmallVector<IE::FakeQuantizeOp>& fakeQuantizeOps,
                                      mlir::PatternRewriter& rewriter) {
    auto fqConstInputs = getFakeQuantizeConstInputs(fakeQuantizeOps);
    auto concatInConst = concatConst(fqConstInputs.inputs, rewriter);
    auto concatOutLowConst = concatConst(fqConstInputs.outLows, rewriter);
    auto concatOutHighConst = concatConst(fqConstInputs.outHighs, rewriter);

    auto refOp = fakeQuantizeOps.front();
    auto newFq = rewriter.create<IE::FakeQuantizeOp>(appendLoc(refOp->getLoc(), "_concat_"), concatInConst.getOutput(),
                                                     refOp.getInputLow(), refOp.getInputHigh(),
                                                     concatOutLowConst.getOutput(), concatOutHighConst.getOutput(),
                                                     refOp.getLevelsAttr(), nullptr, refOp.getAutoBroadcastAttr());
    return newFq;
}

mlir::FailureOr<IE::FullyConnectedOp> mergeFCForReshapeTransposeOrder(ArrayRef<IE::FullyConnectedOp> fullyConnectedOps,
                                                                      IE::FullyConnectedOp origOp,
                                                                      mlir::PatternRewriter& rewriter) {
    auto validTransposeOps = AffineReshapeTranposeOrder::getTransposeOpWithSameAttr(fullyConnectedOps);
    if (!validTransposeOps.has_value()) {
        return matchFailed(rewriter, origOp, "Invalid transpose operations");
    }
    auto transposeOps = validTransposeOps.value();

    auto validAffineReshapeOps = AffineReshapeTranposeOrder::getAffineReshapeOpWithSameAttr(transposeOps);
    if (!validAffineReshapeOps.has_value()) {
        return matchFailed(rewriter, origOp, "Invalid affineReshape operations");
    }
    auto affineReshapeOps = validAffineReshapeOps.value();

    auto validFakeQuantizeOps = getFakeQuantizeOpWithSameAttr<IE::AffineReshapeOp>(affineReshapeOps);
    if (!validFakeQuantizeOps.has_value()) {
        return matchFailed(rewriter, origOp, "Invalid fake quantize operations");
    }
    auto fakeQuantizeOps = validFakeQuantizeOps.value();

    auto newFakeQuantize = createFakeQuantize(fakeQuantizeOps, rewriter);

    const auto newFakeQuantizeOutShape = getShape(newFakeQuantize.getOutput());
    SmallVector<int64_t> reshapeOut{getShape(affineReshapeOps.front().getOutput()).raw().front(),
                                    newFakeQuantizeOutShape.raw().back()};
    const auto reshapeOutAttr = getIntArrayAttr(origOp->getContext(), reshapeOut);
    auto newAffineReshape = rewriter.create<IE::AffineReshapeOp>(
            appendLoc(affineReshapeOps.front()->getLoc(), "_concat_"), newFakeQuantize.getOutput(),
            affineReshapeOps.front().getDimMapping(), reshapeOutAttr);

    auto newTranspose = rewriter.create<IE::TransposeOp>(appendLoc(transposeOps.front()->getLoc(), "_concat_"),
                                                         newAffineReshape.getOutput(), nullptr,
                                                         transposeOps.front().getOrderValueAttr());
    auto newFullyConnected = rewriter.create<IE::FullyConnectedOp>(
            appendLoc(origOp->getLoc(), "_concat_"), origOp.getInput(), newTranspose.getOutput(), origOp.getBias());

    return newFullyConnected;
}

mlir::FailureOr<IE::FullyConnectedOp> mergeFCForTransposeReshapeOrder(ArrayRef<IE::FullyConnectedOp> fullyConnectedOps,
                                                                      IE::FullyConnectedOp origOp,
                                                                      mlir::PatternRewriter& rewriter) {
    auto validAffineReshapeOps = TranposeAffineReshapeOrder::getAffineReshapeOpWithSameAttr(fullyConnectedOps);
    if (!validAffineReshapeOps.has_value()) {
        return matchFailed(rewriter, origOp, "Invalid affineReshape operations");
    }
    auto affineReshapeOps = validAffineReshapeOps.value();

    auto validTransposeOps = TranposeAffineReshapeOrder::getTransposeOpWithSameAttr(affineReshapeOps);
    if (!validTransposeOps.has_value()) {
        return matchFailed(rewriter, origOp, "Invalid transpose operations");
    }
    auto transposeOps = validTransposeOps.value();
    auto validFakeQuantizeOps = getFakeQuantizeOpWithSameAttr<IE::TransposeOp>(transposeOps);
    if (!validFakeQuantizeOps.has_value()) {
        return matchFailed(rewriter, origOp, "Invalid fake quantize operations");
    }
    auto fakeQuantizeOps = validFakeQuantizeOps.value();

    auto newFakeQuantize = createFakeQuantize(fakeQuantizeOps, rewriter);

    auto newTranspose = rewriter.create<IE::TransposeOp>(appendLoc(transposeOps.front()->getLoc(), "_concat_"),
                                                         newFakeQuantize.getOutput(), nullptr,
                                                         transposeOps.front().getOrderValueAttr());

    const auto newTransposeOutShape = getShape(newTranspose.getOutput());
    SmallVector<int64_t> reshapeOut{newTransposeOutShape.raw().front(),
                                    getShape(affineReshapeOps.front().getOutput()).raw().back()};
    const auto reshapeOutAttr = getIntArrayAttr(origOp->getContext(), reshapeOut);
    auto newAffineReshape = rewriter.create<IE::AffineReshapeOp>(
            appendLoc(affineReshapeOps.front()->getLoc(), "_concat_"), newTranspose.getOutput(),
            affineReshapeOps.front().getDimMapping(), reshapeOutAttr);

    auto newFullyConnected = rewriter.create<IE::FullyConnectedOp>(
            appendLoc(origOp->getLoc(), "_concat_"), origOp.getInput(), newAffineReshape.getOutput(), origOp.getBias());

    return newFullyConnected;
}

mlir::LogicalResult MergeParallelFullyConnected::matchAndRewrite(IE::FullyConnectedOp fullyConnectedOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.debug("[{0}] Got FullyConnected layer at '{1}'", fullyConnectedOp->getName(), fullyConnectedOp->getLoc());
    auto nestedLog = _log.nest();

    auto validFullyConnectedOps = getFullyConnectedOpWithSameAttr(fullyConnectedOp.getInput());
    if (!validFullyConnectedOps.has_value()) {
        return matchFailed(rewriter, fullyConnectedOp, "Invalid fullyConnected operations");
    }
    auto fullyConnectedOps = validFullyConnectedOps.value();

    auto parentIsTransposeOp = [](IE::FullyConnectedOp fcOp) {
        auto lhsOp = fcOp.getWeights().getDefiningOp();
        return mlir::dyn_cast_if_present<IE::TransposeOp>(lhsOp) != nullptr;
    };
    auto parentIsAffineReshapeOp = [](IE::FullyConnectedOp fcOp) {
        auto lhsOp = fcOp.getWeights().getDefiningOp();
        return mlir::dyn_cast_if_present<IE::AffineReshapeOp>(lhsOp) != nullptr;
    };
    auto maybeReshapeTranspose = llvm::all_of(fullyConnectedOps, parentIsTransposeOp);
    auto maybeTransposeReshape = llvm::all_of(fullyConnectedOps, parentIsAffineReshapeOp);
    if (!maybeReshapeTranspose && !maybeTransposeReshape) {
        nestedLog.debug("At least one parent is neither AffineReshape, nor Transpose");
        return mlir::failure();
    }

    mlir::FailureOr<IE::FullyConnectedOp> mergedFC;
    // FQ - AffineReshape - Transpose - FullyConnet
    if (maybeReshapeTranspose) {
        mergedFC = mergeFCForReshapeTransposeOrder(fullyConnectedOps, fullyConnectedOp, rewriter);
    } else {
        mergedFC = mergeFCForTransposeReshapeOrder(fullyConnectedOps, fullyConnectedOp, rewriter);
    }

    if (mlir::failed(mergedFC)) {
        nestedLog.debug("Failed to merge parallel FullyConnected ops");
        return mlir::failure();
    }

    nestedLog.trace("New merged FullyConnected op = {0}", mergedFC.value());

    int64_t offset = 0;
    SmallVector<IE::SliceOp> slices;

    // Create Slice ops for each original FullyConnected op
    for (auto p : fullyConnectedOps | indexed) {
        auto fullyConnected = p.value();
        auto shape = getShape(fullyConnected.getOutput());
        Shape offsets(shape.size());
        offsets[Dim(shape.size() - 1)] = offset;
        offset += shape[Dim(shape.size() - 1)];

        nestedLog.trace("Slice output of FullyConnected ops, idx = {0}, offsets = {1}", p.index(), offsets);
        auto slice = rewriter.create<IE::SliceOp>(
                appendLoc(fullyConnected->getLoc(), "_slice_{0}", p.index()), mergedFC.value().getOutput(),
                getIntArrayAttr(rewriter.getContext(), offsets), getIntArrayAttr(rewriter.getContext(), shape.raw()));

        slices.push_back(slice);
    }

    // Replace the original FullyConnected ops
    for (auto p : fullyConnectedOps | indexed) {
        auto fullyConnected = p.value();
        auto slice = slices[p.index()];

        nestedLog.trace("Replace FC = {0}, with Slice output {1}", fullyConnected, slice);
        rewriter.replaceOp(fullyConnected, slice->getResult(0));
    }

    _log.debug("Merge parallel fully connected operation successful");
    return mlir::success();
}

//
// MergeParallelFullyConnectedPass
//

class MergeParallelFullyConnectedPass final :
        public IE::impl::MergeParallelFullyConnectedBase<MergeParallelFullyConnectedPass> {
public:
    explicit MergeParallelFullyConnectedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MergeParallelFullyConnectedPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeParallelFullyConnected>(&ctx, _log);
    IE::ConcatOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getOperation();
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
//  createMergeParallelFullyConnectedPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMergeParallelFullyConnectedPass(Logger log) {
    return std::make_unique<MergeParallelFullyConnectedPass>(log);
}
