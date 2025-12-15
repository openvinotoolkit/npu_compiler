//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_ADDEXPLICITPADDINGBEFORENCEPERMUTE
#define GEN_PASS_DEF_ADDEXPLICITPADDINGBEFORENCEPERMUTE
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

bool isSliceOnChannels(mlir::Operation* userOp) {
    auto sliceOp = mlir::dyn_cast_or_null<VPU::SliceOp>(userOp);
    if (sliceOp == nullptr) {
        return false;
    }
    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(sliceOp.getSource().getType()).getShape();
    const auto offsets = Shape(parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsetsAttr()));
    const auto sizes = Shape(parseIntArrayAttr<int64_t>(sliceOp.getStaticSizesAttr()));
    return offsets[Dims4D::Act::C] + sizes[Dims4D::Act::C] < inputShape[Dims4D::Act::C];
}

bool userNeedsExplicitPad(mlir::Operation* userOp, int64_t ncePermuteInputChannels) {
    if (userOp == nullptr) {
        return false;
    }

    // If there is a Slice op for Channels, this will slice the expanded op
    // to the original channel value and will avoid NaN propagation
    if (isSliceOnChannels(userOp)) {
        return false;
    }

    // If NCE operation has weights sparsity, expanded activation won't be used in actual compute
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(userOp);
    const auto hasSparseWeights = nceOp != nullptr && nceOp.getWeightsOperand() != nullptr &&
                                  mlir::isa<vpux::VPU::SparseTensorType>(nceOp.getWeightsOperand().getType());
    if (hasSparseWeights) {
        return false;
    }

    // Following ops do not have compute on channels and if the user op is Slice on channel or if they produce sliced
    // data directly, NaNs will not be propagated through the model
    if (mlir::isa<VPU::NCEMaxPoolOp, VPU::NCEDepthConvolutionOp, VPU::NCEAveragePoolOp, VPU::NCEEltwiseOp>(userOp)) {
        auto resultShape = mlir::cast<NDTypeInterface>(userOp->getResult(0).getType()).getShape();
        const auto producesSlicedChannels = resultShape[Dims4D::Act::C] <= ncePermuteInputChannels;
        if (producesSlicedChannels) {
            return false;
        }
        return llvm::any_of(userOp->getResult(0).getUsers(), [&](mlir::Operation* nextUserOp) {
            return userNeedsExplicitPad(nextUserOp, ncePermuteInputChannels);
        });
    } else if (mlir::isa<VPU::ViewLikeOpInterface>(userOp)) {
        return llvm::any_of(userOp->getResult(0).getUsers(), [&](mlir::Operation* nextUserOp) {
            return userNeedsExplicitPad(nextUserOp, ncePermuteInputChannels);
        });
    }

    return true;
}

// Method used to find cases where expand done with NCEPermute
// can propagate NaN values and affect accuracy.
bool isExplicitPadNeeded(VPU::NCEPermuteOp origOp) {
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto dstElemAttr = outputType.getElementType();
    const auto expandedChannels = origOp.getExpandedChannels();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputChannels = inputType.getShape()[Dims4D::Act::C];

    // Explicit Padding must be introduced only if output element type of NCE Permute is FP16
    // and output channels are greater than input channels.
    if (!dstElemAttr.isF16() || expandedChannels == inputChannels) {
        return false;
    }

    for (auto userOp : origOp.getResult().getUsers()) {
        if (userNeedsExplicitPad(userOp, inputChannels)) {
            return true;
        }
    }
    return false;
}

void insertExplicitPad(Logger& log, VPU::NCEPermuteOp origOp) {
    const auto expandedChannels = origOp.getExpandedChannels();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());

    log.trace("Insert explicit padding for operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    auto permuteInShape = inputType.getShape();
    const SmallVector<int64_t> padShape = {permuteInShape[Dims4D::Act::N],
                                           expandedChannels - permuteInShape[Dims4D::Act::C],
                                           permuteInShape[Dims4D::Act::H], permuteInShape[Dims4D::Act::W]};
    const auto elemType = inputType.getElementType();
    const auto padType = mlir::RankedTensorType::get(padShape, elemType);
    // create zero const for padding
    auto padData = Const::createZerosConst(builder, origOp.getLoc(), padType);
    auto concat = builder.create<VPU::ConcatOp>(origOp.getLoc(), mlir::ValueRange{origOp.getInput(), padData},
                                                Dims4D::Act::C);
    auto newPermuteOp = builder.create<VPU::NCEPermuteOp>(origOp->getLoc(), origOp.getOutput().getType(),
                                                          concat.getOutput(), origOp.getExpandedChannelsAttr(),
                                                          origOp.getDstElemTypeAttr(), origOp.getDstOrderAttr(),
                                                          origOp.getPpeAttr(), origOp.getMultiClusterStrategyAttr());

    origOp.replaceAllUsesWith(newPermuteOp.getOperation());
    origOp->erase();
}

//
// AddExplicitPaddingBeforeNCEPermute
//

class AddExplicitPaddingBeforeNCEPermutePass final :
        public VPU::impl::AddExplicitPaddingBeforeNCEPermuteBase<AddExplicitPaddingBeforeNCEPermutePass> {
public:
    explicit AddExplicitPaddingBeforeNCEPermutePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void AddExplicitPaddingBeforeNCEPermutePass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPU::NCEPermuteOp origOp) {
        if (isExplicitPadNeeded(origOp)) {
            insertExplicitPad(_log, origOp);
        }
    });
}

}  // namespace

//
// createAddExplicitPaddingBeforeNCEPermutePass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAddExplicitPaddingBeforeNCEPermutePass(Logger log) {
    return std::make_unique<AddExplicitPaddingBeforeNCEPermutePass>(log);
}
