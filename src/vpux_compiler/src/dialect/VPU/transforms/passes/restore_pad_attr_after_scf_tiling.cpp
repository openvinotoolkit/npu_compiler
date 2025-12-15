//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_RESTOREPADATTRAFTERSCFTILING
#define GEN_PASS_DEF_RESTOREPADATTRAFTERSCFTILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// RestorePadAttrAfterSCFTilingPass
//
class RestorePadAttrAfterSCFTilingPass final :
        public VPU::impl::RestorePadAttrAfterSCFTilingBase<RestorePadAttrAfterSCFTilingPass> {
public:
    explicit RestorePadAttrAfterSCFTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void processSCFForOp(mlir::scf::ForOp forOp);
    mlir::Value getInputOperand(mlir::tensor::PadOp padOp, mlir::OpBuilder builder);
};

/**
 * @brief For tensors with dynamic dimensions along padded axes, the PadOp might have identical input and output
 * types. However, for static dimensions, the shapes of PadOp source and destination differ along the padded axes.
 * Add a tensor.cast to make the input dynamic when required, as the input tensor on which the NCE operation operates
 * will determine the shape of the tensor.
 */
mlir::Value RestorePadAttrAfterSCFTilingPass::getInputOperand(mlir::tensor::PadOp padOp, mlir::OpBuilder builder) {
    auto& ctx = getContext();
    mlir::RankedTensorType srcType = padOp.getSourceType(), dstType = padOp.getResultType();
    bool isDynamic = false;
    VPUX_THROW_WHEN(srcType == nullptr || dstType == nullptr, "Expected RankedTensorType for PadOp source and result");

    auto newShape = SmallVector<int64_t>{};
    vpux::ShapeRef newBounds = vpux::ShapeRef{};
    transform(enumerate(padOp.getMixedLowPad()), std::back_inserter(newShape),
              [&srcType, &padOp, &isDynamic](auto&& indexedOffset) {
                  auto [idx, lowOffset] = indexedOffset;
                  auto highOffset = padOp.getMixedHighPad()[idx];

                  // Check if both low and high padding are static integer attributes
                  bool lowIsStatic = false, highIsStatic = false;

                  if (auto attr = mlir::dyn_cast<mlir::Attribute>(lowOffset)) {
                      lowIsStatic = mlir::isa<mlir::IntegerAttr>(attr);
                  }

                  if (auto attr = mlir::dyn_cast<mlir::Attribute>(highOffset)) {
                      highIsStatic = mlir::isa<mlir::IntegerAttr>(attr);
                  }

                  // If either padding is not static, make the dimension dynamic
                  if (!lowIsStatic || !highIsStatic) {
                      isDynamic = true;
                      return mlir::ShapedType::kDynamic;
                  }

                  return srcType.getShape()[idx];
              });

    if (isDynamic) {
        newBounds = getBoundedShape(srcType);
    }

    auto dstBounds = vpux::BoundsRef(newBounds);
    const auto inType = mlir::cast<NDTypeInterface>(srcType);
    auto outDesc = vpux::getTensorAttr(&ctx, inType.getDimsOrder(), /*memSpace=*/nullptr, dstBounds);
    auto newDstType = mlir::RankedTensorType::get(newShape, dstType.getElementType(), outDesc);
    if (newDstType != srcType) {
        return builder.create<mlir::tensor::CastOp>(appendLoc(padOp.getLoc(), "cast"), newDstType, padOp.getSource());
    }

    return padOp.getSource();
}

/**
 * @brief Processes a single scf.for operation to restore padding attributes on convolution-like operations.
 *
 * This function walks through all operations within the given ForOp, identifies convolution-like operations
 * that have a tensor.pad operation as their parent, and restores the original padding attribute while
 * removing the pad operation. The function handles dynamic tensor shapes by inserting tensor.cast operations
 * when necessary to ensure type compatibility.
 *
 * @param forOp The scf.for operation to process
 */
void RestorePadAttrAfterSCFTilingPass::processSCFForOp(mlir::scf::ForOp forOp) {
    SmallVector<std::pair<mlir::Operation*, mlir::tensor::PadOp>> worklist;
    forOp.walk([&](mlir::tensor::PadOp padOp) {
        for (auto user : padOp->getUsers()) {
            if (VPU::isNceOpWithPadAttr(user)) {
                worklist.push_back({user, padOp});
            }
        }
    });

    AffineChainUtils affineUtils;
    auto getPadAttribute = [&](mlir::tensor::PadOp padOp) {
        auto spatialDims = {Dims4D::Act::W, Dims4D::Act::H};
        llvm::SmallVector<int64_t> padValues;
        for (auto dim : spatialDims) {
            auto lowPad = padOp.getMixedLowPad()[dim.ind()];
            auto highPad = padOp.getMixedHighPad()[dim.ind()];

            llvm::DenseMap<mlir::Value, SmallVector<int64_t>> emptyMap;
            auto lowValue = affineUtils.getOpFoldResultValue(lowPad, emptyMap);
            auto highValue = affineUtils.getOpFoldResultValue(highPad, emptyMap);
            VPUX_THROW_WHEN(!lowValue.has_value() || !highValue.has_value(),
                            "Failed to compute static padding values for {0} operation", padOp->getName());
            padValues.emplace_back(lowValue.value()[0]);
            padValues.emplace_back(highValue.value()[0]);
        }

        return VPU::getPaddingAttr(padOp.getContext(), padValues[0], padValues[1], padValues[2], padValues[3]);
    };

    mlir::IRRewriter rewriter(forOp.getContext());
    for (auto [nceOp, padOp] : llvm::make_early_inc_range(worklist)) {
        _log.trace("Found convolution operation {0} with tensor.pad parent", nceOp->getName());

        rewriter.setInsertionPoint(nceOp);
        auto inputOperand = getInputOperand(padOp, rewriter);
        auto restoredPadAttr = getPadAttribute(padOp);

        mlir::IRMapping mapper;
        mapper.map(nceOp->getOperand(0), inputOperand);
        auto newNceOp = rewriter.clone(*nceOp, mapper);
        newNceOp->setAttr("pad", restoredPadAttr);

        // Replace all uses of the original convolution operation with the new one
        rewriter.replaceOp(nceOp, newNceOp->getResults());
        if (padOp->getUsers().empty()) {
            padOp.erase();
        }
    }
}

//
// safeRunOnFunc
//
void RestorePadAttrAfterSCFTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::OpBuilder builder(&ctx);

    _log.trace("Starting RestorePadAttrAfterSCFTiling pass on function: {0}", func.getName());

    // Iterate over all scf.for operations in the entry function
    func.walk([&](mlir::scf::ForOp forOp) {
        processSCFForOp(forOp);
    });
}

}  // namespace

//
// createRestorePadAttrAfterSCFTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createRestorePadAttrAfterSCFTilingPass(Logger log) {
    return std::make_unique<RestorePadAttrAfterSCFTilingPass>(log);
}
