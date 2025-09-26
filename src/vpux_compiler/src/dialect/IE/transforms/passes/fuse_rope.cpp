//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEROPE
#define GEN_PASS_DEF_FUSEROPE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseRoPEPass
//

class FuseRoPEPass final : public IE::impl::FuseRoPEBase<FuseRoPEPass> {
public:
    explicit FuseRoPEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::Operation* getSliceOrStridedSliceOp(mlir::Operation* op) {
    if (mlir::isa_and_nonnull<IE::SliceOp, IE::StridedSliceOp>(op)) {
        return op;
    }
    return nullptr;
}

//
// safeRunOnFunc
//

// Match pattern
// Input --------> IE.Multiply --------------------------------------------------
//       |                                                                      |
//       --> IE.StridedSlice -> IE.Multiply                                     |   -> IE.Add
//       |                                 | ---> IE.Concat ---> IE.Multiply ----
//       --> IE.StridedSlice ---------------                                              ^
//   |                                                                                    |
//   |                                                                                    |
//    -------------------------------------------------------------------------------------
// Or
// Input --------> IE.Multiply ---------------IE.AffineReshape-----------------------------------------
//       |                                                                                            |
//       |                    |--> IE.StridedSlice -> IE.Multiply                                     |   -> IE.Add
//       IE.AffineReshape ----|                                  | ---> IE.Concat ---> IE.Multiply ----
//                            |--> IE.StridedSlice ---------------                                              ^
//   |                                                                                                          |
//   |                                                                                                          |
//    -----------------------------------------------------------------------------------------------------------

void FuseRoPEPass::safeRunOnFunc() {
    auto func = getOperation();
    func->walk([&](IE::AddOp addOp) {
        auto skipReshapeIfPresent = [](mlir::Operation* op) -> mlir::Operation* {
            if (!mlir::isa_and_nonnull<IE::AffineReshapeOp>(op)) {
                return op;
            }
            if (!op->hasOneUse()) {
                return nullptr;
            }
            return op->getOperand(0).getDefiningOp();
        };

        auto mulOp1 =
                mlir::dyn_cast_or_null<IE::MultiplyOp>(skipReshapeIfPresent(addOp->getOperand(0).getDefiningOp()));
        auto mulOp2 = mlir::dyn_cast_or_null<IE::MultiplyOp>(addOp.getOperand(1).getDefiningOp());
        if (!mulOp1 || !mulOp2) {
            return;
        }
        auto input_cos = mulOp1.getOperand(1);
        auto input_sin = mulOp2.getOperand(1);

        auto concatOp = mulOp2.getOperand(0).getDefiningOp<IE::ConcatOp>();
        if (!concatOp || concatOp.getInputs().size() != 2) {
            return;
        }
        auto mulOp3 = concatOp.getOperand(0).getDefiningOp<IE::MultiplyOp>();
        auto stridedSliceOp2 = getSliceOrStridedSliceOp(concatOp.getOperand(1).getDefiningOp());
        if (!mulOp3 || !stridedSliceOp2) {
            return;
        }
        auto stridedSliceOp1 = getSliceOrStridedSliceOp(mulOp3.getOperand(0).getDefiningOp());
        if (!stridedSliceOp1) {
            return;
        }
        auto tensorType = mlir::dyn_cast<vpux::NDTypeInterface>(stridedSliceOp1->getOperand(0).getType());
        if (!tensorType || tensorType.getRank() != 4) {
            return;
        }

        const auto shape = tensorType.getShape();

        // For avoiding performance decrease for certain networks, we limit the cases below for H = 1.
        // Follow next ticket for updates on generalizing the pass: E#162922
        constexpr int64_t unsupportedH = 1;
        const auto channelAndWidth = SmallVector<SmallVector<int64_t>>{{1, 64}, {64, 64}, {16, 128}, {2, 128}};

        if (shape[Dims4D::Act::H] == unsupportedH) {
            auto it = std::find(channelAndWidth.begin(), channelAndWidth.end(),
                                SmallVector<int64_t>{shape[Dims4D::Act::C], shape[Dims4D::Act::W]});
            if (it == channelAndWidth.end()) {
                return;
            }
        }

        _log.trace("RoPE pattern matched for operation {0} at {1}", addOp->getName(), addOp->getLoc());
        auto builder = mlir::OpBuilder(addOp);
        auto cosShape = mlir::cast<mlir::RankedTensorType>(input_cos.getType()).getShape();
        auto sinShape = mlir::cast<mlir::RankedTensorType>(input_sin.getType()).getShape();
        if (cosShape != sinShape) {
            input_cos = builder.create<IE::ReshapeOp>(appendLoc(addOp->getLoc(), "_cos_reshape"), input_cos, nullptr,
                                                      false, getIntArrayAttr(builder, sinShape));
            _log.trace("Reshaped input_cos to match input_sin shape");
        }
        auto ropeOp = builder.create<IE::RoPEOp>(appendLoc(addOp->getLoc(), "rope"), stridedSliceOp1->getOperand(0),
                                                 input_cos, input_sin);
        addOp->replaceAllUsesWith(ropeOp);
    });
}

}  // namespace

//
// createFuseRoPEPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseRoPEPass(Logger log) {
    return std::make_unique<FuseRoPEPass>(log);
}
