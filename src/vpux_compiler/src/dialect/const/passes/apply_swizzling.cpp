//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"

using namespace vpux;

namespace vpux::Const {
#define GEN_PASS_DECL_APPLYSWIZZLING
#define GEN_PASS_DEF_APPLYSWIZZLING
#include "vpux/compiler/dialect/const/passes.hpp.inc"
}  // namespace vpux::Const

namespace {

class ApplySwizzlingPass final : public Const::impl::ApplySwizzlingBase<ApplySwizzlingPass> {
public:
    explicit ApplySwizzlingPass() {
    }

private:
    void safeRunOnFunc() final;
};

void ApplySwizzlingPass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](Const::DeclareOp constOp) {
        auto constType = mlir::cast<vpux::NDTypeInterface>(constOp.getOutput().getType());
        auto swizzlingScheme = VPUIP::getSwizzlingSchemeAttr(constType);
        if (swizzlingScheme == nullptr) {
            return;
        }

        const auto contentAttr = constOp.getContentAttr();
        for (auto transf : contentAttr.getTransformations()) {
            if (mlir::isa<Const::SwizzleConstantAttr>(transf)) {
                return;
            }
        }

        auto module = constOp->getParentOfType<mlir::ModuleOp>();
        auto newContentAttr = constOp.getContentAttr()
                                      .transform()
                                      .swizzleConstant(VPUIP::getSwizzlingKey(constType),
                                                       static_cast<uint64_t>(config::getArch(module)))
                                      .get();
        mlir::OpBuilder builder(constOp);
        auto newConstOp =
                builder.create<vpux::Const::DeclareOp>(constOp.getLoc(), constType, std::move(newContentAttr));
        constOp.replaceAllUsesWith(newConstOp.getOutput());
        constOp.erase();
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Const::createApplySwizzlingPass() {
    return std::make_unique<ApplySwizzlingPass>();
}
