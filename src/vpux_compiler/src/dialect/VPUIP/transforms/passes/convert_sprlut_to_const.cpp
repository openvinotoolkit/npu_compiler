//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/common_rewriters/convert_lut_to_const.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTSPRLUTTOCONST
#define GEN_PASS_DEF_CONVERTSPRLUTTOCONST
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// SprLUTConverter
//

class SprLUTConverter final : public VPUIP::LUTConverterBase {
public:
    SprLUTConverter(mlir::MLIRContext* ctx, Logger log, mlir::func::FuncOp netFunc)
            : LUTConverterBase(ctx, log, netFunc) {
        setDebugName("ConvertSprLUTToConstPass::SprLUTConverter");
    }

private:
    mlir::Value createLookupTableConst(VPUIP::NCEClusterTaskOp nceClusterTask,
                                       mlir::PatternRewriter& rewriter) const override;
    void replaceWithConstInput(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value sprLUT,
                               mlir::PatternRewriter& rewriter) const override;
    void removeSprLUTFromPPE(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::PatternRewriter& rewriter) const;
    VPU::PPEFpAttr createPPEWithoutSprLUT(VPU::PPEFpAttr prevPPE) const;
};

mlir::Value SprLUTConverter::createLookupTableConst(VPUIP::NCEClusterTaskOp nceClusterTask,
                                                    mlir::PatternRewriter& rewriter) const {
    const auto ppeOps = nceClusterTask.getPpe().getOps<VPUIP::PPETaskOp>();
    VPUX_THROW_WHEN(ppeOps.empty(), "{0}: expected PPE inside {1}, but it was not found", getDebugName(),
                    nceClusterTask);
    auto ppeOp = *ppeOps.begin();
    const auto nceClusterTaskPPEAttr = mlir::dyn_cast<VPU::PPEFpAttr>(ppeOp.getPpeAttr());
    const auto sprLUT = nceClusterTaskPPEAttr.getSprlut();

    const auto bufferType = vpux::getBufferType(sprLUT.getType());
    Const::ContentSetup setup(sprLUT, mlir::cast<mlir::Type>(bufferType));
    const auto contentAttr = Const::ContentAttr::get(sprLUT, setup);
    return rewriter.create<Const::DeclareOp>(nceClusterTask->getLoc(), bufferType, contentAttr).getOutput();
}

void SprLUTConverter::replaceWithConstInput(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value sprLUT,
                                            mlir::PatternRewriter& rewriter) const {
    auto newInput = [&]() -> mlir::Value {
        if (vpux::VPUIP::hasDistributedOperand(nceClusterTask)) {
            const auto sprLUTOutType = mlir::dyn_cast<VPUIP::DistributedBufferType>(sprLUT.getType());
            VPUX_THROW_WHEN(sprLUTOutType == nullptr,
                            "{0}: sprLUT output type is expected to be DistributedBufferType, but got {1}",
                            getDebugName(), sprLUT.getType());
            nceClusterTask.getSprLookupTableMutable().append(sprLUT);
        }
        return sprLUT;
    }();
    rewriter.modifyOpInPlace(nceClusterTask, [&] {
        nceClusterTask.getSprLookupTableMutable().assign(newInput);
        removeSprLUTFromPPE(nceClusterTask, rewriter);
    });
}

void SprLUTConverter::removeSprLUTFromPPE(VPUIP::NCEClusterTaskOp nceClusterTask,
                                          mlir::PatternRewriter& rewriter) const {
    rewriter.setInsertionPoint(&nceClusterTask.getPpe().front().front());
    for (auto ppeOp : nceClusterTask.getPpe().getOps<VPUIP::PPETaskOp>()) {
        const auto prevPPE = mlir::dyn_cast<VPU::PPEFpAttr>(ppeOp.getPpeAttr());
        VPUX_THROW_WHEN(prevPPE == nullptr, "{0}: expected PPEFpAttr as PPE attribute, but got {1}", getDebugName(),
                        ppeOp.getPpeAttr());
        const auto newPPE = createPPEWithoutSprLUT(prevPPE);
        ppeOp.setPpeAttr(newPPE);
    }
}

VPU::PPEFpAttr SprLUTConverter::createPPEWithoutSprLUT(VPU::PPEFpAttr prevPPE) const {
    return VPU::PPEFpAttr::get(prevPPE.getContext(), prevPPE.getMode(), prevPPE.getClampLow(), prevPPE.getClampHigh(),
                               prevPPE.getScale(), prevPPE.getPreluAlpha(), prevPPE.getBias(), prevPPE.getAdder(),
                               prevPPE.getIn1Mult(), prevPPE.getIn2Mult(), /*sprlut=*/nullptr);
}

//
// ConvertSprLUTToConstPass
//

class ConvertSprLUTToConstPass final : public VPUIP::impl::ConvertSprLUTToConstBase<ConvertSprLUTToConstPass> {
public:
    explicit ConvertSprLUTToConstPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertSprLUTToConstPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<Const::DeclareOp, VPUIP::CopyOp, VPURT::AllocDistributed, mlir::memref::AllocOp>();
    target.addDynamicallyLegalOp<VPUIP::NCEClusterTaskOp>([](VPUIP::NCEClusterTaskOp op) {
        if (op.getSprLookupTable() != nullptr) {
            return true;
        }
        for (auto ppeOp : op.getPpe().getOps<VPUIP::PPETaskOp>()) {
            const auto nceClusterTaskPPEAttr = mlir::dyn_cast<VPU::PPEFpAttr>(ppeOp.getPpeAttr());
            if (nceClusterTaskPPEAttr != nullptr && nceClusterTaskPPEAttr.getSprlut() != nullptr) {
                return false;
            }
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SprLUTConverter>(&ctx, _log, func);
    if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSprLUTToConstPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertSprLUTToConstPass(Logger log) {
    return std::make_unique<ConvertSprLUTToConstPass>(log);
}
