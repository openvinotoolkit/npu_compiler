//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/sprlut_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

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

class SprLUTConverter final : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    SprLUTConverter(mlir::MLIRContext* ctx, Logger log, mlir::func::FuncOp netFunc)
            : mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log), _netFunc(netFunc) {
        setDebugName("ConvertSprLUTToConstPass::SprLUTConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceClusterTask,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value createSprLookupTableConst(VPUIP::NCEClusterTaskOp nceClusterTask,
                                          mlir::PatternRewriter& rewriter) const;
    mlir::Value createCopyDestination(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value sprLUTConst,
                                      mlir::PatternRewriter& rewriter) const;
    VPU::DistributionInfoAttr createDistributionInfoAttr(VPUIP::DistributedBufferType inputDistribType,
                                                         VPUIP::NCEClusterTaskOp nceClusterTask) const;
    VPUIP::DistributedBufferType createDistributedBufferType(VPU::DistributionInfoAttr distributedInfo,
                                                             VPUIP::NCEClusterTaskOp nceClusterTask,
                                                             mlir::Value sprLUTConst) const;
    void replaceAttrWithConst(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value sprLUT,
                              mlir::PatternRewriter& rewriter) const;
    void removeSprLUTFromPPE(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::PatternRewriter& rewriter) const;
    VPU::PPEFpAttr createPPEWithoutSprLUT(VPU::PPEFpAttr prevPPE) const;

private:
    Logger _log;
    mutable mlir::func::FuncOp _netFunc;
};

mlir::LogicalResult SprLUTConverter::matchAndRewrite(VPUIP::NCEClusterTaskOp nceClusterTask,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), nceClusterTask->getName(), nceClusterTask->getLoc());

    const auto sprLUTConst = [&]() {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(&_netFunc.getBody().front().front());
        return createSprLookupTableConst(nceClusterTask, rewriter);
    }();

    const auto copyDst = createCopyDestination(nceClusterTask, sprLUTConst, rewriter);
    const auto sprLutNceInput =
            rewriter.create<VPUIP::CopyOp>(nceClusterTask->getLoc(), sprLUTConst, copyDst).getOutput();

    replaceAttrWithConst(nceClusterTask, sprLutNceInput, rewriter);

    return mlir::success();
}

mlir::Value SprLUTConverter::createSprLookupTableConst(VPUIP::NCEClusterTaskOp nceClusterTask,
                                                       mlir::PatternRewriter& rewriter) const {
    const auto ppeOps = nceClusterTask.getPpe().getOps<VPUIP::PPETaskOp>();
    VPUX_THROW_WHEN(ppeOps.empty(), "{0}: expected PPE inside {1}, but it was not found", getDebugName(),
                    nceClusterTask);
    auto ppeOp = *ppeOps.begin();
    const auto nceClusterTaskPPEAttr = mlir::dyn_cast<VPU::PPEFpAttr>(ppeOp.getPpeAttr());
    const auto sprLUT = nceClusterTaskPPEAttr.getSprlut();

    const auto bufferType = vpux::getBufferType(sprLUT.getType());
    Const::ContentSetup setup(mlir::cast<mlir::Type>(bufferType));
    const auto contentAttr = Const::ContentAttr::get(sprLUT, setup);
    return rewriter.create<Const::DeclareOp>(nceClusterTask->getLoc(), bufferType, contentAttr).getOutput();
}

mlir::Value SprLUTConverter::createCopyDestination(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value sprLUTConst,
                                                   mlir::PatternRewriter& rewriter) const {
    const auto input = VPUIP::getTopBufferOfNCEClusterTiling(nceClusterTask, nceClusterTask.getInput());
    const auto inputType = input.getType();

    return llvm::TypeSwitch<mlir::Type, mlir::Value>(inputType)
            .Case<VPUIP::DistributedBufferType>([&](VPUIP::DistributedBufferType distributedBufferType) {
                const auto distributedInfo = createDistributionInfoAttr(distributedBufferType, nceClusterTask);
                const auto ditributedBufferType =
                        createDistributedBufferType(distributedInfo, nceClusterTask, sprLUTConst);

                auto alignment = vpux::getIntAttr(nceClusterTask.getContext(), VPU::SPRLUT_ALIGNMENT_REQUIREMENT);

                if (auto nceClusterTiling = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(nceClusterTask->getParentOp())) {
                    rewriter.setInsertionPoint(nceClusterTiling);
                }
                auto allocDistributed = rewriter.create<VPURT::AllocDistributed>(
                        nceClusterTask.getLoc(), ditributedBufferType, alignment, nullptr);
                return allocDistributed->getResult(0);
            })
            .Case<mlir::MemRefType>([&](auto) {
                const auto constOutType = mlir::dyn_cast<mlir::MemRefType>(sprLUTConst.getType());
                VPUX_THROW_WHEN(constOutType == nullptr,
                                "{0}: sprLUT const output type is expected to be MemRefType, but got {1}",
                                getDebugName(), sprLUTConst.getType());
                const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(nceClusterTask.getContext(),
                                                                      stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
                const auto cmxMemType = mlir::MemRefType::get(constOutType.getShape(), constOutType.getElementType(),
                                                              constOutType.getLayout(), memSpaceCMX);
                const auto allocOp = rewriter.create<mlir::memref::AllocOp>(nceClusterTask.getLoc(), cmxMemType);
                return allocOp->getResult(0);
            })
            .Default([&](mlir::Type inputType) {
                VPUX_THROW("{0}: `{1}` is not supported as an input type", getDebugName(), inputType);
                return mlir::Value{};
            });
}

VPU::DistributionInfoAttr SprLUTConverter::createDistributionInfoAttr(VPUIP::DistributedBufferType inputDistribType,
                                                                      VPUIP::NCEClusterTaskOp nceClusterTask) const {
    auto inputDistribInfo = inputDistribType.getDistribution();
    VPUX_THROW_WHEN(inputDistribInfo == nullptr, "{0}: inputDistribInfo == nullptr for the input type is not allowed",
                    getDebugName());
    const auto duplicatedDistrModeAttr =
            VPU::DistributionModeAttr::get(nceClusterTask.getContext(), VPU::DistributionMode::DUPLICATED);
    return VPU::DistributionInfoAttr::get(nceClusterTask.getContext(), duplicatedDistrModeAttr, nullptr, nullptr,
                                          nullptr, nullptr, inputDistribInfo.getNumClusters(), nullptr, nullptr,
                                          nullptr, nullptr, nullptr, nullptr, nullptr);
}

VPUIP::DistributedBufferType SprLUTConverter::createDistributedBufferType(VPU::DistributionInfoAttr distributedInfo,
                                                                          VPUIP::NCEClusterTaskOp nceClusterTask,
                                                                          mlir::Value sprLUTConst) const {
    const auto memSpaceCMX =
            vpux::IndexedSymbolAttr::get(nceClusterTask.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
    const auto ndTypeInterface = mlir::cast<vpux::NDTypeInterface>(sprLUTConst.getType());
    return VPUIP::DistributedBufferType::get(
            nceClusterTask.getContext(), ndTypeInterface.getShape().raw(), ndTypeInterface.getElementType(),
            mlir::dyn_cast<mlir::MemRefType>(sprLUTConst.getType()).getLayout(), memSpaceCMX, distributedInfo);
}

void SprLUTConverter::replaceAttrWithConst(VPUIP::NCEClusterTaskOp nceClusterTask, mlir::Value sprLUT,
                                           mlir::PatternRewriter& rewriter) const {
    auto newInput = [&]() -> mlir::Value {
        if (auto nceClusterTiling = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(nceClusterTask->getParentOp())) {
            auto& bodyBlock = nceClusterTiling.getBody().front();
            const auto sprLUTOutType = mlir::dyn_cast<VPUIP::DistributedBufferType>(sprLUT.getType());
            VPUX_THROW_WHEN(sprLUTOutType == nullptr,
                            "{0}: sprLUT output type is expected to be DistributedBufferType, but got {1}",
                            getDebugName(), sprLUT.getType());
            nceClusterTiling.getInputsMutable().append(sprLUT);
            return bodyBlock.insertArgument(bodyBlock.getNumArguments() - nceClusterTiling.getOutputs().size(),
                                            sprLUTOutType.getCompactType(), sprLUT.getLoc());
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
