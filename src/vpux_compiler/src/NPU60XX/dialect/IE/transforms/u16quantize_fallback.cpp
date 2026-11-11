//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU60XX/dialect/IE/utils/mpe_engine_info_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

#include <llvm/ADT/DenseMap.h>

#include <utility>

namespace vpux::IE::arch60xx {
#define GEN_PASS_DECL_U16QUANTIZATIONFALLBACK
#define GEN_PASS_DEF_U16QUANTIZATIONFALLBACK
#include "vpux/compiler/NPU60XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch60xx

using namespace vpux;

namespace {

constexpr int64_t K_U8_TO_I8_ZERO_POINT_OFFSET = 128;

//
// Convert quant<u8:fp16> -> quant<i8:fp16>
//

static mlir::quant::QuantizedType convertU8ToI8QuantType(mlir::quant::QuantizedType q) {
    if (auto uq = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(q)) {
        const int64_t low = uq.getStorageTypeMin();

        return mlir::quant::UniformQuantizedType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt8Type(uq.getContext()), uq.getExpressedType(),
                uq.getScale(), uq.getZeroPoint() - low - K_U8_TO_I8_ZERO_POINT_OFFSET,
                uq.getStorageTypeMin() - low - K_U8_TO_I8_ZERO_POINT_OFFSET,
                uq.getStorageTypeMax() - low - K_U8_TO_I8_ZERO_POINT_OFFSET);
    }

    if (auto pa = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(q)) {
        const int64_t low = pa.getStorageTypeMin();

        SmallVector<int64_t> zpNew;
        zpNew.reserve(pa.getZeroPoints().size());
        for (auto zp : pa.getZeroPoints()) {
            zpNew.push_back(zp - low - K_U8_TO_I8_ZERO_POINT_OFFSET);
        }

        return mlir::quant::UniformQuantizedPerAxisType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt8Type(pa.getContext()), pa.getExpressedType(),
                pa.getScales(), zpNew, pa.getQuantizedDimension(),
                pa.getStorageTypeMin() - low - K_U8_TO_I8_ZERO_POINT_OFFSET,
                pa.getStorageTypeMax() - low - K_U8_TO_I8_ZERO_POINT_OFFSET);
    }

    VPUX_THROW("Unsupported U8 quant type in convertU8ToI8QuantType: {0}", q);
}

static bool isInput16BitsQuantized(mlir::Type inputType) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(inputType);
    auto qType = mlir::cast<mlir::quant::QuantizedType>(ndType.getElementType());
    auto qStorageType = qType.getStorageType();
    return qStorageType.isInteger(16);
}

static bool hasAsym16BitZP(vpux::NDTypeInterface outputType) {
    auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputType.getElementType());
    return qType && qType.getStorageTypeIntegralWidth() == 16 && !IE::isSymmetricQuantType(qType) &&
           !IE::areAllQuantTypeZeroPointsEqualToZero(qType);
}

static bool shouldForceF16Output(mlir::Operation* op, vpux::NDTypeInterface outputType,
                                 bool isInput16BitsQuantizedLocal) {
    const auto arch = config::getArch(op);
    if (arch != config::ArchKind::NPU60XX) {
        return false;
    }

    auto rev = config::getRevisionID(op);
    Logger::global().warning("Default revision is {0}; revision A path will be deprecated", rev);
    if (rev < config::RevisionID::REVISION_A0 || rev >= config::RevisionID::REVISION_B) {
        return false;
    }

    // For NPU6, revision < B it is needed to mitigate the HW erratum which produce accuracy problems when
    // fusing asymmetric 16 bits quantization at the output. Details available on E-#182547
    return hasAsym16BitZP(outputType) && isInput16BitsQuantizedLocal;
}

static Const::DeclareOp maybeConvertConstU8ToI8(mlir::PatternRewriter& rewriter, mlir::Value value) {
    Const::DeclareOp newConst;

    if (auto decl = value.getDefiningOp<Const::DeclareOp>()) {
        auto ndType = mlir::cast<vpux::NDTypeInterface>(decl.getType());
        auto elemType = ndType.getElementType();

        if (auto q = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
            if (auto storageInt = mlir::dyn_cast<mlir::IntegerType>(q.getStorageType())) {
                const bool isU8 = (storageInt.getWidth() == 8) && !q.isSigned();
                if (isU8) {
                    auto newQ = convertU8ToI8QuantType(q);
                    auto newType = ndType.changeElemType(newQ);
                    auto newContent = decl.getContentAttr().transform().convertElemType(newQ).get();
                    newConst = Const::DeclareOp::create(rewriter, decl.getLoc(), newType, newContent);
                }
            }
        }
    }

    return newConst;
}

static bool requiresDCIMConstraintsCheck(mlir::Value inputVal) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(inputVal.getType());
    auto elemType = inputType.getElementType();

    if (elemType.isF16()) {
        return false;
    }
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
        return qType.getStorageTypeIntegralWidth() == 16;
    }
    return false;
}

bool shouldConvertU16QuantizationFallback(IE::ConvolutionOp op) {
    if (!requiresDCIMConstraintsCheck(op.getInput())) {
        return false;
    }
    auto mpeEngineInterface = mlir::cast<IE::MPEEngineInfoOpInterface>(op.getOperation());
    if (auto concreteAttr = mlir::dyn_cast<VPU::MPEEngine60XXAttr>(mpeEngineInterface.getMPEEngine())) {
        if (concreteAttr.getMode().getValue() == VPU::MPEEngine60XXMode::DCIM) {
            return false;
        }
    }
    return true;
}

bool shouldConvertU16QuantizationFallback(IE::MatMulOp op) {
    if (!requiresDCIMConstraintsCheck(op.getInput1())) {
        return false;
    }
    auto mpeEngineInterface = mlir::cast<IE::MPEEngineInfoOpInterface>(op.getOperation());
    if (auto concreteAttr = mlir::dyn_cast<VPU::MPEEngine60XXAttr>(mpeEngineInterface.getMPEEngine())) {
        if (concreteAttr.getMode().getValue() == VPU::MPEEngine60XXMode::DCIM) {
            return false;
        }
    }
    return true;
}

struct QuantDequantCounts {
    size_t quantize = 0;
    size_t dequantize = 0;
};

QuantDequantCounts countQuantDequantOps(mlir::func::FuncOp func) {
    QuantDequantCounts counts;
    func.walk([&](mlir::Operation* op) {
        if (mlir::isa<IE::QuantizeOp>(op)) {
            ++counts.quantize;
        }
        if (mlir::isa<IE::DequantizeOp>(op)) {
            ++counts.dequantize;
        }
    });
    return counts;
}

// Merge identical input Dequantize ops that share the same producer.
//
// DequantizeUnsupportedConv/DequantizeUnsupportedMatMul insert a dedicated Dequantize for every
// consumer of a 16-bit quantized value. When several consumers share one producer, this yields
// multiple identical Dequantize ops, keeping the producer multi-use and blocking
// FuseOutstandingDequant from folding the Dequantize into the producer. Redirecting the uses to a
// single representative restores the single-use producer that folding relies on.
void removeDuplicateInputDequantizes(mlir::func::FuncOp func) {
    llvm::DenseMap<std::pair<mlir::Value, mlir::Type>, IE::DequantizeOp> representatives;
    SmallVector<IE::DequantizeOp> duplicates;
    func.walk([&](IE::DequantizeOp dequantOp) {
        const auto key = std::make_pair(dequantOp.getInput(), dequantOp.getOutput().getType());
        const auto it = representatives.find(key);
        if (it == representatives.end()) {
            representatives.try_emplace(key, dequantOp);
            return;
        }
        auto representative = it->second;
        // Redirecting uses requires the kept op to dominate them, so only merge within a block and
        // keep whichever Dequantize appears first.
        if (representative->getBlock() != dequantOp->getBlock()) {
            return;
        }
        if (dequantOp->isBeforeInBlock(representative)) {
            std::swap(representative, dequantOp);
            it->second = representative;
        }
        dequantOp.getOutput().replaceAllUsesWith(representative.getOutput());
        duplicates.push_back(dequantOp);
    });
    for (auto dequantOp : duplicates) {
        dequantOp.erase();
    }
}

//
// DequantizeUnsupportedConv
//

class DequantizeUnsupportedConv final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    DequantizeUnsupportedConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("DequantizeUnsupportedConv ");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeUnsupportedConv::matchAndRewrite(IE::ConvolutionOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertU16QuantizationFallback(origOp)) {
        return mlir::failure();
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    Const::DeclareOp newConst = maybeConvertConstU8ToI8(rewriter, origOp.getFilter());
    const auto isInput16BitsQuantizedLocal = isInput16BitsQuantized(origOp.getInput().getType());
    bool forceF16Output = shouldForceF16Output(origOp.getOperation(), outputType, isInput16BitsQuantizedLocal);
    auto f16Type = rewriter.getF16Type();
    auto dequantInput =
            IE::DequantizeOp::create(rewriter, takeOpLoc(origOp, "dequant_in"), inputType.changeElemType(f16Type),
                                     origOp.getInput(), mlir::TypeAttr::get(f16Type));
    auto convOutType = forceF16Output ? outputType.changeElemType(f16Type) : outputType;

    auto newConv = cloneConvolutionOp(rewriter, origOp, convOutType, dequantInput,
                                      newConst != nullptr ? newConst.getOutput() : origOp.getFilter());

    if (forceF16Output) {
        auto finalQuantize =
                IE::QuantizeOp::create(rewriter, takeOpLoc(origOp, "quantize_out"), outputType, newConv.getResult(),
                                       mlir::TypeAttr::get(outputType.getElementType()));
        rewriter.replaceOp(origOp, finalQuantize.getResult());
    } else {
        rewriter.replaceOp(origOp, newConv.getResult());
    }

    return mlir::success();
}

//
// DequantizeUnsupportedMatMul
//

class DequantizeUnsupportedMatMul final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    DequantizeUnsupportedMatMul(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
        setDebugName("DequantizeUnsupportedMatMul ");
    }

    mlir::LogicalResult matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeUnsupportedMatMul::matchAndRewrite(IE::MatMulOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertU16QuantizationFallback(origOp)) {
        return mlir::failure();
    }

    auto location = origOp->getLoc();
    auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput1().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    Const::DeclareOp newConst = maybeConvertConstU8ToI8(rewriter, origOp.getInput2());
    mlir::Value weights = newConst != nullptr ? newConst.getOutput() : origOp.getInput2();

    const auto isInput16BitsQuantizedLocal = isInput16BitsQuantized(origOp.getInput1().getType());
    bool forceF16Output = shouldForceF16Output(origOp.getOperation(), outputType, isInput16BitsQuantizedLocal);

    auto f16Type = rewriter.getF16Type();
    auto dequantInput =
            IE::DequantizeOp::create(rewriter, takeOpLoc(origOp, "dequant_in"), inputType.changeElemType(f16Type),
                                     origOp.getInput1(), mlir::TypeAttr::get(f16Type));

    auto matmulOutType = forceF16Output ? outputType.changeElemType(f16Type) : outputType;
    const bool transposeA = origOp.getTransposeAAttr() != nullptr;
    const bool transposeB = origOp.getTransposeBAttr() != nullptr;
    auto newMatMul = IE::MatMulOp::create(rewriter, location, matmulOutType, dequantInput.getOutput(), weights,
                                          transposeA, transposeB);

    if (forceF16Output) {
        auto finalQuantize =
                IE::QuantizeOp::create(rewriter, takeOpLoc(origOp, "quantize_out"), outputType, newMatMul.getOutput(),
                                       mlir::TypeAttr::get(outputType.getElementType()));
        rewriter.replaceOp(origOp, finalQuantize.getResult());
    } else {
        rewriter.replaceOp(origOp, newMatMul.getOutput());
    }

    return mlir::success();
}

//
// U16QuantizationFallbackPass
//

class U16QuantizationFallbackPass final :
        public IE::arch60xx::impl::U16QuantizationFallbackBase<U16QuantizationFallbackPass> {
public:
    explicit U16QuantizationFallbackPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

mlir::LogicalResult U16QuantizationFallbackPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    return mlir::success();
}

void U16QuantizationFallbackPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto countsBefore = countQuantDequantOps(func);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DequantizeUnsupportedConv>(&ctx, _log);
    patterns.add<DequantizeUnsupportedMatMul>(&ctx, _log);

    walkAndApplyPatterns(func, std::move(patterns));

    const auto countsAfter = countQuantDequantOps(func);
    if (countsAfter.quantize > countsBefore.quantize || countsAfter.dequantize > countsBefore.dequantize) {
        _log.trace("U16QuantizationFallback introduced new quantize/dequantize ops (Quantize: {0}→{1}, Dequantize: "
                   "{2}→{3}), running nested ConvertQuantizeOpsToNceOps",
                   countsBefore.quantize, countsAfter.quantize, countsBefore.dequantize, countsAfter.dequantize);
        // Collapse identical input Dequantize ops first, so producers feeding multiple consumers
        // become single-use and can be folded by FuseOutstandingDequant below.
        removeDuplicateInputDequantizes(func);
        mlir::OpPassManager nestedPipeline(mlir::func::FuncOp::getOperationName());
        nestedPipeline.addPass(IE::createFuseOutstandingDequant(_log));
        nestedPipeline.addPass(IE::createFuseOutstandingQuantPass(_log));
        if (mlir::failed(runPipeline(nestedPipeline, func))) {
            signalPassFailure();
        }
    }
}

}  // namespace

//
// createU16QuantizationFallbackPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch60xx::createU16QuantizationFallbackPass(Logger log) {
    return std::make_unique<U16QuantizationFallbackPass>(log);
}
