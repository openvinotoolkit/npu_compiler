//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTDEQUANTIZETODYNAMICDEQUANTIZE
#define GEN_PASS_DEF_CONVERTDEQUANTIZETODYNAMICDEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertDequantizeToDynamicDequantizePass
//

class ConvertDequantizeToDynamicDequantizePass final :
        public IE::impl::ConvertDequantizeToDynamicDequantizeBase<ConvertDequantizeToDynamicDequantizePass> {
public:
    explicit ConvertDequantizeToDynamicDequantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

// Converts a list of values to a ranked float tensor constant.
static mlir::Value createFloatConst(mlir::IRRewriter& rewriter, mlir::Location loc, mlir::FloatType floatType,
                                    ArrayRef<int64_t> shape, ArrayRef<double> vals) {
    SmallVector<llvm::APFloat> apFloats;
    apFloats.reserve(vals.size());
    for (double v : vals) {
        bool lossy = false;
        llvm::APFloat apVal(v);
        apVal.convert(floatType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &lossy);
        apFloats.push_back(std::move(apVal));
    }
    const auto type = mlir::RankedTensorType::get(shape, floatType);
    const auto attr = mlir::DenseElementsAttr::get(type, ArrayRef(apFloats));

    return rewriter.create<Const::DeclareOp>(loc, type, Const::ContentAttr::get(attr)).getOutput();
}

void ConvertDequantizeToDynamicDequantizePass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    // Converts IE.Dequantize(quant_input) into IE.DynamicDequantize(quant_input, scale_const, zp_const).
    // The scale and zero-point constants are extracted from the quant.uniform element type embedded in quant_input.
    func->walk([&](IE::DequantizeOp origOp) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
        const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputType.getElementType());
        if (qType == nullptr) {
            return;
        }
        if (!mlir::isa<mlir::quant::UniformQuantizedType>(qType) &&
            !mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(qType)) {
            return;
        }

        const auto [scaleVals, zpVals] = extractScalesAndZeroPoints(qType);
        const auto inputRank = static_cast<int64_t>(inputType.getShape().size());

        // Determine the per-channel quantization axis; per-tensor defaults to axis 0.
        int64_t quantDim = 0;
        if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qType)) {
            quantDim = perAxisType.getQuantizedDimension();
        }

        // Build scale/zp tensor shape: same rank as input, numScales at quantDim, 1 elsewhere.
        SmallVector<int64_t> paramShape(inputRank, 1);
        paramShape[quantDim] = static_cast<int64_t>(scaleVals.size());

        rewriter.setInsertionPoint(origOp);

        // Create scale constant using the expressed type (e.g. f32, f16).
        const auto expressedFloatType = mlir::cast<mlir::FloatType>(qType.getExpressedType());
        const mlir::Value scaleConst =
                createFloatConst(rewriter, takeOpLoc(origOp, "scale"), expressedFloatType, paramShape, scaleVals);

        // Create ZP constant using the storage type.
        // Integer storage (si16, u16, si8, u8, si4, u4, si2, u2).
        // Float storage (f8e4m3fn, f8e5m2, f4e2m1fn).
        const auto storageType = qType.getStorageType();
        mlir::Type zpElemType;
        mlir::Value zpValue;

        if (const auto storageIntType = mlir::dyn_cast<mlir::IntegerType>(storageType)) {
            const auto zpIntSignedness = qType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
            zpElemType = mlir::IntegerType::get(rewriter.getContext(), storageIntType.getWidth(), zpIntSignedness);
            const auto zpIntType = mlir::cast<mlir::IntegerType>(zpElemType);
            SmallVector<llvm::APInt> zpAPInts;
            zpAPInts.reserve(zpVals.size());
            for (int64_t zp : zpVals) {
                zpAPInts.push_back(llvm::APInt(zpIntType.getWidth(), static_cast<uint64_t>(zp), qType.isSigned()));
            }
            const auto zpType = mlir::RankedTensorType::get(paramShape, zpElemType);
            const auto zpAttr = mlir::DenseElementsAttr::get(zpType, ArrayRef(zpAPInts));
            zpValue =
                    rewriter.create<Const::DeclareOp>(takeOpLoc(origOp, "zp"), zpType, Const::ContentAttr::get(zpAttr))
                            .getOutput();
        } else {
            const auto storageFloatType = mlir::cast<mlir::FloatType>(storageType);
            zpElemType = storageFloatType;
            const SmallVector<double> zpDoubles(zpVals.begin(), zpVals.end());
            zpValue = createFloatConst(rewriter, takeOpLoc(origOp, "zp"), storageFloatType, paramShape, zpDoubles);
        }

        // Rewrite quant type constant with the raw storage type
        mlir::Value rawInput = origOp.getInput();
        if (auto inputConst = rawInput.getDefiningOp<Const::DeclareOp>()) {
            const auto rawInputType = mlir::cast<vpux::NDTypeInterface>(rawInput.getType()).changeElemType(zpElemType);
            auto rawContentAttr = inputConst.getContentAttr().transform().castElemType(zpElemType).get();
            rawInput =
                    rewriter.replaceOpWithNewOp<Const::DeclareOp>(inputConst, rawInputType, rawContentAttr).getOutput();
        }

        rewriter.replaceOpWithNewOp<IE::DynamicDequantizeOp>(origOp, rawInput, scaleConst, zpValue,
                                                             origOp.getDstElemType());
    });
}

}  // namespace

//
// createConvertDequantizeToDynamicDequantizePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertDequantizeToDynamicDequantizePass(Logger log) {
    return std::make_unique<ConvertDequantizeToDynamicDequantizePass>(log);
}
