//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTEXPAND
#define GEN_PASS_DEF_CONVERTEXPAND
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

// Helper class to wrap the arguments for ExpandConverter::applyPadding
class PaddingContext {
public:
    PaddingContext(const mlir::Location loc, const ShapeRef inShape, const mlir::Value expandedBuffer,
                   const mlir::Value constantBuffer)
            : _loc(loc), _inShape(inShape), _expandedBuffer(expandedBuffer), _constantBuffer(constantBuffer) {};
    PaddingContext(const PaddingContext&) = delete;
    PaddingContext(const PaddingContext&&) = delete;
    PaddingContext& operator=(const PaddingContext&) = delete;
    PaddingContext& operator=(const PaddingContext&&) = delete;
    ~PaddingContext() = default;

    const mlir::Location _loc;
    ShapeRef _inShape;
    const mlir::Value _expandedBuffer;
    const mlir::Value _constantBuffer;
};

//
// ConvertExpandPass
//

class ConvertExpandPass final : public VPUIP::impl::ConvertExpandBase<ConvertExpandPass> {
public:
    explicit ConvertExpandPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    mlir::Value applyPadding(const int64_t padAxis, const int64_t padValue, ArrayRef<int64_t> inSubViewOffsets,
                             const PaddingContext& padCtx, mlir::Type expectedElemType, mlir::OpBuilder& builder) const;

    std::array<int64_t, 4> getMaxExpandConstShapes(mlir::func::FuncOp func, Logger log);
    std::array<Const::DeclareOp, 4> getZeroConstOps(mlir::func::FuncOp func, mlir::MLIRContext& ctx,
                                                    mlir::OpBuilder& builder);

    Dim getPadDim(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType);
};

mlir::Value ConvertExpandPass::applyPadding(const int64_t padAxis, const int64_t padValue,
                                            ArrayRef<int64_t> inSubViewOffsets, const PaddingContext& padCtx,
                                            mlir::Type expectedElemType, mlir::OpBuilder& builder) const {
    const auto& location = padCtx._loc;
    const auto& inShape = padCtx._inShape;
    const auto& expandedBuffer = padCtx._expandedBuffer;
    const auto& constantBuffer = padCtx._constantBuffer;
    SmallVector<int64_t> subViewOffsets;
    std::copy(inSubViewOffsets.begin(), inSubViewOffsets.end(), std::back_inserter(subViewOffsets));

    auto constantOp = constantBuffer.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(constantOp != nullptr, "Can not get constant Op");

    const auto constShapeType = mlir::cast<vpux::NDTypeInterface>(constantOp.getOutput().getType());
    const auto constOuputShape = constShapeType.getShape();
    Shape subViewShape;
    std::copy(inShape.begin(), inShape.end(), std::back_inserter(subViewShape));
    subViewShape[Dim(padAxis)] = padValue;
    VPUX_THROW_UNLESS(subViewShape.totalSize() <= constOuputShape.totalSize(),
                      "Constant subview shape size '{0}' large than full size '{1}'", subViewShape.totalSize(),
                      constOuputShape.totalSize());

    // Step 1: Create SubView Op to get the right constant size
    VPUX_THROW_UNLESS(constOuputShape.size() == 1, "Constant Op unexpect shape size '{0}'", constOuputShape);
    const auto constSubviewOffset = SmallVector<int64_t>(1, 0);
    const auto constSubviewShape = SmallVector<int64_t>(1, subViewShape.totalSize());
    auto constSubView =
            builder.create<VPUIP::SubViewOp>(appendLoc(location, "_constant_subview_{0}_{1}", padAxis, padValue),
                                             constantOp, constSubviewOffset, constSubviewShape);

    // Step 2: Create Reshape Op to match concat shape with expected type
    const auto shapeType = mlir::cast<NDTypeInterface>(constSubView.getType());
    auto newShapeType = shapeType.changeShape(subViewShape);
    if (isFloat8Quantized(expectedElemType)) {
        // Reinterpret the constant from fp8 to quant.uniform<fp8:...>
        newShapeType = newShapeType.changeElemType(expectedElemType);
    }
    auto reshapeOp =
            builder.create<VPUIP::GenericReshapeOp>(appendLoc(location, "_constant_reshape_{0}_{1}", padAxis, padValue),
                                                    newShapeType, constSubView.getResult());

    // Step 3: Create PermuteCast Op to match concat layout
    const auto expandOutBufferType = mlir::cast<NDTypeInterface>(expandedBuffer.getType());
    const auto newLayoutType = newShapeType.changeDimsOrder(expandOutBufferType.getDimsOrder());
    const auto dstOrderAttr =
            mlir::AffineMapAttr::get(expandOutBufferType.getDimsOrder().toAffineMap(reshapeOp.getContext()));
    const auto memPermAttr = mlir::AffineMapAttr::get(
            DimsOrder::fromNumDims(mlir::cast<NDTypeInterface>(reshapeOp.getOutput().getType()).getRank())
                    .toAffineMap(reshapeOp.getContext()));
    auto permuteCastOp =
            builder.create<VPUIP::PermuteCastOp>(appendLoc(location, "_constant_permute_{0}_{1}", padAxis, padValue),
                                                 newLayoutType, reshapeOp.getOutput(), dstOrderAttr, memPermAttr);

    // Step 4: Create Copy Op for concat concatant input
    auto subView = builder.create<VPUIP::SubViewOp>(appendLoc(location, "_expand_subview_{0}_{1}", padAxis, padValue),
                                                    expandedBuffer, ShapeRef(subViewOffsets), subViewShape);
    auto subViewCopy = builder.create<VPUIP::CopyOp>(appendLoc(location, "_expand_copy_{0}_{1}", padAxis, padValue),
                                                     permuteCastOp.getResult(), subView);

    return subViewCopy.getOutput();
}

std::array<int64_t, 4> ConvertExpandPass::getMaxExpandConstShapes(mlir::func::FuncOp func, Logger log) {
    int64_t maxFP16ShapeSize = 0;
    int64_t maxINT8ShapeSize = 0;
    int64_t maxFP8E4M3FNShapeSize = 0;
    int64_t maxFP8E5M2ShapeSize = 0;

    func->walk([&](VPUIP::ExpandOp origOp) {
        auto inShape = getShape(origOp.getInput());
        auto outShape = getShape(origOp.getOutput());
        VPUX_THROW_UNLESS(outShape.totalSize() > inShape.totalSize(),
                          "Unexpect Expand input shape '{0}' output shape '{1}'", inShape, outShape);

        auto diffShapeSize = outShape.totalSize() - inShape.totalSize();

        const auto elemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
        if (mlir::isa<mlir::Float16Type>(elemType)) {
            maxFP16ShapeSize = std::max(checked_cast<int64_t>(diffShapeSize), maxFP16ShapeSize);

        } else if (const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
            const auto storageType = qType.getStorageType();
            if (storageType.isInteger(8)) {
                maxINT8ShapeSize = std::max(checked_cast<int64_t>(diffShapeSize), maxINT8ShapeSize);
            } else if (mlir::isa<mlir::Float8E4M3FNType>(storageType)) {
                maxFP8E4M3FNShapeSize = std::max(checked_cast<int64_t>(diffShapeSize), maxFP8E4M3FNShapeSize);
            } else if (mlir::isa<mlir::Float8E5M2Type>(storageType)) {
                maxFP8E5M2ShapeSize = std::max(checked_cast<int64_t>(diffShapeSize), maxFP8E5M2ShapeSize);
            } else {
                log.trace("Unexpected Expand '{0}' with quantized input storage type '{1}'", origOp->getLoc(),
                          storageType);
            }

        } else {
            log.trace("Unexpected Expand '{0}' with input type '{1}'", origOp->getLoc(), elemType);
        }

        log.trace("Found Expand Operation '{0}' with inshape: '{1}', outshape: '{2}', type: '{3}'", origOp->getLoc(),
                  inShape, outShape, elemType);
    });

    log.trace("Expand constant sizes:\n - FP16: {0}\n - INT8: {1}\n - FP8E4M3FN: {2}\n - FP8E5M2: {3}",
              maxFP16ShapeSize, maxINT8ShapeSize, maxFP8E4M3FNShapeSize, maxFP8E5M2ShapeSize);

    return {maxFP16ShapeSize, maxINT8ShapeSize, maxFP8E4M3FNShapeSize, maxFP8E5M2ShapeSize};
}

std::array<Const::DeclareOp, 4> ConvertExpandPass::getZeroConstOps(mlir::func::FuncOp func, mlir::MLIRContext& ctx,
                                                                   mlir::OpBuilder& builder) {
    const auto constantShapeSize = getMaxExpandConstShapes(func, _log);
    Const::DeclareOp constantFP16Op = nullptr;
    Const::DeclareOp constantINT8Op = nullptr;
    Const::DeclareOp constantFP8E4M3FNOp = nullptr;
    Const::DeclareOp constantFP8E5M2Op = nullptr;

    const auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "global_expand_const"));
    if (const auto size = constantShapeSize[0]; size != 0) {
        const auto dataFP16StorageType = mlir::RankedTensorType::get({size}, mlir::Float16Type::get(&ctx));
        const vpux::type::float16 value = 0.f;
        const auto denseFP16ElementVal = Const::createConstContent(dataFP16StorageType, ArrayRef(value));

        constantFP16Op = builder.create<Const::DeclareOp>(loc, vpux::convertToMemRef(dataFP16StorageType),
                                                          Const::ContentAttr::get(denseFP16ElementVal));
    }

    if (const auto size = constantShapeSize[1]; size != 0) {
        const auto dataQuantizeStorageType = mlir::RankedTensorType::get({size}, getUInt8Type(&ctx));
        constexpr auto value = uint8_t(0);
        const auto denseINT8ElementVal = Const::createConstContent(dataQuantizeStorageType, ArrayRef(value));

        constantINT8Op = builder.create<Const::DeclareOp>(loc, vpux::convertToMemRef(dataQuantizeStorageType),
                                                          Const::ContentAttr::get(denseINT8ElementVal));
    }

    if (const auto size = constantShapeSize[2]; size != 0) {
        const auto dataQuantizeStorageType = mlir::RankedTensorType::get({size}, mlir::Float8E4M3FNType::get(&ctx));
        const type::float8_e4m3 value = 0.f;
        const auto denseFP8E4M3FNElementVal = Const::createConstContent(dataQuantizeStorageType, ArrayRef(value));

        constantFP8E4M3FNOp = builder.create<Const::DeclareOp>(loc, vpux::convertToMemRef(dataQuantizeStorageType),
                                                               Const::ContentAttr::get(denseFP8E4M3FNElementVal));
    }

    if (const auto size = constantShapeSize[3]; size != 0) {
        const auto dataQuantizeStorageType = mlir::RankedTensorType::get({size}, mlir::Float8E5M2Type::get(&ctx));
        const type::float8_e5m2 value = 0.f;
        const auto denseFP8E5M2ElementVal = Const::createConstContent(dataQuantizeStorageType, ArrayRef(value));

        constantFP8E5M2Op = builder.create<Const::DeclareOp>(loc, vpux::convertToMemRef(dataQuantizeStorageType),
                                                             Const::ContentAttr::get(denseFP8E5M2ElementVal));
    }

    return {constantFP16Op, constantINT8Op, constantFP8E4M3FNOp, constantFP8E5M2Op};
}

Dim ConvertExpandPass::getPadDim(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType) {
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    const auto ioShapes = zip(inShape, outShape);
    const auto dimDiffPredicate = [](const std::tuple<int64_t, int64_t>& ioDims) -> bool {
        const auto& inDim = std::get<0>(ioDims);
        const auto& outDim = std::get<1>(ioDims);
        return inDim != outDim;
    };

    const auto diffAxisIter = std::find_if(ioShapes.begin(), ioShapes.end(), dimDiffPredicate);
    VPUX_THROW_UNLESS(diffAxisIter != ioShapes.end(), "Expand inShape '{0}' same with the outShape '{1}'", inShape,
                      outShape);

    const auto padAxis = std::distance(ioShapes.begin(), diffAxisIter);
    return Dim(padAxis);
}

//
// safeRunOnFunc
//

void ConvertExpandPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::OpBuilder builder(&func.getBody().front().front());

    // For Expand(FP16), Expand(FP8) and Expand(INT8) with PadsBegin, replace the op with concat a const op
    //     input                input      const
    //       |                    \          /
    //     Expand         =>         Concat
    //       |                         |
    // Note that only create one largest Constant Op and reuse for all Expand layers in the model
    // Always Set this Constant with 1D shape size, it is convenient to reshape for specific Expand
    // For Expand(U8) without PadsBegin, the op will be replaced by single DMA directly in later
    // pass(ConvertToDMA). The DMA solution does not support PadsBegin.

    auto constOps = getZeroConstOps(func, ctx, builder);

    func->walk([&](VPUIP::ExpandOp origOp) {
        _log.trace("Found Expand Operation '{0}'", origOp.getLoc());

        const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
        const auto elemType = inputType.getElementType();

        auto padBeginCheck = llvm::any_of(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()), [](auto padValue) {
            return padValue != 0;
        });
        if (!mlir::isa<mlir::Float16Type>(elemType) && !isFloat8Quantized(elemType) && !padBeginCheck) {
            _log.nest().trace(
                    "ExpandOp type should have float precision or integral precision with PadsBegin, but got '{0}'",
                    elemType);
            return;
        }

        mlir::Value constOutput = nullptr;
        if (mlir::isa<mlir::Float16Type>(elemType)) {
            constOutput = constOps[0].getOutput();
        } else if (const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
            const auto storageType = qElemType.getStorageType();
            if (storageType.isInteger(8)) {
                constOutput = constOps[1].getOutput();
            } else if (mlir::isa<mlir::Float8E4M3FNType>(storageType)) {
                constOutput = constOps[2].getOutput();
            } else if (mlir::isa<mlir::Float8E5M2Type>(storageType)) {
                constOutput = constOps[3].getOutput();
            }
        }
        VPUX_THROW_WHEN(constOutput == nullptr, "Missing constant definition for ExpandOp type : '{0}'", elemType);

        mlir::OpBuilder builder(origOp.getOperation());
        auto expandedBuffer =
                builder.create<mlir::memref::AllocOp>(origOp->getLoc(), mlir::cast<mlir::MemRefType>(outputType));

        const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
            return dim > 0;
        };

        SmallVector<mlir::Value> concatInputs;
        const auto inShape = inputType.getShape();
        auto subViewOffsets = SmallVector<int64_t>(inShape.size(), 0);
        PaddingContext padCtx(origOp->getLoc(), ShapeRef(inShape), expandedBuffer, constOutput);

        // Apply pads_begin
        _log.nest().trace("Process Expand Operation '{0}' for pads begin", origOp->getLoc());
        const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
        const auto padBeginAxisIter = std::find_if(padsBegin.begin(), padsBegin.end(), nonZeroAxisPredicate);
        if (padBeginAxisIter != padsBegin.end()) {
            const auto padBeginAxis = std::distance(padsBegin.begin(), padBeginAxisIter);
            const auto padValue = padsBegin[padBeginAxis];
            const auto padOut = applyPadding(padBeginAxis, padValue, subViewOffsets, padCtx, elemType, builder);
            concatInputs.push_back(padOut);
            subViewOffsets[padBeginAxis] += padValue;
        }

        // Copy the input with offset according to the padding in the beginning
        _log.nest().trace("Process Expand Operation '{0}' for real input data", origOp->getLoc());
        builder.setInsertionPoint(origOp);
        const auto tensorShape = to_small_vector(inShape);
        auto tensorSubView =
                builder.create<VPUIP::SubViewOp>(origOp.getLoc(), expandedBuffer, subViewOffsets, tensorShape);
        auto tensorSubViewCopy = builder.create<VPUIP::CopyOp>(origOp->getLoc(), origOp.getInput(), tensorSubView);

        concatInputs.push_back(tensorSubViewCopy.getOutput());

        // Increment offsets
        const auto padAxis = getPadDim(inputType, outputType);
        subViewOffsets[padAxis.ind()] += tensorShape[padAxis.ind()];

        // Apply pads_end
        _log.nest().trace("Process Expand Operation '{0}' for pads end", origOp->getLoc());
        const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
        const auto padEndAxisIter = std::find_if(padsEnd.begin(), padsEnd.end(), nonZeroAxisPredicate);
        if (padEndAxisIter != padsEnd.end()) {
            const auto padEndAxis = std::distance(padsEnd.begin(), padEndAxisIter);
            const auto padValue = padsEnd[padEndAxis];
            const auto padOut = applyPadding(padEndAxis, padValue, subViewOffsets, padCtx, elemType, builder);
            concatInputs.push_back(padOut);
        }

        auto concatViewOp = builder.create<VPUIP::ConcatViewOp>(origOp->getLoc(), concatInputs, expandedBuffer);
        _log.nest().trace("Create ConcatViewOp '{0}'", concatViewOp->getLoc());

        origOp->replaceAllUsesWith(concatViewOp);
        origOp->erase();
    });
}

}  // namespace

//
// createConvertExpandPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertExpandPass(Logger log) {
    return std::make_unique<ConvertExpandPass>(log);
}
