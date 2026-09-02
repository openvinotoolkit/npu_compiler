//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <array>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTEXPAND
#define GEN_PASS_DEF_CONVERTEXPAND
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool isUniformQuantizedInt8Storage(mlir::Type elemType) {
    const auto qElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    return qElemType != nullptr && qElemType.getStorageType().isInteger(8) && qElemType.getZeroPoint() == 0;
}

bool isUniformQuantizedFloat8Storage(mlir::Type elemType) {
    const auto qElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    return qElemType != nullptr && isFloat8(qElemType.getStorageType()) && qElemType.getZeroPoint() == 0;
}

// Element-type buckets for the per-type shared zero constant.
enum ExpandConstKind : size_t { FP16 = 0, I8, U8, FP8E4M3FN, FP8E5M2, FP32 };

constexpr std::array<ExpandConstKind, 6> SUPPORTED_EXPAND_CONST_KINDS = {
        ExpandConstKind::FP16,      ExpandConstKind::I8,      ExpandConstKind::U8,
        ExpandConstKind::FP8E4M3FN, ExpandConstKind::FP8E5M2, ExpandConstKind::FP32};
constexpr size_t EXPAND_CONST_KIND_COUNT = SUPPORTED_EXPAND_CONST_KINDS.size();

constexpr size_t getExpandConstKindIndex(ExpandConstKind kind) {
    return static_cast<size_t>(kind);
}

// Per-bucket constant metadata. `maxSize` is the largest expansion (= longest pad slice
// we'll need to copy out); `storageType` preserves observed int8 storage signedness.
struct ExpandConstBucketInfo {
    int64_t maxSize = 0;
    mlir::Type storageType = nullptr;
};

using ExpandConstInfo = std::array<ExpandConstBucketInfo, EXPAND_CONST_KIND_COUNT>;
using ExpandConstOps = std::array<Const::DeclareOp, EXPAND_CONST_KIND_COUNT>;

ExpandConstKind getExpandConstKind(mlir::Type elemType) {
    if (mlir::isa<mlir::Float16Type>(elemType)) {
        return ExpandConstKind::FP16;
    }
    if (mlir::isa<mlir::Float32Type>(elemType)) {
        return ExpandConstKind::FP32;
    }

    const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType);
    VPUX_THROW_UNLESS(qType != nullptr, "Unsupported Expand input type '{0}'", elemType);

    const auto storageType = qType.getStorageType();
    if (storageType.isInteger(8)) {
        return storageType.isUnsignedInteger(8) ? ExpandConstKind::U8 : ExpandConstKind::I8;
    }
    if (mlir::isa<mlir::Float8E4M3FNType>(storageType)) {
        return ExpandConstKind::FP8E4M3FN;
    }
    if (mlir::isa<mlir::Float8E5M2Type>(storageType)) {
        return ExpandConstKind::FP8E5M2;
    }

    VPUX_THROW("Unsupported Expand quantized storage type '{0}'", storageType);
}

bool isSupportedExpandElementType(mlir::Type elemType) {
    if (mlir::isa<mlir::Float16Type, mlir::Float32Type>(elemType) || isUniformQuantizedFloat8Storage(elemType)) {
        return true;
    }

    return isUniformQuantizedInt8Storage(elemType);
}

// Helper class to wrap the arguments for ExpandConverter::applyPadding
class PaddingContext {
public:
    PaddingContext(const mlir::Location loc, const ShapeRef inShape, const mlir::Value expandedBuffer,
                   const mlir::Value constantBuffer)
            : loc(loc), inShape(inShape), expandedBuffer(expandedBuffer), constantBuffer(constantBuffer) {};
    PaddingContext(const PaddingContext&) = delete;
    PaddingContext(const PaddingContext&&) = delete;
    PaddingContext& operator=(const PaddingContext&) = delete;
    PaddingContext& operator=(const PaddingContext&&) = delete;
    ~PaddingContext() = default;

    const mlir::Location loc;
    ShapeRef inShape;
    const mlir::Value expandedBuffer;
    const mlir::Value constantBuffer;
};

//
// ConvertExpandPass
//

class ConvertExpandPass final : public VPUIP::impl::ConvertExpandBase<ConvertExpandPass> {
public:
    ConvertExpandPass(bool deferToExpandDMAArg, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        this->deferToExpandDMA = deferToExpandDMAArg;
    }

private:
    void safeRunOnFunc() final;

    mlir::Value applyPadding(const int64_t padAxis, const int64_t padValue, ArrayRef<int64_t> inSubViewOffsets,
                             const PaddingContext& padCtx, mlir::Type expectedElemType, mlir::OpBuilder& builder) const;

    bool shouldDeferToExpandDMA(mlir::Type elemType, mlir::ArrayAttr padsBegin) const;

    ExpandConstInfo collectExpandConstInfo(mlir::func::FuncOp func, Logger log);
    ExpandConstOps getZeroConstOps(mlir::func::FuncOp func, mlir::MLIRContext& ctx, mlir::OpBuilder& builder);

    Dim getPadDim(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType);
};

bool hasZeroPadsBegin(mlir::ArrayAttr padsBegin) {
    return llvm::all_of(parseIntArrayAttr<int64_t>(padsBegin), [](auto padValue) {
        return padValue == 0;
    });
}

// ConvertToDMA lowers integral ExpandOp to VPUIP.ExpandDMA. The descriptor generator supports only
// padding at the end, so this pass leaves only all-zero `pads_begin` cases for that lowering.
// When `deferToExpandDMA` is false, every supported case is decomposed here regardless.
bool ConvertExpandPass::shouldDeferToExpandDMA(mlir::Type elemType, mlir::ArrayAttr padsBegin) const {
    if (!deferToExpandDMA) {
        return false;
    }
    if (!hasZeroPadsBegin(padsBegin)) {
        return false;
    }

    return !mlir::isa<mlir::FloatType>(elemType) && !isLowFpTypeQuantized(elemType);
}

mlir::Value ConvertExpandPass::applyPadding(const int64_t padAxis, const int64_t padValue,
                                            ArrayRef<int64_t> inSubViewOffsets, const PaddingContext& padCtx,
                                            mlir::Type expectedElemType, mlir::OpBuilder& builder) const {
    const auto& location = padCtx.loc;
    const auto& inShape = padCtx.inShape;
    const auto& expandedBuffer = padCtx.expandedBuffer;
    const auto& constantBuffer = padCtx.constantBuffer;
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
            builder.create<VPUIP::SubViewOp>(appendLoc(location, "constant_subview_{0}_{1}", padAxis, padValue),
                                             constantOp, constSubviewOffset, constSubviewShape);

    // Step 2: Create Reshape Op to match concat shape with expected type
    const auto shapeType = mlir::cast<NDTypeInterface>(constSubView.getType());
    auto newShapeType = shapeType.changeShape(subViewShape);
    if (isUniformQuantizedFloat8Storage(expectedElemType) || isUniformQuantizedInt8Storage(expectedElemType)) {
        // Reinterpret the storage-typed constant to the expected quantized element type.
        newShapeType = newShapeType.changeElemType(expectedElemType);
    }
    auto reshapeOp = builder.create<VPUIP::GenericReshapeOp>(
            appendLoc(location, "constant_reshape_{0}_{1}", padAxis, padValue), newShapeType, constSubView.getResult());

    // Step 3: Create PermuteCast Op to match concat layout
    const auto expandOutBufferType = mlir::cast<NDTypeInterface>(expandedBuffer.getType());
    const auto newLayoutType = newShapeType.changeDimsOrder(expandOutBufferType.getDimsOrder());
    const auto dstOrderAttr =
            mlir::AffineMapAttr::get(expandOutBufferType.getDimsOrder().toAffineMap(reshapeOp.getContext()));
    const auto memPermAttr = mlir::AffineMapAttr::get(
            DimsOrder::fromNumDims(mlir::cast<NDTypeInterface>(reshapeOp.getOutput().getType()).getRank())
                    .toAffineMap(reshapeOp.getContext()));
    auto permuteCastOp =
            builder.create<VPUIP::PermuteCastOp>(appendLoc(location, "constant_permute_{0}_{1}", padAxis, padValue),
                                                 newLayoutType, reshapeOp.getOutput(), dstOrderAttr, memPermAttr);

    // Step 4: Create Copy Op for concat concatant input
    auto subView = builder.create<VPUIP::SubViewOp>(appendLoc(location, "expand_subview_{0}_{1}", padAxis, padValue),
                                                    expandedBuffer, ShapeRef(subViewOffsets), subViewShape);
    auto subViewCopy = builder.create<VPUIP::CopyOp>(appendLoc(location, "expand_copy_{0}_{1}", padAxis, padValue),
                                                     permuteCastOp.getResult(), subView);

    return subViewCopy.getOutput();
}

ExpandConstInfo ConvertExpandPass::collectExpandConstInfo(mlir::func::FuncOp func, Logger log) {
    ExpandConstInfo info{};

    func->walk([&](VPUIP::ExpandOp origOp) {
        auto inShape = getShape(origOp.getInput());
        auto outShape = getShape(origOp.getOutput());
        VPUX_THROW_UNLESS(outShape.totalSize() > inShape.totalSize(),
                          "Unexpect Expand input shape '{0}' output shape '{1}'", inShape, outShape);

        const auto diffShapeSize = checked_cast<int64_t>(outShape.totalSize() - inShape.totalSize());
        const auto elemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();

        if (shouldDeferToExpandDMA(elemType, origOp.getPadsBegin())) {
            log.trace("Skipping Expand bucket bump for '{0}': leaving for ConvertToDMA -> ExpandDMA", origOp->getLoc());
            return;
        }

        const auto bumpBucket = [&](ExpandConstKind kind, mlir::Type storageType = nullptr) {
            auto& bucketInfo = info.at(getExpandConstKindIndex(kind));
            bucketInfo.maxSize = std::max(diffShapeSize, bucketInfo.maxSize);
            if (storageType == nullptr) {
                return;
            }

            if (bucketInfo.storageType == nullptr) {
                bucketInfo.storageType = storageType;
                return;
            }

            VPUX_THROW_UNLESS(bucketInfo.storageType == storageType,
                              "Found Expand ops in the same constant bucket with different storage types '{0}' and "
                              "'{1}'",
                              bucketInfo.storageType, storageType);
        };

        if (mlir::isa<mlir::Float16Type>(elemType)) {
            bumpBucket(ExpandConstKind::FP16);
        } else if (mlir::isa<mlir::Float32Type>(elemType)) {
            bumpBucket(ExpandConstKind::FP32);
        } else if (isUniformQuantizedInt8Storage(elemType)) {
            const auto storageType = mlir::cast<mlir::quant::QuantizedType>(elemType).getStorageType();
            bumpBucket(getExpandConstKind(elemType), storageType);
        } else if (isUniformQuantizedFloat8Storage(elemType)) {
            bumpBucket(getExpandConstKind(elemType));
        } else if (const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
            const auto storageType = qType.getStorageType();
            log.trace("Unexpected Expand '{0}' with unsupported quantized input type '{1}' and storage type '{2}'",
                      origOp->getLoc(), elemType, storageType);
        } else {
            log.trace("Unexpected Expand '{0}' with input type '{1}'", origOp->getLoc(), elemType);
        }

        log.trace("Found Expand Operation '{0}' with inshape: '{1}', outshape: '{2}', type: '{3}'", origOp->getLoc(),
                  inShape, outShape, elemType);
    });

    log.trace("Expand constant sizes:\n - FP16: {0}\n - FP32: {1}\n - I8: {2}\n - U8: {3}\n - FP8E4M3FN: {4}\n - "
              "FP8E5M2: {5}",
              info[getExpandConstKindIndex(ExpandConstKind::FP16)].maxSize,
              info[getExpandConstKindIndex(ExpandConstKind::FP32)].maxSize,
              info[getExpandConstKindIndex(ExpandConstKind::I8)].maxSize,
              info[getExpandConstKindIndex(ExpandConstKind::U8)].maxSize,
              info[getExpandConstKindIndex(ExpandConstKind::FP8E4M3FN)].maxSize,
              info[getExpandConstKindIndex(ExpandConstKind::FP8E5M2)].maxSize);

    return info;
}

ExpandConstOps ConvertExpandPass::getZeroConstOps(mlir::func::FuncOp func, mlir::MLIRContext& ctx,
                                                  mlir::OpBuilder& builder) {
    const auto constInfo = collectExpandConstInfo(func, _log);
    ExpandConstOps ops{};

    const auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "global_expand_const"));

    // The bucket constant carries plain storage matching the Expand storage type; each
    // Expand op later reinterprets it to its own quantized element type via `changeElemType`.
    const auto declareBucketConst = [&](mlir::Type storageType, auto zeroValue, int64_t maxSize) {
        const auto tensorType = mlir::RankedTensorType::get({maxSize}, storageType);
        const auto denseAttr = Const::createConstContent(tensorType, ArrayRef(zeroValue));
        return builder.create<Const::DeclareOp>(loc, vpux::convertToMemRef(tensorType),
                                                Const::ContentAttr::get(denseAttr));
    };

    for (const auto kind : SUPPORTED_EXPAND_CONST_KINDS) {
        const auto kindIdx = getExpandConstKindIndex(kind);
        const auto& bucketInfo = constInfo.at(kindIdx);
        const auto maxSize = bucketInfo.maxSize;
        if (maxSize == 0) {
            continue;
        }
        switch (kind) {
        case ExpandConstKind::FP16:
            ops.at(kindIdx) = declareBucketConst(mlir::Float16Type::get(&ctx), vpux::type::float16(0.f), maxSize);
            break;
        case ExpandConstKind::FP32:
            ops.at(kindIdx) = declareBucketConst(mlir::Float32Type::get(&ctx), static_cast<float>(0.f), maxSize);
            break;
        case ExpandConstKind::I8:
            VPUX_THROW_UNLESS(bucketInfo.storageType != nullptr,
                              "Can not get storage type for I8 quantized Expand constant");
            ops.at(kindIdx) = declareBucketConst(bucketInfo.storageType, int8_t{0}, maxSize);
            break;
        case ExpandConstKind::U8:
            VPUX_THROW_UNLESS(bucketInfo.storageType != nullptr,
                              "Can not get storage type for U8 quantized Expand constant");
            ops.at(kindIdx) = declareBucketConst(bucketInfo.storageType, uint8_t{0}, maxSize);
            break;
        case ExpandConstKind::FP8E4M3FN:
            ops.at(kindIdx) =
                    declareBucketConst(mlir::Float8E4M3FNType::get(&ctx), vpux::type::float8_e4m3(0.f), maxSize);
            break;
        case ExpandConstKind::FP8E5M2:
            ops.at(kindIdx) =
                    declareBucketConst(mlir::Float8E5M2Type::get(&ctx), vpux::type::float8_e5m2(0.f), maxSize);
            break;
        }
    }

    return ops;
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

    // Replace each supported VPUIP.Expand that cannot be handled by ConvertToDMA with a SubView+Copy chain over a
    // single shared zero-constant per element-type bucket.
    //     input                input      const
    //       |                    \          /
    //     Expand         =>         Concat
    //       |                         |

    // Pre-walk all Expand ops to find, per element-type bucket, the largest expansion
    // (= longest pad slice we'll need to copy out). Then emit one shared 1D zero
    // `Const::DeclareOp` per non-empty bucket, sized to the bucket's max.
    auto constOps = getZeroConstOps(func, ctx, builder);

    func->walk([&](VPUIP::ExpandOp origOp) {
        _log.trace("Found Expand Operation '{0}'", origOp.getLoc());

        const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
        const auto elemType = inputType.getElementType();

        if (shouldDeferToExpandDMA(elemType, origOp.getPadsBegin())) {
            _log.nest().trace("Skipping ExpandOp with element type '{0}' and all-zero pads_begin: "
                              "ConvertToDMA will lower it to ExpandDMA.",
                              elemType);
            return;
        }

        const bool isSupportedElemType = isSupportedExpandElementType(elemType);
        if (!isSupportedElemType) {
            _log.nest().trace("ExpandOp element type unsupported: '{0}' (expected FP16, FP8 quantized with zero "
                              "point 0, or integral type with all-zero pads_begin for ConvertToDMA)",
                              elemType);
            return;
        }

        const auto constKind = getExpandConstKind(elemType);
        auto constOutput = constOps[getExpandConstKindIndex(constKind)].getOutput();
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

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertExpandPass(bool lowerToExpandDMA, Logger log) {
    return std::make_unique<ConvertExpandPass>(lowerToExpandDMA, log);
}
