//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/SymbolTable.h>

namespace vpux::Const {

namespace {
// Note: some users create *memref* constants; however, as per DenseElementsAttr
// documentation, one cannot create dense<> memrefs, so this set of functions
// ensures we deal with tensors where necessary -- making this internally
// simplifies the job for the user as one does not have to specify custom
// conversions or supply 2 types instead of 1.
mlir::RankedTensorType ensureRankedTensor(mlir::RankedTensorType type) {
    return type;
}
mlir::RankedTensorType ensureRankedTensor(mlir::MemRefType type) {
    return mlir::cast<mlir::RankedTensorType>(reconstructTensorType(type));
}

template <typename TensorOrMemref>
mlir::Value createZerosConstImpl(mlir::OpBuilder& builder, mlir::Location loc, TensorOrMemref type) {
    const auto elemType = type.getElementType();

    mlir::DenseElementsAttr denseElementVal = nullptr;
    if (const auto uniformElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType)) {
        const auto quantizedType = type.cloneWith(type.getShape(), normalizeQuantStorageType(uniformElemType));
        const auto quantizedTensorType = ensureRankedTensor(quantizedType);
        const auto zeroPoint = uniformElemType.getZeroPoint();
        if (uniformElemType.isSigned()) {
            denseElementVal = createConstContent(quantizedTensorType, ArrayRef(checked_cast<int8_t>(zeroPoint)));
        } else {
            denseElementVal = createConstContent(quantizedTensorType, ArrayRef(checked_cast<uint8_t>(zeroPoint)));
        }
    } else {
        denseElementVal = createConstContent(ensureRankedTensor(type), ArrayRef(0.f));
    }

    VPUX_THROW_WHEN(
            denseElementVal == nullptr,
            "Upsampling has incompatible data type {0}, only float16, float32 or uniform quantized type are supported",
            elemType);

    return builder.create<Const::DeclareOp>(loc, type, Const::ContentAttr::get(denseElementVal)).getOutput();
}
}  // namespace

mlir::StringRef getOvKey(Const::DeclareOp declareOp) {
    auto key = getResourceName(declareOp.getContentAttr().getBaseContent());
    static_assert(std::is_same_v<mlir::StringRef, decltype(key)>,
                  "Cannot return StringRef if the underlying getResourceName() doesn't return it - potential dangling "
                  "reference otherwise");
    if (key.starts_with(Const::OPENVINO_CONST_PREFIX)) {
        return key;
    }
    return {};
}

bool isOpenVINOConstant(Const::DeclareOp declareOp) {
    return !getOvKey(declareOp).empty();
}

mlir::Value createZerosConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type) {
    return createZerosConstImpl(builder, loc, type);
}

mlir::Value createZerosConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::MemRefType type) {
    return createZerosConstImpl(builder, loc, type);
}

mlir::Value createFloatConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type,
                             ArrayRef<float> values) {
    const auto constShape = type.getShape();
    const auto shapeTotalSize = vpux::details::calcTotalShapeSize(constShape);
    VPUX_THROW_UNLESS(values.size() == 1 || shapeTotalSize == checked_cast<int64_t>(values.size()),
                      "Create float Const failed with unexpect data size");

    const auto denseElementVal = createConstContent(type, values);
    VPUX_THROW_UNLESS(denseElementVal != nullptr, "Incompatible data type {0}, only float16 or float32 are supported",
                      type.getElementType());

    return builder.create<Const::DeclareOp>(loc, type, Const::ContentAttr::get(denseElementVal)).getOutput();
}

Const::ContentAttr createFloatContentAttr(mlir::OpBuilder&, mlir::Location, mlir::RankedTensorType type,
                                          ArrayRef<float> values) {
    const auto constShape = type.getShape();
    const auto shapeTotalSize =
            std::accumulate(constShape.begin(), constShape.end(), int64_t(1), std::multiplies<int64_t>());
    VPUX_THROW_UNLESS(values.size() == 1 || shapeTotalSize == checked_cast<int64_t>(values.size()),
                      "Create float Const failed with unexpect data size");

    const auto denseElementVal = createConstContent(type, values);
    VPUX_THROW_UNLESS(denseElementVal != nullptr, "Incompatible data type {0}, only float16 or float32 are supported",
                      type.getElementType());

    return Const::ContentAttr::get(denseElementVal);
}

bool hasNegativeValues(const Const::Content& content) {
    if (content.isSplat()) {
        return content.getSplatValue<double>() < 0.0;
    }

    const auto vals = content.getValues<double>();
    return std::any_of(vals.begin(), vals.end(), [](double val) {
        return val < 0.0;
    });
}

mlir::Value buildWeightsConst(mlir::OpBuilder& builder, mlir::Location loc, mlir::RankedTensorType type,
                              ArrayRef<float> values) {
    const auto ctx = builder.getContext();
    const auto origElemType = type.getElementType();

    mlir::Type filterElemType = mlir::Float16Type::get(ctx);
    if (const auto qInputElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(origElemType)) {
        const auto scale = 1.0f;
        const auto zeroPoint = 0;

        if (vpux::isFloat8Quantized(qInputElemType)) {
            filterElemType = mlir::quant::UniformQuantizedType::get(
                    /*flags=*/0, /*storageType=*/qInputElemType.getStorageType(),
                    /*expressedType=*/mlir::Float16Type::get(ctx),
                    /*scale=*/scale, /*zeroPoint=*/zeroPoint, /*storageTypeMin=*/qInputElemType.getStorageTypeMin(),
                    /*storageTypeMax=*/qInputElemType.getStorageTypeMax());

        } else if (qInputElemType.getStorageType().isInteger(8)) {
            filterElemType = mlir::quant::UniformQuantizedType::get(
                    0, getUInt8Type(ctx), mlir::Float16Type::get(ctx), scale, zeroPoint,
                    std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

        } else {
            VPUX_THROW("Unsupported quantized storage type: {0}", qInputElemType.getStorageType());
        }
    }

    const auto dataType = mlir::RankedTensorType::get(type.getShape(), mlir::Float32Type::get(ctx));
    const auto dataAttr = createConstContent(dataType, values);

    Const::ContentSetup contentAttrSetup(dataType);
    VPUX_THROW_WHEN(!(mlir::isa<mlir::quant::QuantizedType, mlir::Float16Type>(origElemType)), "Unsupported type {0}",
                    origElemType);
    if (auto qElemType = filterElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        contentAttrSetup = contentAttrSetup.castElemType(qElemType);
    } else if (origElemType.isa<mlir::Float16Type>()) {
        contentAttrSetup = contentAttrSetup.castElemType(mlir::Float16Type::get(ctx));
    }
    contentAttrSetup = contentAttrSetup.reorder(mlir::cast<NDTypeInterface>(type).getDimsOrder());
    auto contentAttr = Const::ContentAttr::get(dataAttr, std::move(contentAttrSetup));

    return builder.create<Const::DeclareOp>(loc, contentAttr.getType(), std::move(contentAttr)).getOutput();
}

SmallVector<Const::DeclareOp> getDeclareOpsUses(Const::RodataOp rodataOp, mlir::Operation* from) {
    auto usesOpt = rodataOp.getSymbolUses(from);

    if (!usesOpt.has_value()) {
        return {};
    }

    auto uses = usesOpt.value();

    return to_small_vector(uses | transformed([](auto use) {
                               return mlir::dyn_cast_or_null<Const::DeclareOp>(use.getUser());
                           }) |
                           filtered([](Const::DeclareOp declareOp) {
                               return declareOp != nullptr;
                           }));
}

SmallVector<Const::DeclareOp> getDeclareOpsUses(mlir::SymbolRefAttr symbol, mlir::ModuleOp from) {
    auto op = mlir::SymbolTable::lookupSymbolIn(from, symbol);
    auto rodataOp = llvm::dyn_cast_or_null<Const::RodataOp>(op);

    if (rodataOp == nullptr) {
        return {};
    }

    return getDeclareOpsUses(rodataOp, from);
}

void foldSingleConstant(Const::DeclareOp& origOp) {
    const auto content = origOp.getContent();
    const auto contentType = content.getType();
    const auto contentElemType = contentType.getElementType();

    const auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
    std::vector<char> tempBuf(bufSize);
    content.copyTo(MutableArrayRef(tempBuf.data(), bufSize));

    auto rankedTensorType = contentType.cast<mlir::RankedTensorType>();

    const auto elemTypeBitSize = contentType.getElemTypeSize().count();
    // As of now sub byte types are not supported as DenseElementsAttr storage, I1 is an exception
    const auto isUnsupportedSubByteStorageType = elemTypeBitSize < CHAR_BIT && elemTypeBitSize > 1;
    if (isUnsupportedSubByteStorageType) {
        rankedTensorType = contentType
                                   .changeShapeElemType(Shape({1, 1, 1, checked_cast<int32_t>(bufSize)}),
                                                        getUInt8Type(contentType.getContext()))
                                   .cast<mlir::RankedTensorType>();
    } else if (auto qtype = contentElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        rankedTensorType = contentType.changeElemType(normalizeQuantStorageType(qtype)).cast<mlir::RankedTensorType>();
    }

    const auto denseAttr = Const::createConstContent(rankedTensorType, tempBuf);
    auto origType = origOp.getType().cast<NDTypeInterface>();

    if (isUnsupportedSubByteStorageType) {
        // Temporary fix to enable compilation.
        // Final design to also include a mechanism to FREEZE constants
        // from accepting future transformations due to the fact of packed
        // sub byte values stored, which would require an unpacking and a repacking
        origOp.getProperties().content = Const::ContentAttr::get(
                denseAttr, Const::ContentSetup(denseAttr.getType())
                                   .changeShapeAndElemType(origType.getShape(), origType.getElementType()));
    } else {
        origOp.getProperties().content = Const::ContentAttr::get(denseAttr);
    }
}

void appendContentToVector(Const::Content& content, MutableArrayRef<char> buffer, size_t& start) {
    const auto bufSizeBytes = checked_cast<size_t>(content.getType().getTotalAllocSize().count());
    auto* oldEnd = buffer.data() + start;
    VPUX_THROW_UNLESS(start + bufSizeBytes <= buffer.size(),
                      "Overflow during fusing buffer size {0}, size after copying {1}", buffer.size(),
                      start + bufSizeBytes);
    MutableArrayRef<char> newBufferSlice(reinterpret_cast<char*>(oldEnd), bufSizeBytes);
    content.copyTo(newBufferSlice);
    start += bufSizeBytes;
}

}  // namespace vpux::Const
