//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/attributes/stable_hash_storage.hpp"
#include "vpux/compiler/dialect/const/constant_call_stack.hpp"
#include "vpux/compiler/dialect/const/constant_transformations_control.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/const_logger.hpp"
#include "vpux/compiler/utils/stable_hash.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/DialectResourceBlobManager.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>

#include <cassert>
#include <cstring>
#include <utility>

using namespace vpux;

namespace {

//
// ConstInlinerInterface
//

struct ConstInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};

/// @brief Caches splatness status for dense_resource<> blobs (for which it is
/// expensive to calculate manually).
class SplatnessCache final : public mlir::DialectInterface::Base<SplatnessCache> {
public:
    using ValueType = std::optional<std::pair<mlir::ArrayRef<char>, bool>>;

    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplatnessCache)

    SplatnessCache(mlir::Dialect* dialect): Base(dialect) {
    }

    void cacheRawDataAndSplatness(mlir::DenseResourceElementsAttr denseResource);
    ValueType getRawDataAndSplatness(mlir::DenseResourceElementsAttr denseResource);

private:
    mlir::DenseMap<mlir::DenseResourceElementsAttr, ValueType> _cache;
    std::recursive_mutex _cacheMutex{};  // Note: recursive because "get" can call "cache"
};

template <typename Cache>
Cache& getCache(mlir::MLIRContext* ctx) {
    return vpux::getCache<Cache, vpux::Const::ConstDialect>(ctx);
}

/// @brief Caches LazyFoldingOptions inside of the MLIRContext to be able to
/// access them throughout the folding procedure.
class LazyFoldingCache final : public mlir::DialectInterface::Base<LazyFoldingCache> {
    Const::LazyFoldingOptions _options;

public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LazyFoldingCache)

    LazyFoldingCache(mlir::Dialect* dialect): Base(dialect) {
    }

    const Const::LazyFoldingOptions& getOptions() const {
        return _options;
    }

    void setOptions(Const::LazyFoldingOptions options) {
        _options = std::move(options);
    }
};

size_t getByteSize(mlir::ShapedType type) {
    if (isSubByteType(type)) {
        return 1;
    }

    const auto bitWidth = getElemTypeSize(type.getElementType());
    assert(bitWidth.count() % CHAR_BIT == 0 && "If a type is not a sub-byte, only byte-aligned types are supported");
    return checked_cast<size_t>(Byte{bitWidth}.count());
}

}  // namespace

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/const/attributes.cpp.inc>

//
// ConstDialect::initialize
//

void vpux::Const::ConstDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/const/ops.cpp.inc>
            >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/const/attributes.cpp.inc>
            >();

    addInterfaces<ConstInlinerInterface>();
    addInterfaces<SplatnessCache, LazyFoldingCache>();

    if (isDeveloperBuild()) {
        addInterfaces<CallStackCache>();
    }
}

//
// ContentAttr::verify
//

mlir::LogicalResult vpux::Const::ContentAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ElementsAttr baseContent,
                                                     vpux::Const::TransformAttrInterfaceArrayAttr transformations,
                                                     vpux::NDTypeInterface, mlir::UnitAttr) {
    if (baseContent == nullptr) {
        return printTo(emitError(), "Got NULL 'baseContent' in 'ContentAttr'");
    }

    auto baseContentElemType = baseContent.getShapedType().getElementType();

    // Note: base content element type must be directly aligned to a respective
    // C++ type that is supported by NPU compiler (and also by MLIR!). Thus, any
    // quantized type is impossible to be specified as base content (aka raw
    // data). In general, consult Content::dispatchByElemType() to understand
    // which types are directly supported in C++. For instance, quantized types
    // are NOT supported.
    if (!baseContentElemType.isIntOrFloat()) {
        return printTo(emitError(), "Got unsupported element type for 'baseContent' in 'ContentAttr' : '{0}'",
                       baseContent.getShapedType().getElementType());
    }

    if (!mlir::isa<mlir::DenseElementsAttr, mlir::DenseResourceElementsAttr, Const::SymElementsAttr>(baseContent)) {
        return printTo(emitError(), "Got unsupported 'baseContent' in 'ContentAttr'");
    }

    if (auto denseResource = mlir::dyn_cast<mlir::DenseResourceElementsAttr>(baseContent)) {
        auto blob = denseResource.getRawHandle().getBlob();
        // If blob is null we might be in the IR parsing scenario where ContentAttr is parsed before
        // dialect_resources section. For such case Const::DeclareOp::verify will verify dense resource content
        if (blob != nullptr && mlir::failed(verifyDenseResource(emitError, denseResource))) {
            return mlir::failure();
        }
    }

    const auto isValid = [](const vpux::Const::TransformAttrInterface& value) -> bool {
        return value != nullptr;
    };
    if (!llvm::all_of(transformations.getValue(), isValid)) {
        return printTo(emitError(), "Got invalid transformations attribute in 'ContentAttr'");
    }

    return mlir::success();
}

mlir::LogicalResult vpux::Const::ContentAttr::verifyDenseResource(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                                  mlir::DenseResourceElementsAttr denseResource) {
    if (denseResource == nullptr) {
        return printTo(emitError(), "Got NULL 'denseResource' in 'ContentAttr'");
    }

    auto cacheEntry = ::getCache<SplatnessCache>(denseResource.getContext()).getRawDataAndSplatness(denseResource);
    if (!cacheEntry.has_value()) {
        return printTo(emitError(), "Can't access constant content cache for verification, resource handle : {0}",
                       denseResource.getRawHandle().getKey());
    }

    auto [rawData, isSplat] = cacheEntry.value();

    // Note: manual checks required since dense resource blob is opaque and does not perform much validation itself
    const auto type = denseResource.getShapedType();
    const auto splatSize = getByteSize(type);
    const auto expectedSize = isSplat ? splatSize : getExpectedBufferSize(type).count();

    const auto actualSize = rawData.size();
    if (actualSize != checked_cast<size_t>(expectedSize)) {
        return printTo(emitError(), "Size of dense resource buffer '{0}' in 'baseContent' doesn't match its type '{1}'",
                       actualSize, type);
    }

    return mlir::success();
}

namespace {

bool isSplat(mlir::ShapedType type, mlir::ArrayRef<char> data) {
    if (data.empty()) {
        return false;
    }

    const auto elementSizeInBytes = getByteSize(type);
    const auto expectedBufferSize = static_cast<size_t>(getExpectedBufferSize(type).count());
    const auto dataHasExpectedSize = data.size() == elementSizeInBytes || data.size() == expectedBufferSize;
    if (!dataHasExpectedSize) {
        // don't assert on this, tests expect verifier to catch this error
        // verifier check is called after this function
        return false;
    }

    // check `data` has all elements of the same value via shifted-equal
    // store range now to skip last padded byte later in case of sub-byte data type
    auto rangeForCompare = data;

    if (isSubByteType(type)) {
        const auto bitWidth = getElemTypeSize(type.getElementType());
        if (CHAR_BIT % bitWidth.count() != 0) {
            // splatness check is unsupported for non-power-of-2 sub-byte types
            // PSS still triggers this, so assert is not an option here
            // treat as non-splat buffer for safety
            return false;
        }

        const auto numElements = type.getNumElements();
        const auto elementsPerByte = CHAR_BIT / bitWidth.count();

        // `splatByte` is one sub-byte element repeated until end of the byte
        const auto splatByte = [&] {
            const auto mask = (int64_t{1} << bitWidth.count()) - 1;
            const auto firstElement = data.front() & mask;

            char splat = 0;
            for (auto i : irange(elementsPerByte)) {
                splat |= static_cast<char>(firstElement << (i * bitWidth.count()));
            }
            return splat;
        }();

        // returns true iff `byte` is splat with `count` elements inside
        const auto isSplatByte = [splatByte, bitWidth](char byte, size_t count) {
            // use mask to ignore possible padding bits
            const auto mask = (int64_t{1} << (count * bitWidth.count())) - 1;
            return (byte & mask) == (splatByte & mask);
        };

        const auto elementsInFirstByte = std::min(numElements, elementsPerByte);
        if (!isSplatByte(data.front(), elementsInFirstByte)) {
            return false;
        }

        const auto elementsInLastByte = numElements % elementsPerByte;
        // check data has more than one byte to avoid emptying rangeForCompare via drop_back;
        // single-byte buffers are fully covered by the first-byte check above
        if (data.size() > 1 && elementsInLastByte > 0) {
            if (!isSplatByte(data.back(), elementsInLastByte)) {
                return false;
            }
            // skip last padded byte as it is legally not equal to other bytes
            rangeForCompare = data.drop_back(1);
        }
    }

    // shift the range for comparison, such that `equal` compares adjacent elements
    // byte-wise; for multi-byte elements this means 'high byte' is compared
    // to 'high byte', and 'low byte' is compared to 'low byte'
    assert(!rangeForCompare.empty());
    const auto shiftedRange = rangeForCompare.drop_front(elementSizeInBytes);
    return std::equal(shiftedRange.begin(), shiftedRange.end(), rangeForCompare.begin());
}

std::pair<mlir::ArrayRef<char>, bool> detectSplatManually(mlir::ShapedType type, mlir::ArrayRef<char> data) {
    if (!isSplat(type, data)) {
        return {data, false};
    }

    return {data.take_front(getByteSize(type)), true};
}

/// Returns pointer to baseContent's data and whether the data is splat.
SplatnessCache::ValueType getRawDataAndSplatness(mlir::ElementsAttr baseContent) {
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(baseContent)) {
        return std::pair{dense.getRawData(), dense.isSplat()};
    }

    // We cannot know if we have a splat value because we cannot dereference the symbol from here.
    if (mlir::isa<Const::SymElementsAttr>(baseContent)) {
        return std::pair{mlir::ArrayRef<char>(), false};
    }

    auto denseResource = mlir::cast<mlir::DenseResourceElementsAttr>(baseContent);
    return ::getCache<SplatnessCache>(baseContent.getContext()).getRawDataAndSplatness(denseResource);
}

void SplatnessCache::cacheRawDataAndSplatness(mlir::DenseResourceElementsAttr denseResource) {
    // dense resource doesn't support splat detection in MLIR itself
    auto blob = denseResource.getRawHandle().getBlob();
    if (blob != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(_cacheMutex);
        auto& entry = _cache[denseResource];

        // Note: In an unlikely but possible event, the same dense_resource<>
        // can already be cached (OV model is compressed). In this case, there
        // is no need to run the possibly expensive calculation again.
        if (entry.has_value()) {
            return;
        }

        entry = detectSplatManually(denseResource.getShapedType(), blob->getData());
    }
}

typename SplatnessCache::ValueType SplatnessCache::getRawDataAndSplatness(
        mlir::DenseResourceElementsAttr denseResource) {
    std::lock_guard<std::recursive_mutex> lock(_cacheMutex);
    auto it = _cache.find(denseResource);
    if (it == _cache.end()) {
        cacheRawDataAndSplatness(denseResource);
        it = _cache.find(denseResource);
        return it != _cache.end() ? it->second : std::nullopt;
    }
    return it->second;
}

//
// wrapBaseContent
//

Const::Content wrapBaseContent(mlir::ElementsAttr baseContent) {
    auto content = getRawDataAndSplatness(baseContent);
    assert(content.has_value() && "ContentAttr verification should guarantee cache access");

    auto [data, isSplat] = content.value();

    return Const::Content::fromRawBuffer(mlir::cast<vpux::NDTypeInterface>(baseContent.getShapedType()), data,
                                         baseContent.getShapedType().getElementType(), isSplat);
}

}  // namespace

mlir::DenseResourceElementsAttr Const::createExternalConstContent(mlir::ShapedType type, ArrayRef<char> rawData,
                                                                  StringRef resourceName,
                                                                  ExternalConstContentCreationOptions options) {
    auto createNewBlob = [&]() -> mlir::AsmResourceBlob {
        constexpr size_t defaultAlignment =
                alignof(std::max_align_t);  // seemingly used nowhere except deleter - use C++ default
        constexpr bool isMutable = false;

        if (options.deepCopyConstData) {
            // copy and manage the memory internally (debug)
            auto ownedData = std::make_unique<char[]>(rawData.size());
            std::memcpy(ownedData.get(), rawData.data(), rawData.size());
            auto* rawPtr = ownedData.get();
            return mlir::AsmResourceBlob(
                    llvm::ArrayRef<char>(rawPtr, rawData.size()), defaultAlignment,
                    [_ownedData = std::move(ownedData)](void*, size_t, size_t) {
                        // _ownedData will be destroyed when the deleter is destroyed
                    },
                    isMutable);
        } else {
            constexpr auto noopDeleter = [](void*, size_t, size_t) {};
            return mlir::AsmResourceBlob(rawData, defaultAlignment, noopDeleter, isMutable);
        }
    };

    auto blobHandle = [&]() -> mlir::DenseResourceElementsHandle {
        auto& builtinDialectManager = mlir::DenseResourceElementsHandle::getManagerInterface(type.getContext());
        // assumption (as per MLIR documented behavior): inserting a new blob
        // with the same key would internally cause the key to change, so that
        // there are no collisions - thus, the blob is never overwritten here
        if (!options.allowDuplicatesForTheSameResourceName) {
            return builtinDialectManager.insert(resourceName, createNewBlob());
        }

        if (auto existingBlob = builtinDialectManager.getBlobManager().lookup(resourceName)) {
            assert(existingBlob->getBlob() != nullptr);
            assert(existingBlob->getBlob()->getData().size() == rawData.size() &&
                   "When existing blob is found, its data buffer must match new constant's data. This is guaranteed by "
                   "OpenVINO.");
            return mlir::DenseResourceElementsHandle(
                    existingBlob,
                    mlir::cast<mlir::DenseResourceElementsHandle::Dialect>(builtinDialectManager.getDialect()));
        }
        return builtinDialectManager.insert(resourceName, createNewBlob());
    }();

    auto res = mlir::DenseResourceElementsAttr::get(type, blobHandle);
    ::getCache<SplatnessCache>(type.getContext()).cacheRawDataAndSplatness(res);
    return res;
}

mlir::DenseElementsAttr Const::createConstContent(mlir::ShapedType type, ArrayRef<char> values) {
    return mlir::DenseElementsAttr::getFromRawBuffer(type, values);
}

mlir::DenseElementsAttr Const::detail::createConstContentWithConversion(mlir::ShapedType type, ArrayRef<float> array) {
    const auto elemType = type.getElementType();
    if (elemType.isF32()) {
        return mlir::DenseElementsAttr::get(type, array);
    } else if (elemType.isF16()) {
        const auto arrayFP16 = to_small_vector(array | transformed([](float val) {
                                                   return static_cast<vpux::type::float16>(val);
                                               }));
        return mlir::DenseElementsAttr::get(type, ArrayRef(arrayFP16));
    } else if (mlir::isa<mlir::Float8E5M2Type>(elemType)) {
        const auto arrayFloat8E5M2 = to_small_vector(array | transformed([](float val) {
                                                         return static_cast<vpux::type::float8_e5m2>(val);
                                                     }));
        return mlir::DenseElementsAttr::get(type, ArrayRef(arrayFloat8E5M2));
    } else if (mlir::isa<mlir::Float8E4M3FNType>(elemType)) {
        const auto arrayFloat8E4M3FN = to_small_vector(array | transformed([](float val) {
                                                           return static_cast<vpux::type::float8_e4m3>(val);
                                                       }));
        return mlir::DenseElementsAttr::get(type, ArrayRef(arrayFloat8E4M3FN));
    } else if (mlir::isa<mlir::Float8E8M0FNUType>(elemType)) {
        const auto arrayFloat8E8M0FNU = to_small_vector(array | transformed([](float val) {
                                                            return static_cast<vpux::type::float8_e8m0>(val);
                                                        }));
        return mlir::DenseElementsAttr::get(type, ArrayRef(arrayFloat8E8M0FNU));
    } else if (mlir::isa<mlir::Float4E2M1FNType>(elemType)) {
        const auto arrayFloat4E2M1FN = to_small_vector(array | transformed([](float val) {
                                                           return static_cast<vpux::type::float4_e2m1>(val);
                                                       }));
        return mlir::DenseElementsAttr::get(type, ArrayRef(arrayFloat4E2M1FN));
    }
    VPUX_THROW("Unsupported element type '{0}'", elemType);
    return {};
}

//
// ContentAttr::fold
//

Const::Content vpux::Const::ContentAttr::fold() const {
    auto baseContent = getBaseContent();
    auto res = wrapBaseContent(baseContent);
    for (const auto& attr : getTransformations()) {
        res = attr.transform(res);
    }
    return res;
}

//
// ContentAttr::print
//

void vpux::Const::ContentAttr::print(mlir::AsmPrinter& printer) const {
    if (auto symElementsAttr = mlir::dyn_cast_or_null<SymElementsAttr>(getBaseContent())) {
        printer << "ref";
        symElementsAttr.print(printer);
    } else {
        printer.printAttribute(getBaseContent());
    }

    // For dense resources print splat attribute since dense resource data is not yet
    // present at the time of const.DeclareOp parsing and splatness can't be inferred from
    // raw data. This is due to mlir printing and parsing order in which IR is parsed before
    // dialect_resources section which holds content data.
    if (mlir::isa<mlir::DenseResourceElementsAttr>(getBaseContent()) && isSplat()) {
        printer << " isSplat";
    }

    if (const auto transformations = getTransformations(); !transformations.empty()) {
        printer << ", " << '[' << transformations << ']';
    }
}

//
// ContentAttr::parse
//

mlir::Attribute vpux::Const::ContentAttr::parse(::mlir::AsmParser& parser, ::mlir::Type) {
    // What we are trying to parse:
    // ( ref<@symbol> : type | dense<...> : type | dense_resource<...> : type ) [, list_of_transformations]

    mlir::ElementsAttr baseContent;

    // parse SymElementsAttr or ElementsAttr
    if (mlir::succeeded(parser.parseOptionalKeyword("ref"))) {
        auto parseResult = mlir::FieldParser<Const::SymElementsAttr>::parse(parser);

        if (mlir::failed(parseResult)) {
            return nullptr;
        }

        baseContent = parseResult.value();
    } else if (mlir::failed(parser.parseAttribute(baseContent))) {
        return nullptr;
    }

    bool explicitSplat = false;
    if (mlir::succeeded(parser.parseOptionalKeyword("isSplat"))) {
        if (!mlir::isa<mlir::DenseResourceElementsAttr>(baseContent)) {
            // Note: scope 'isSplat' to dense_resource<> only - it makes *zero*
            // sense to specify it for other types of base content.
            std::ignore =
                    parser.emitError(parser.getNameLoc(), "isSplat keyword can only be specified for dense_resource<>");
            return nullptr;
        }
        explicitSplat = true;
    }

    // parse list of transformations
    mlir::SmallVector<vpux::Const::TransformAttrInterface> transformations{};
    if (mlir::succeeded(parser.parseOptionalComma())) {
        mlir::ArrayAttr arrayAttr;
        if (mlir::failed(parser.parseAttribute(arrayAttr))) {
            return nullptr;
        }

        transformations.reserve(arrayAttr.size());
        for (const auto attr : arrayAttr.getValue()) {
            const auto trAttr = mlir::dyn_cast<vpux::Const::TransformAttrInterface>(attr);
            VPUX_THROW_WHEN(trAttr == nullptr, "Got non transformation attribute : '{0}'", attr);
            transformations.push_back(trAttr);
        }
    }

    if (explicitSplat) {
        return parser.getChecked<ContentAttr>(baseContent, mlir::UnitAttr::get(baseContent.getContext()),
                                              ArrayRef(transformations));
    }
    return parser.getChecked<ContentAttr>(baseContent, ArrayRef(transformations));
}

mlir::ParseResult vpux::Const::parseContentAttr(mlir::AsmParser& parser, ContentAttr& content) {
    auto result = ContentAttr::parse(parser, nullptr);
    if (result == nullptr) {
        return mlir::failure();
    }
    content = mlir::cast<ContentAttr>(result);
    return mlir::success();
}

mlir::ElementsAttr vpux::Const::ContentAttr::getBaseContent() const {
    return getImpl()->baseContent;
}

mlir::ArrayRef<vpux::Const::TransformAttrInterface> vpux::Const::ContentAttr::getTransformations() const {
    return getTransformationsAttr().getValue();
}

vpux::NDTypeInterface vpux::Const::ContentAttr::getType() const {
    return getImpl()->finalType;
}

bool vpux::Const::ContentAttr::isSplat() const {
    return getImpl()->isSplat != nullptr;
}

vpux::Const::TransformAttrInterfaceArrayAttr vpux::Const::ContentAttr::getTransformationsAttr() const {
    return getImpl()->transformations;
}

llvm::hash_code vpux::Const::ContentAttr::getTransformationHash() const {
    return ContentAttr::getTransformationHash(getBaseContent(), getTransformations());
}

llvm::hash_code vpux::Const::ContentAttr::getTransformationHash(mlir::ElementsAttr baseContent,
                                                                ArrayRef<TransformAttrInterface> transformations) {
    const auto hashes = transformations | transformed([](TransformAttrInterface attr) {
                            return attr.getStableHashValue();
                        });
    // Note: base content is a rather simple container that supports limited
    // amount of information (unlike a generic tensor type); thus, it should
    // generally be enough to just hash shape + element type to ensure stable
    // hash with no conflicts (having e.g. an order attribute set on a tensor
    // type of a base content is not expected).
    const auto baseType = mlir::cast<vpux::NDTypeInterface>(baseContent.getType());
    return llvm::hash_combine(baseType.getShape().raw(), getStableHash(baseType.getElementType()),
                              llvm::hash_combine_range(hashes.begin(), hashes.end()));
}

void vpux::Const::printContentAttr(mlir::AsmPrinter& printer, const ContentAttr& content) {
    content.print(printer);
}

vpux::NDTypeInterface vpux::Const::inferFinalType(vpux::NDTypeInterface contentType,
                                                  mlir::ArrayRef<TransformAttrInterface> transformations) {
    auto inferredType = contentType;
    for (const auto& attr : transformations) {
        inferredType = attr.inferOutputType(inferredType);
    }
    return inferredType;
}

// Returns the output type and splatness of the content with transformations "as
// if" applied to this content.
std::pair<vpux::NDTypeInterface, bool> vpux::Const::inferFinalTypeAndSplat(
        mlir::ElementsAttr content, mlir::ArrayRef<vpux::Const::TransformAttrInterface> transformations) {
    // empty optional might be returned in case of the call during LIT test parsing
    // due to multi-stage parsing process of DenseResourceElementsAttr and
    // buffer being unavailable at the initial stage
    // assume non-splat in this case, splatness will be computed later during verification
    // when buffer becomes available
    auto inferredSplat = getRawDataAndSplatness(content).value_or(std::pair{mlir::ArrayRef<char>{}, false}).second;
    auto inferredType = mlir::cast<vpux::NDTypeInterface>(content.getType());
    for (const auto& attr : transformations) {
        inferredSplat = attr.inferOutputSplat(inferredSplat, inferredType);
        inferredType = attr.inferOutputType(inferredType);
    }
    return {inferredType, inferredSplat};
}

void vpux::Const::detail::ContentSetupTracingBase::updateConstantCallStack(
        mlir::MLIRContext* ctx, mlir::ArrayRef<TransformAttrInterface> transformations) const {
    if (id == nullptr) {
        return;
    }
    auto& csCache = vpux::getCache<vpux::Const::CallStackCache, vpux::Const::ConstDialect>(ctx);
    auto trace = vpux::Const::gatherTrace();
    if (trace.empty()) {
        return;
    }

    {
        auto callStack = csCache.lockCallStack();
        auto& currentTransformations = (*callStack)[id];
        auto tempTransformations = currentTransformations;

        // Add new elements or update existing ones if changed
        for (auto transformation : transformations) {
            auto matchIt = std::find_if(tempTransformations.begin(), tempTransformations.end(),
                                        [&](const vpux::Const::CallStackCache::TransformationTy& tEntry) {
                                            return std::get<0>(tEntry) == transformation;
                                        });
            if (matchIt == tempTransformations.end()) {
                currentTransformations.push_back(std::make_tuple(transformation, trace));
            } else {
                tempTransformations.erase(matchIt);
            }
        }

        // Remove outdated elements
        for (auto extra : tempTransformations) {
            auto remIt = std::find_if(currentTransformations.begin(), currentTransformations.end(),
                                      [&](const vpux::Const::CallStackCache::TransformationTy& tEntry) {
                                          return std::get<0>(tEntry) == std::get<0>(extra);
                                      });
            if (remIt != currentTransformations.end()) {
                currentTransformations.erase(remIt);
            }
        }
    }
}

void vpux::Const::detail::ContentSetupBase::addTransformation(TransformAttrInterface newTransformation) {
    checkInvalidated();

    auto comp = [](const vpux::Const::TransformAttrInterface& a, const vpux::Const::TransformAttrInterface& b) {
        return a.getPositionRequirement() < b.getPositionRequirement();
    };

    // Get an iterator to the FIRST element that is ordered AFTER newTransformation.
    // Examples:
    //   1) When inserting NONE, the first PREFERRED_LAST is returned.
    //   2) When inserting PREFERRED_LAST, the first LAST is returned.
    //   3) When inserting LAST, the last LAST is returned.
    // This ensures the following order of elements in _transformations and preserves insertion order:
    // [NONE, ..., NONE, PREFERRED_LAST, ..., PREFERRED_LAST, LAST, ..., LAST]
    auto insertionPosition = llvm::upper_bound(_transformations, newTransformation, comp);
    insertionPosition = _transformations.insert(insertionPosition, newTransformation);

    const auto& options = ::getCache<LazyFoldingCache>(getContext()).getOptions();
    const auto optimizations = options.getFoldingSequenceOptimizations(_baseType);

    bool optimized = false;
    auto currentPos = insertionPosition;
    do {
        for (auto& optimize : optimizations) {
            std::tie(insertionPosition, optimized) = optimize(_transformations, currentPos);
            if (optimized) {
                currentPos = insertionPosition;
                break;
            }
        }
    } while (optimized);

    // check single LAST requirement
    bool lastRequirementViolated =
            _transformations.size() >= 2 &&
            (_transformations.end() - 2)->getPositionRequirement() == details::PositionRequirement::LAST;
    VPUX_THROW_WHEN(lastRequirementViolated, "At most 1 attribute with LAST requirement allowed!");

    Base::updateConstantCallStack(getContext(), _transformations);
}

void vpux::Const::setLazyFoldingOptions(mlir::MLIRContext* ctx, const LazyFoldingOptions& options) {
    ::getCache<LazyFoldingCache>(ctx).setOptions(options);
}

// Transformation attributes stable hashes
namespace vpux::Const {
namespace details {
template <>
llvm::hash_code StableHashStorage<CastElemTypeAttrStorage>::calculateStableHash() const {
    return llvm::hash_combine(CastElemTypeAttr::getMnemonic(), getStableHash(this->elemType));
}

template <>
llvm::hash_code StableHashStorage<ConvertElemTypeAttrStorage>::calculateStableHash() const {
    return llvm::hash_combine(ConvertElemTypeAttr::getMnemonic(), getStableHash(this->elemType));
}

template <>
llvm::hash_code StableHashStorage<QuantizeAttrStorage>::calculateStableHash() const {
    return llvm::hash_combine(QuantizeAttr::getMnemonic(), getStableHash(this->targetType));
}

template <>
llvm::hash_code StableHashStorage<RescaleAttrStorage>::calculateStableHash() const {
    if (!this->scale.isSplat()) {
        // Note: non-splat rescale is currently not supported
        return {};
    }

    const auto scale = this->scale.fold().getSplatValue<double>();
    return llvm::hash_combine(RescaleAttr::getMnemonic(), llvm::APFloat(scale));
}

template <>
llvm::hash_code StableHashStorage<AddAttrStorage>::calculateStableHash() const {
    if (this->bias) {
        const auto bias = this->bias.getValue();
        return llvm::hash_combine(AddAttr::getMnemonic(), bias);
    } else {
        // biasArray case
        SmallVector<llvm::APFloat> biasValues;
        for (auto attr : this->biasArray) {
            biasValues.push_back(mlir::cast<mlir::FloatAttr>(attr).getValue());
        }
        return llvm::hash_combine(AddAttr::getMnemonic(),
                                  llvm::hash_combine_range(biasValues.begin(), biasValues.end()));
    }
}

template <>
llvm::hash_code StableHashStorage<ReshapeAttrStorage>::calculateStableHash() const {
    const auto newShape = parseIntArrayAttr<int64_t>(this->shape);
    return llvm::hash_combine(ReshapeAttr::getMnemonic(), llvm::hash_combine_range(newShape.begin(), newShape.end()));
}

template <>
llvm::hash_code StableHashStorage<ReorderAttrStorage>::calculateStableHash() const {
    const auto order = DimsOrder::fromAffineMap(this->order.getValue());
    return llvm::hash_combine(ReorderAttr::getMnemonic(), order.code());
}

template <>
llvm::hash_code StableHashStorage<ReverseAttrStorage>::calculateStableHash() const {
    return llvm::hash_combine(ReverseAttr::getMnemonic(), this->axis.getValue());
}

template <>
llvm::hash_code StableHashStorage<PadWithZeroAttrStorage>::calculateStableHash() const {
    const auto padBefore = parseIntArrayAttr<int64_t>(this->padBefore);
    const auto padAfter = parseIntArrayAttr<int64_t>(this->padAfter);
    return llvm::hash_combine(PadWithZeroAttr::getMnemonic(),
                              llvm::hash_combine_range(padBefore.begin(), padBefore.end()),
                              llvm::hash_combine_range(padAfter.begin(), padAfter.end()));
}

template <>
llvm::hash_code StableHashStorage<SubViewAttrStorage>::calculateStableHash() const {
    const auto shape = parseIntArrayAttr<int64_t>(this->shape);
    const auto offset = parseIntArrayAttr<int64_t>(this->offset);
    return llvm::hash_combine(SubViewAttr::getMnemonic(), llvm::hash_combine_range(shape.begin(), shape.end()),
                              llvm::hash_combine_range(offset.begin(), offset.end()));
}

template <>
llvm::hash_code StableHashStorage<BroadcastAttrStorage>::calculateStableHash() const {
    const auto axis = this->axis.getValue();
    const auto value = this->value.getValue();
    return llvm::hash_combine(BroadcastAttr::getMnemonic(), axis, value);
}

template <>
llvm::hash_code StableHashStorage<TransposeAttrStorage>::calculateStableHash() const {
    const auto order = DimsOrder::fromAffineMap(this->order.getValue());
    return llvm::hash_combine(TransposeAttr::getMnemonic(), order.code());
}

template <>
llvm::hash_code StableHashStorage<MemPermuteAttrStorage>::calculateStableHash() const {
    const auto order = DimsOrder::fromAffineMap(this->dstOrder.getValue());
    const auto perm = DimsOrder::fromAffineMap(this->memPerm.getValue());
    return llvm::hash_combine(MemPermuteAttr::getMnemonic(), order.code(), perm.code());
}

template <>
llvm::hash_code StableHashStorage<LayoutCastAttrStorage>::calculateStableHash() const {
    const auto order = DimsOrder::fromAffineMap(this->dstOrder.getValue());
    return llvm::hash_combine(LayoutCastAttr::getMnemonic(), order.code());
}

template <>
llvm::hash_code StableHashStorage<ChangeShapeAndElemTypeAttrStorage>::calculateStableHash() const {
    const auto shape = parseIntArrayAttr<int64_t>(this->shape);
    return llvm::hash_combine(ChangeShapeAndElemTypeAttr::getMnemonic(),
                              llvm::hash_combine_range(shape.begin(), shape.end()), getStableHash(this->elemType));
}

template <>
llvm::hash_code StableHashStorage<AffineReshapeAttrStorage>::calculateStableHash() const {
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(this->dimMapping);
    const auto dimMappingRefs = llvm::map_range(dimMapping, [](const auto& array) {
        return ArrayRef<int64_t>(array);
    });
    const auto shapeValue = parseIntArrayAttr<int64_t>(this->shapeValue);

    return llvm::hash_combine(AffineReshapeAttr::getMnemonic(),
                              llvm::hash_combine_range(dimMappingRefs.begin(), dimMappingRefs.end()),
                              llvm::hash_value(ArrayRef<int64_t>(shapeValue)));
}

template <>
llvm::hash_code StableHashStorage<GatherElementsAttrStorage>::calculateStableHash() const {
    const auto indicesDense = this->indices;
    const auto indicesRaw = indicesDense.getRawData();
    return llvm::hash_combine(GatherElementsAttr::getMnemonic(), this->axis.getValue(),
                              getStableHash(this->indices.getType()),
                              llvm::hash_combine_range(indicesRaw.begin(), indicesRaw.end()));
}

template <>
llvm::hash_code StableHashStorage<CumSumAttrStorage>::calculateStableHash() const {
    return llvm::hash_combine(CumSumAttr::getMnemonic(), this->axis.getValue(), this->exclusive.getValue(),
                              this->reverse.getValue());
}
}  // namespace details

llvm::hash_code RescaleAttr::getStableHashValue() const {
    VPUX_THROW_UNLESS(getScale().isSplat(), "RescaleAttr scale must be splat");
    return static_cast<details::StableHashStorage<details::RescaleAttrStorage>*>(getImpl())->stableHash;
}

}  // namespace vpux::Const
