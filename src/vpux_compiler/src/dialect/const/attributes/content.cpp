//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/constant_transformations_control.hpp"
#include "vpux/compiler/dialect/const/utils/const_logger.hpp"

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/DialectResourceBlobManager.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>

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
    using ValueType = std::pair<mlir::ArrayRef<char>, bool>;
    mlir::DenseMap<StringRef, ValueType> _cache;

public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplatnessCache)

    SplatnessCache(mlir::Dialect* dialect): Base(dialect) {
    }

    void cacheRawDataAndSplatness(mlir::DenseResourceElementsAttr denseResource);
    ValueType getRawDataAndSplatness(mlir::DenseResourceElementsAttr denseResource);

private:
    std::recursive_mutex _cacheMutex{};  // Note: recursive because "get" can call "cache"
};

template <typename Cache>
Cache& getCache(mlir::MLIRContext* ctx) {
    auto* dialect = ctx->getOrLoadDialect<vpux::Const::ConstDialect>();
    assert(dialect != nullptr && "ConstDialect must be present in the context");

    auto* iface = dialect->getRegisteredInterface<Cache>();
    assert(iface != nullptr && "The requested cache must be registered in the context");
    return *iface;
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
}

//
// ContentAttr::verify
//

mlir::LogicalResult vpux::Const::ContentAttr::verify(FuncRef<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::ElementsAttr baseContent,
                                                     vpux::Const::TransformAttrInterfaceArrayAttr transformations,
                                                     vpux::NDTypeInterface, mlir::UnitAttr isSplat) {
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
        if (blob != nullptr && mlir::failed(verifyDenseResource(emitError, denseResource, isSplat != nullptr))) {
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
                                                                  mlir::DenseResourceElementsAttr denseResource,
                                                                  bool isSplat) {
    if (denseResource == nullptr) {
        return printTo(emitError(), "Got NULL 'denseResource' in 'ContentAttr'");
    }
    auto blob = denseResource.getRawHandle().getBlob();
    if (blob == nullptr) {
        return printTo(emitError(), "Can't access constant content for verification, resource handle : {0}",
                       denseResource.getRawHandle().getKey());
    }
    // Note: manual checks required since dense resource blob is opaque and does not perform much validation itself
    const auto bytes = blob->getData();
    auto bitWidth = vpux::getElemTypeSize(denseResource.getShapedType().getElementType()).count();
    if (vpux::Const::isSubByte(bitWidth)) {
        const auto bufferSize = checked_cast<size_t>(bytes.size());
        const auto numBytes = static_cast<size_t>(getExpectedBufferSize(denseResource.getShapedType()).count());
        // Note: limit sub-byte data splats to 1 byte
        const bool valid = (isSplat && bufferSize == 1) || (bufferSize == numBytes);
        if (!valid) {
            return printTo(emitError(),
                           "Size of dense resource buffer '{0}' in 'baseContent' doesn't match its type '{1}'",
                           bytes.size(), denseResource.getShapedType());
        }
    } else {
        bool ignored = false;
        if (!mlir::DenseElementsAttr::isValidRawBuffer(denseResource.getShapedType(), bytes, ignored)) {
            return printTo(emitError(),
                           "Size of dense resource buffer '{0}' in 'baseContent' doesn't match its type '{1}'",
                           bytes.size(), denseResource.getShapedType());
        }
    }

    return mlir::success();
}

namespace {

std::pair<mlir::ArrayRef<char>, bool> detectSplatElementWise(mlir::ArrayRef<char> data, size_t bitWidth) {
    const auto elemIsSplat = [&](size_t offset) {
        const char* firstElemAddr = data.data();
        for (size_t i = offset; i < data.size(); i += offset) {
            if (std::memcmp(firstElemAddr + i, firstElemAddr, offset) != 0) {
                return false;
            }
        }

        return true;
    };

    if (vpux::Const::isSubByte(bitWidth)) {
        const char firstByte = *data.data();

        const auto elemPerByte = CHAR_BIT / bitWidth;
        VPUX_THROW_UNLESS(vpux::isPowerOfTwo(elemPerByte), "Invalid number of elements per byte '{0}'", elemPerByte);
        const size_t mask = checked_cast<uint8_t>(checked_cast<uint16_t>(std::pow(2, bitWidth)) - 1);
        size_t shift = 0;
        // Compare first byte.
        for (size_t i = 0; i < elemPerByte - 1; i += 1) {
            uint8_t preVal = (firstByte >> shift) & mask;
            shift += bitWidth;
            uint8_t nextVal = (firstByte >> shift) & mask;
            if (preVal != nextVal) {
                return {data, false};
            }
        }

        if (!elemIsSplat(1)) {
            return {data, false};
        }

        return {data.take_front(1), true};
    }

    auto elementSizeBytes = bitWidth / CHAR_BIT;
    VPUX_THROW_WHEN((data.size() < elementSizeBytes), "The data must contain at least one element");
    VPUX_THROW_WHEN(((data.size() % elementSizeBytes) != 0), "The data array has unexpected length");

    if (data.size() == elementSizeBytes) {
        return {data, true};
    }

    if (!elemIsSplat(elementSizeBytes)) {
        return {data, false};
    }

    return {data.take_front(elementSizeBytes), true};
}

// Returns whether the data is a splat, correcting the data array when it is.
std::pair<mlir::ArrayRef<char>, bool> detectSplatManually(mlir::ShapedType type, mlir::ArrayRef<char> data) {
    if (data.empty()) {
        return {data, false};  // empty data is not a splat
    }

    const auto bitWidth = vpux::getElemTypeSize(type).count();

    // Use isValidRawBuffer() for the side effects to detect whether a buffer is a splat.
    // Because of the limitation of MLIR, we shouldn't use isValidRawBuffer() for sub byte type except i1.
    // For example, 0x12 will return true but the byte actually contains two different I4 elements.
    bool isSplat = false;
    if (!vpux::Const::isSubByte(bitWidth)) {
        std::ignore = mlir::DenseElementsAttr::isValidRawBuffer(type, data, isSplat);
        if (isSplat) {
            return {data, true};
        }
    }

    // isValidRawBuffer() only checks single-element splats but if the data
    // array has identical elements, a manual check is required
    return detectSplatElementWise(data, static_cast<size_t>(bitWidth));
}

/// Returns pointer to baseContent's data and whether the data is splat.
std::pair<mlir::ArrayRef<char>, bool> getRawDataAndSplatness(mlir::ElementsAttr baseContent) {
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(baseContent)) {
        return {dense.getRawData(), dense.isSplat()};
    }

    // We cannot know if we have a splat value because we cannot dereference the symbol from here.
    if (mlir::isa<Const::SymElementsAttr>(baseContent)) {
        return {mlir::ArrayRef<char>(), false};
    }

    auto denseResource = mlir::cast<mlir::DenseResourceElementsAttr>(baseContent);
    return getCache<SplatnessCache>(baseContent.getContext()).getRawDataAndSplatness(denseResource);
}

void SplatnessCache::cacheRawDataAndSplatness(mlir::DenseResourceElementsAttr denseResource) {
    auto key = denseResource.getRawHandle().getKey();
    // dense resource doesn't support splat detection in MLIR itself
    auto blob = denseResource.getRawHandle().getBlob();
    if (blob != nullptr) {
        std::lock_guard<std::recursive_mutex> lock(_cacheMutex);
        _cache[key] = detectSplatManually(denseResource.getShapedType(), blob->getData());
    }
}

typename SplatnessCache::ValueType SplatnessCache::getRawDataAndSplatness(
        mlir::DenseResourceElementsAttr denseResource) {
    auto key = denseResource.getRawHandle().getKey();
    std::lock_guard<std::recursive_mutex> lock(_cacheMutex);
    auto it = _cache.find(key);
    if (it == _cache.end()) {
        cacheRawDataAndSplatness(denseResource);
        it = _cache.find(key);
        return it != _cache.end() ? it->second : std::make_pair(ArrayRef<char>{}, false);
    }
    return it->second;
}

//
// wrapBaseContent
//

Const::Content wrapBaseContent(mlir::ElementsAttr baseContent) {
    ArrayRef<char> data = {};
    bool isSplat = false;

    std::tie(data, isSplat) = getRawDataAndSplatness(baseContent);

    return Const::Content::fromRawBuffer(mlir::cast<vpux::NDTypeInterface>(baseContent.getShapedType()), data,
                                         baseContent.getShapedType().getElementType(), isSplat);
}

}  // namespace

mlir::DenseResourceElementsAttr Const::createExternalConstContent(mlir::ShapedType type, ArrayRef<char> rawData,
                                                                  StringRef resourcePrefix, bool deepCopyConstData) {
    constexpr size_t defaultAlignment =
            alignof(std::max_align_t);  // seemingly used nowhere except deleter - use C++ default
    constexpr bool isMutable = false;

    auto blob = [&]() -> mlir::AsmResourceBlob {
        if (deepCopyConstData) {
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
    }();

    auto& builtinDialectManager = mlir::DenseResourceElementsHandle::getManagerInterface(type.getContext());
    // assumption (as per MLIR documented behavior): inserting a new blob with the same key would internally cause
    // the key to change, so that there are no collisions - thus, the blob is never overwritten here
    auto res =
            mlir::DenseResourceElementsAttr::get(type, builtinDialectManager.insert(resourcePrefix, std::move(blob)));
    getCache<SplatnessCache>(type.getContext()).cacheRawDataAndSplatness(res);
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
    }
    VPUX_THROW("Unsupported element type '{0}'", elemType);
    return nullptr;
}

//
// ContentAttr::fold
//

Const::Content vpux::Const::ContentAttr::fold(bool bypassCache) const {
    auto baseContent = getBaseContent();

#ifdef BACKGROUND_FOLDING_ENABLED
    if (!bypassCache) {
        auto& cacheManager = Const::ConstantFoldingCacheManager::getInstance();
        auto ctx = baseContent.getContext();
        if (cacheManager.contains(ctx)) {
            auto& cache = cacheManager.get(ctx);
            auto content = cache.getContent(*this);
            if (content.has_value()) {
                return std::move(content.value());
            }
        }
    }
#else
    VPUX_UNUSED(bypassCache);
#endif

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
    return ContentAttr::getTransformationHash(getTransformations());
}

llvm::hash_code vpux::Const::ContentAttr::getTransformationHash(ArrayRef<TransformAttrInterface> transformations) {
    const auto hashes = transformations | transformed([](TransformAttrInterface attr) {
                            return attr.getStableHashValue();
                        });
    return llvm::hash_combine_range(hashes.begin(), hashes.end());
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
    bool inferredSplat = getRawDataAndSplatness(content).second;
    auto inferredType = mlir::cast<vpux::NDTypeInterface>(content.getType());
    for (const auto& attr : transformations) {
        inferredSplat = attr.inferOutputSplat(inferredSplat, inferredType);
        inferredType = attr.inferOutputType(inferredType);
    }
    return {inferredType, inferredSplat};
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

    const auto& options = getCache<LazyFoldingCache>(getContext()).getOptions();
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
}

void vpux::Const::setLazyFoldingOptions(mlir::MLIRContext* ctx, const LazyFoldingOptions& options) {
    getCache<LazyFoldingCache>(ctx).setOptions(options);
}
