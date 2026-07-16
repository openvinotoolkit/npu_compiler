//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/utils/constant_tracing.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>

namespace vpux::Const {

SmallVector<int64_t> calculateMultiIndex(ArrayRef<int64_t> shape, int64_t linearIndex);
int64_t calculateLinearIndex(ArrayRef<int64_t> shape, ArrayRef<int64_t> indices);

namespace detail {

struct ContentSetupTracingBaseEmpty {
    template <typename... Args>
    ContentSetupTracingBaseEmpty(Args...) {
    }
    void updateConstantCallStack(mlir::MLIRContext*, ArrayRef<TransformAttrInterface>) const {
    }
};

struct ContentSetupTracingBase {
public:
    vpux::Const::TraceId id;

    ContentSetupTracingBase() = default;
    ContentSetupTracingBase(vpux::Const::TraceId traceId): id(traceId) {
    }

    void updateConstantCallStack(mlir::MLIRContext* ctx, ArrayRef<TransformAttrInterface> transformations) const;
};

using TracingBase = std::conditional_t<isDeveloperBuild(), ContentSetupTracingBase, ContentSetupTracingBaseEmpty>;

/// Base class for constant data transformations setup. Provides basic API. Not
/// intended for direct use.
class ContentSetupBase : public TracingBase {
    using Base = TracingBase;

    NDTypeInterface _baseType;
    SmallVector<TransformAttrInterface> _transformations;

public:
    ContentSetupBase() = default;
    ~ContentSetupBase() = default;
    ContentSetupBase(const ContentSetupBase&) = default;
    ContentSetupBase& operator=(const ContentSetupBase&) = default;
    ContentSetupBase(ContentSetupBase&& other);
    ContentSetupBase& operator=(ContentSetupBase&& other);

    // This constructor throws an exception when base type is undefined.
    ContentSetupBase(vpux::Const::TraceId id, mlir::Type baseType, ArrayRef<TransformAttrInterface> transformations);

    // getters
    mlir::MLIRContext* getContext() const;
    ArrayRef<TransformAttrInterface> getTransformations() const&;
    ArrayRef<TransformAttrInterface> getTransformations() && = delete;

    // transformations
    void addTransformation(TransformAttrInterface newTransformation);

protected:
    bool isInvalidated() const;

    // Ensures (by the means of exception being thrown) that this object is not
    // invalidated and could still be used by the user.
    void checkInvalidated() const;
};
}  // namespace detail

// Returns the output type "as if" the transformations were applied to a tensor of type contentType.
vpux::NDTypeInterface inferFinalType(vpux::NDTypeInterface contentType,
                                     mlir::ArrayRef<TransformAttrInterface> transformations);
// Returns the output type and splatness of the content with transformations "as
// if" applied to this content.
// Use inferFinalType() instead if you are only interested in the type.
std::pair<vpux::NDTypeInterface, bool> inferFinalTypeAndSplat(mlir::ElementsAttr content,
                                                              mlir::ArrayRef<TransformAttrInterface> transformations);

class ContentAttr;

namespace detail {
// used as a fallback in ContentSetup
struct NoopGet {
    using return_type = void;
};
};  // namespace detail

template <typename Get = detail::NoopGet>
class SpecializedContentSetup final : public detail::ContentSetupBase {
    Get _get;
    // Note: we want to query return_type from Get callable in order to force
    // users of SpecializedContentSetup to provide *custom* types -- this is
    // necessary since C++ lambdas are not move-assignable and thus do not work
    // with SpecializedContentSetup's usages. for instance:
    // ```cpp
    // auto setup = ...;
    // // requires assignment:
    // if (something) { setup = setup.add(42.0); }
    // else { setup = setup.rescale(5.0); }
    // ```
    using GetReturnType = typename Get::return_type;

    // Require the user to explicitly use clone() when there's a need to copy.
    SpecializedContentSetup(const SpecializedContentSetup&) = default;
    SpecializedContentSetup& operator=(const SpecializedContentSetup&) = default;

public:
    SpecializedContentSetup(vpux::Const::TraceId id, mlir::Type baseType,
                            ArrayRef<TransformAttrInterface> transformations = {}, Get&& get = detail::NoopGet{})
            : ContentSetupBase(id, baseType, transformations), _get(std::move(get)) {
    }

    SpecializedContentSetup(SpecializedContentSetup&&) = default;
    SpecializedContentSetup& operator=(SpecializedContentSetup&&) = default;
    ~SpecializedContentSetup() = default;

    SpecializedContentSetup clone() const;

    // shadows base class' version
    [[nodiscard]] SpecializedContentSetup addTransformation(TransformAttrInterface newTransformation);

    // implemented by <concrete transformation attribute>.cpp
    [[nodiscard]] SpecializedContentSetup broadcast(Dim axis, int64_t value);
    [[nodiscard]] SpecializedContentSetup castElemType(mlir::Type newElemType);
    [[nodiscard]] SpecializedContentSetup convertElemType(mlir::Type newElemType);
    [[nodiscard]] SpecializedContentSetup dequantize();
    [[nodiscard]] SpecializedContentSetup rescale(double scale);
    [[nodiscard]] SpecializedContentSetup rescale(vpux::Const::ContentAttr attr);
    [[nodiscard]] SpecializedContentSetup relocateWeightsTablePointers(
            ArrayRef<uint32_t> weightsPtr, uint64_t sparsityPtr, vpux::ShapeRef offsets, uint64_t weightsTableSize,
            uint64_t weightsElemBitSize, VPUIP::SparsityCompressionAttr weightsCompression, uint64_t channelOffset,
            uint64_t originalOC);
    [[nodiscard]] SpecializedContentSetup swizzleConstant(uint64_t swizzleKey, uint64_t arch);
    [[nodiscard]] SpecializedContentSetup add(double bias);
    [[nodiscard]] SpecializedContentSetup reshape(vpux::ShapeRef newShape);
    [[nodiscard]] SpecializedContentSetup reverse(Dim axis);
    [[nodiscard]] SpecializedContentSetup reorder(const vpux::DimsOrder& newOrder);
    [[nodiscard]] SpecializedContentSetup padWithZero(vpux::ShapeRef padBefore, vpux::ShapeRef padAfter);
    [[nodiscard]] SpecializedContentSetup subview(vpux::ShapeRef offset, vpux::ShapeRef shape);
    [[nodiscard]] SpecializedContentSetup transpose(const vpux::DimsOrder& newOrder);
    [[nodiscard]] SpecializedContentSetup memPermute(const vpux::DimsOrder& dstOrder, const vpux::DimsOrder& memPerm);
    [[nodiscard]] SpecializedContentSetup layoutCast(const vpux::DimsOrder& dstOrder);
    [[nodiscard]] SpecializedContentSetup expandDilated(vpux::ShapeRef dilations);
    [[nodiscard]] SpecializedContentSetup getSparsityMap();
    [[nodiscard]] SpecializedContentSetup sparsify(bool compressOutputType,
                                                   mlir::ElementsAttr numActualElements = nullptr);
    [[nodiscard]] SpecializedContentSetup changeShapeAndElemType(vpux::ShapeRef newShape, mlir::Type newElemType);
    [[nodiscard]] SpecializedContentSetup scalarMultInverse();
    [[nodiscard]] SpecializedContentSetup fuse(mlir::RankedTensorType fusedTensorType, const ContentAttr& weightsTable,
                                               const ContentAttr& weights, const ContentAttr& sparsity,
                                               const ContentAttr& activations);
    [[nodiscard]] SpecializedContentSetup fuse(mlir::RankedTensorType fusedTensorType,
                                               const std::vector<ContentAttr>& constants);
    [[nodiscard]] SpecializedContentSetup quantize(mlir::quant::QuantizedType newElemType);
    [[nodiscard]] SpecializedContentSetup affineReshape(mlir::ArrayAttr dimMapping, mlir::ArrayAttr shapeValue);
    [[nodiscard]] SpecializedContentSetup interpolate(mlir::ArrayAttr axes, mlir::ArrayAttr sizes,
                                                      mlir::StringAttr mode, mlir::StringAttr coordMode,
                                                      mlir::StringAttr nearestMode, mlir::BoolAttr antialias,
                                                      mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd,
                                                      mlir::FloatAttr cubeCoeff);
    [[nodiscard]] SpecializedContentSetup gatherElements(mlir::IntegerAttr axisAttr,
                                                         mlir::DenseElementsAttr indicesAttr);
    [[nodiscard]] SpecializedContentSetup cumSum(mlir::IntegerAttr axis, mlir::BoolAttr exclusive,
                                                 mlir::BoolAttr reverse);
    [[nodiscard]] SpecializedContentSetup logicalNot();

    // Note: this method only exists when there's an explicit "Get" method
    // provided by the user.
    template <typename T = Get>
    [[nodiscard]] GetReturnType get() const {
        constexpr bool validGet = !std::is_same_v<Get, detail::NoopGet>;
        static_assert(validGet, "This version of content setup does not support .get()");
        checkInvalidated();
        return _get(*this);
    }
};
// ctad's explicit deduction guide for "Get" method
template <typename Callable>
SpecializedContentSetup(Const::TraceId, mlir::Type, ArrayRef<TransformAttrInterface>,
                        Callable&&) -> SpecializedContentSetup<Callable>;

/// Default version of the content setup object. Users are highly recommended to
/// use this instead of the "specialized" version: prefer explicit content
/// construction (from setup's transformations) to implicit `.get()`.
using ContentSetup = SpecializedContentSetup<detail::NoopGet>;

}  // namespace vpux::Const

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/const/attributes.hpp.inc>

namespace vpux::Const {

// Default custom<ContentAttr> parsing & printing
mlir::ParseResult parseContentAttr(mlir::AsmParser& parser, ContentAttr& content);
void printContentAttr(mlir::AsmPrinter& printer, const ContentAttr& content);

/** @brief External constant prefix.

    This prefix is used also in the context of weights separation. Be careful when changing it.
    "ow" stands for "original weights"
*/

constexpr const char* IMPORTED_WEIGHT_PREFIX = "vpux_ow_";

/** @brief A collection of options to control createExternalConstContent()
           behaviour.
*/
struct ExternalConstContentCreationOptions {
    //! Controls whether external data is copied into MLIR.
    /*! This flag overrides the contract of the API and allows compiler to own
        the external data (by making a data copy). It is only intended for debug
        purposes (in particular, for vpux-translate), as it can (and will)
        considerably increase the memory consumption.
    */
    bool deepCopyConstData = false;

    //! Controls whether resource name must be unique.
    /*! This flag overrides the default behaviour of the API, that is to create
        unique entries for external data. For example, passing the same resource
        name 'X' multiple times would result in 'X', 'X_1', ..., 'X_N'
        dense_resource<> entries being produced. This flag changes this so that
        if there's already a created blob for a given resource name, it is
        reused. This is useful in OV model compression scenarios where multiple
        OV constant nodes point to the exact same memory buffer.
    */
    bool allowDuplicatesForTheSameResourceName = false;
};

/** @brief Returns new dense_resource<> "base" content.

    This function is used to create the base content for the constant that is
    external to the compiler. As the memory is explicitly external, it is *not*
    owned by the created content (users must ensure the lifetime of the data is
    longer than the lifetime of the created content).
    An additional input flag deepCopyConstData was added to actually allow
    to copy and own the data internally. This is an exception to the normal usage
    and it's only intended for debug purposes (in particular for vpux-translate),
    as it can considerably increase memory consumption.

    @note This function is required instead of manual content creation since it
    performs additional optimizations not done by MLIR.
 */
mlir::DenseResourceElementsAttr createExternalConstContent(mlir::ShapedType type, ArrayRef<char> rawData,
                                                           StringRef resourceName,
                                                           ExternalConstContentCreationOptions options = {});

namespace detail {
mlir::DenseElementsAttr createConstContentWithConversion(mlir::ShapedType type, ArrayRef<float> values);
}

/** @brief Returns new dense<> "base" content.

    This function is used to create the base content for the constant that is
    internal to the compiler. In this case, the created content owns the data.

    Additionally, a float -> float16 conversion is performed for float values
    when the type specified requires float16 elements.

    @note Call this function for constant operations instead of MLIR
 */
template <typename T, std::enable_if_t<!std::is_same<T, char>::value, bool> = true>
mlir::DenseElementsAttr createConstContent(mlir::ShapedType type, ArrayRef<T> values) {
    if constexpr (std::is_same<T, float>::value) {
        return detail::createConstContentWithConversion(type, values);
    } else {
        return mlir::DenseElementsAttr::get(type, values);
    }
}

/** @brief Returns new dense<> "base" content.

    @note This is an overload that assumes the constant data provided is a raw
    buffer.
 */
mlir::DenseElementsAttr createConstContent(mlir::ShapedType type, ArrayRef<char> values);

// Returns a special zero-sized base content used by transformations that operate on an array of
// constants (e.g. ConcatAttr). Must not be used as real data — accessing its memory is undefined.
mlir::ElementsAttr getDummyBaseContent(mlir::MLIRContext* ctx);

// Returns true if the given base content is the dummy created by getDummyBaseContent().
bool isDummyBaseContent(mlir::ElementsAttr baseContent);

// Creates a ContentAttr with a ConcatAttr transformation and a dummy base content.
// The output type is inferred from axis and the constants' shapes.
ContentAttr createConcatContentAttr(mlir::MLIRContext* ctx, mlir::ArrayAttr staticOffsets, int64_t axis,
                                    mlir::ArrayRef<ContentAttr> inputContents);

}  // namespace vpux::Const
