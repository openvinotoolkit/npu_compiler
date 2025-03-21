//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>

namespace vpux::Const {

namespace detail {
/// Base class for constant data transformations setup. Provides basic API. Not
/// intended for direct use.
class ContentSetupBase {
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
    ContentSetupBase(mlir::Type baseType, ArrayRef<TransformAttrInterface> transformations);

    // getters
    mlir::MLIRContext* getContext() const;
    ArrayRef<TransformAttrInterface> getTransformations() const;

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
    SpecializedContentSetup(mlir::Type baseType, ArrayRef<TransformAttrInterface> transformations = {},
                            Get&& get = detail::NoopGet{})
            : ContentSetupBase(baseType, transformations), _get(std::move(get)) {
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
    [[nodiscard]] SpecializedContentSetup relocateWeightsTablePointers(
            ArrayRef<uint32_t> weightsPtr, uint64_t sparsityPtr, vpux::ShapeRef offsets, uint64_t weightsTableSize,
            uint64_t weightsElemBitSize, VPUIP::SparsityCompressionAttr weightsCompression, uint64_t channelOffset,
            uint64_t originalOC);
    [[nodiscard]] SpecializedContentSetup swizzleConstant(uint64_t swizzleKey, uint64_t arch);
    [[nodiscard]] SpecializedContentSetup add(double bias);
    [[nodiscard]] SpecializedContentSetup reshape(vpux::ShapeRef newShape);
    [[nodiscard]] SpecializedContentSetup reverse(Dim axis);
    [[nodiscard]] SpecializedContentSetup reorder(vpux::DimsOrder newOrder);
    [[nodiscard]] SpecializedContentSetup padWithZero(vpux::ShapeRef padBefore, vpux::ShapeRef padAfter);
    [[nodiscard]] SpecializedContentSetup subview(vpux::ShapeRef offset, vpux::ShapeRef shape);
    [[nodiscard]] SpecializedContentSetup bitPack(int64_t width);
    [[nodiscard]] SpecializedContentSetup transpose(vpux::DimsOrder newOrder);
    [[nodiscard]] SpecializedContentSetup memPermute(vpux::DimsOrder dstOrder, vpux::DimsOrder memPerm);
    [[nodiscard]] SpecializedContentSetup layoutCast(vpux::DimsOrder dstOrder);
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
SpecializedContentSetup(mlir::Type, ArrayRef<TransformAttrInterface>, Callable &&)->SpecializedContentSetup<Callable>;

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

/// @brief External constant prefix used for OpenVino constants.
constexpr const char* OPENVINO_CONST_PREFIX = "ov";

/** @brief Returns new dense_resource<> "base" content.

    This function is used to create the base content for the constant that is
    external to the compiler. As the memory is explicitly external, it is *not*
    owned by the created content (users must ensure the lifetime of the data is
    longer than the lifetime of the created content).

    @note This function is required instead of manual content creation since it
    performs additional optimizations not done by MLIR.
 */
mlir::DenseResourceElementsAttr createExternalConstContent(mlir::ShapedType type, ArrayRef<char> rawData,
                                                           StringRef resourcePrefix);

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

}  // namespace vpux::Const
