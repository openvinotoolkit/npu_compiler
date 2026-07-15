//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Support/LLVM.h>
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/stable_hash.hpp"
#ifdef BACKGROUND_FOLDING_ENABLED
#include "vpux/compiler/dialect/const/utils/constant_folding_cache.hpp"
#endif
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::Const {
namespace detail {
ContentSetupBase::ContentSetupBase(vpux::Const::TraceId id, mlir::Type baseType,
                                   ArrayRef<TransformAttrInterface> transformations)
        : Base(id), _baseType(mlir::cast_if_present<NDTypeInterface>(baseType)), _transformations(transformations) {
    VPUX_THROW_WHEN(_baseType == nullptr, "base type must not be null");
}

ContentSetupBase::ContentSetupBase(ContentSetupBase&& other)
        : Base(static_cast<Base&&>(other)),
          _baseType(std::exchange(other._baseType, nullptr)),
          _transformations(std::move(other._transformations)) {
}

ContentSetupBase& ContentSetupBase::operator=(ContentSetupBase&& other) {
    ContentSetupBase tmp(std::move(other));
    // avoids calling move assignment operator when using std::swap(*this, tmp)
    Base::operator=(static_cast<Base&&>(tmp));
    std::swap(_baseType, tmp._baseType);
    std::swap(_transformations, tmp._transformations);
    return *this;
}

mlir::MLIRContext* ContentSetupBase::getContext() const {
    checkInvalidated();
    return _baseType.getContext();
}

ArrayRef<TransformAttrInterface> ContentSetupBase::getTransformations() const& {
    checkInvalidated();
    return _transformations;
}

bool ContentSetupBase::isInvalidated() const {
    return _baseType == nullptr;
}

void ContentSetupBase::checkInvalidated() const {
    VPUX_THROW_WHEN(isInvalidated(),
                    "The object was marked invalidated because it was moved. Did you forget to call clone()?");
}
}  // namespace detail

template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::clone() const {
    checkInvalidated();
    return *this;
}

template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::addTransformation(TransformAttrInterface newTransformation) {
    ContentSetupBase::addTransformation(newTransformation);
    return std::move(*this);
}

template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::broadcast(Dim axis, int64_t value) {
    return addTransformation(
            Const::BroadcastAttr::get(getIntAttr(getContext(), axis.ind()), getIntAttr(getContext(), value)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::castElemType(mlir::Type newElemType) {
    return addTransformation(Const::CastElemTypeAttr::get(newElemType));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::convertElemType(mlir::Type newElemType) {
    return addTransformation(Const::ConvertElemTypeAttr::get(newElemType));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::dequantize() {
    return addTransformation(Const::DequantizeAttr::get(getContext()));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::rescale(double scale) {
    return addTransformation(Const::RescaleAttr::get(getFPAttr(getContext(), scale)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::rescale(vpux::Const::ContentAttr attr) {
    return addTransformation(Const::RescaleAttr::get(attr));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::relocateWeightsTablePointers(
        ArrayRef<uint32_t> weightsPtr, uint64_t sparsityPtr, vpux::ShapeRef offsets, uint64_t weightsTableSize,
        uint64_t weightsElemBitSize, VPUIP::SparsityCompressionAttr weightsCompression, uint64_t channelOffset,
        uint64_t originalOC) {
    return addTransformation(Const::RelocateWeightsTableAttr::get(
            getIntArrayAttr(getContext(), weightsPtr), getIntAttr(getContext(), sparsityPtr),
            getIntArrayAttr(getContext(), offsets), getIntAttr(getContext(), weightsTableSize),
            getIntAttr(getContext(), weightsElemBitSize), weightsCompression, getIntAttr(getContext(), channelOffset),
            getIntAttr(getContext(), originalOC)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::swizzleConstant(uint64_t swizzleKey, uint64_t arch) {
    return addTransformation(
            Const::SwizzleConstantAttr::get(getIntAttr(getContext(), swizzleKey), getIntAttr(getContext(), arch)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::add(double bias) {
    return addTransformation(Const::AddAttr::get(getFPAttr(getContext(), bias)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::reshape(vpux::ShapeRef newShape) {
    return addTransformation(Const::ReshapeAttr::get(getIntArrayAttr(getContext(), newShape)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::reverse(Dim axis) {
    return addTransformation(Const::ReverseAttr::get(getIntAttr(getContext(), axis.ind())));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::reorder(const vpux::DimsOrder& newOrder) {
    return addTransformation(Const::ReorderAttr::get(mlir::AffineMapAttr::get(newOrder.toAffineMap(getContext()))));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::padWithZero(vpux::ShapeRef padBefore, vpux::ShapeRef padAfter) {
    return addTransformation(Const::PadWithZeroAttr::get(getIntArrayAttr(getContext(), padBefore),
                                                         getIntArrayAttr(getContext(), padAfter)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::subview(vpux::ShapeRef offset, vpux::ShapeRef shape) {
    return addTransformation(
            Const::SubViewAttr::get(getIntArrayAttr(getContext(), offset), getIntArrayAttr(getContext(), shape)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::transpose(const vpux::DimsOrder& newOrder) {
    return addTransformation(Const::TransposeAttr::get(mlir::AffineMapAttr::get(newOrder.toAffineMap(getContext()))));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::memPermute(const vpux::DimsOrder& dstOrder,
                                                                  const vpux::DimsOrder& memPerm) {
    return addTransformation(Const::MemPermuteAttr::get(mlir::AffineMapAttr::get(dstOrder.toAffineMap(getContext())),
                                                        mlir::AffineMapAttr::get(memPerm.toAffineMap(getContext()))));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::layoutCast(const vpux::DimsOrder& dstOrder) {
    return addTransformation(Const::LayoutCastAttr::get(mlir::AffineMapAttr::get(dstOrder.toAffineMap(getContext()))));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::expandDilated(vpux::ShapeRef dilations) {
    return addTransformation(Const::ExpandDilatedAttr::get(getIntArrayAttr(getContext(), dilations)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::getSparsityMap() {
    return addTransformation(Const::GetSparsityMapAttr::get(getContext()));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::sparsify(bool compressOutputType,
                                                                mlir::ElementsAttr numActualElements) {
    return addTransformation(
            Const::SparsifyAttr::get(mlir::BoolAttr::get(getContext(), compressOutputType), numActualElements));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::changeShapeAndElemType(vpux::ShapeRef newShape,
                                                                              mlir::Type newElemType) {
    return addTransformation(
            Const::ChangeShapeAndElemTypeAttr::get(getIntArrayAttr(getContext(), newShape), newElemType));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::scalarMultInverse() {
    return addTransformation(Const::ScalarMultInverseAttr::get(getContext()));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::fuse(mlir::RankedTensorType fusedTensorType,
                                                            const ContentAttr& weightsTable, const ContentAttr& weights,
                                                            const ContentAttr& sparsity,
                                                            const ContentAttr& activations) {
    return addTransformation(
            Const::FuseWeightsAttr::get(getContext(), fusedTensorType, weightsTable, weights, sparsity, activations));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::fuse(mlir::RankedTensorType fusedTensorType,
                                                            const std::vector<ContentAttr>& constants) {
    return addTransformation(
            Const::FuseAttr::get(getContext(), fusedTensorType, ContentArrayAttr::get(getContext(), constants)));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::quantize(mlir::quant::QuantizedType newElemType) {
    return addTransformation(Const::QuantizeAttr::get(getContext(), newElemType));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::affineReshape(mlir::ArrayAttr dimMapping,
                                                                     mlir::ArrayAttr shapeValue) {
    return addTransformation(Const::AffineReshapeAttr::get(dimMapping, shapeValue));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::interpolate(mlir::ArrayAttr axes, mlir::ArrayAttr sizes,
                                                                   mlir::StringAttr mode, mlir::StringAttr coordMode,
                                                                   mlir::StringAttr nearestMode,
                                                                   mlir::BoolAttr antialias, mlir::ArrayAttr padsBegin,
                                                                   mlir::ArrayAttr padsEnd, mlir::FloatAttr cubeCoeff) {
    return addTransformation(Const::InterpolateAttr::get(axes, sizes, mode, coordMode, nearestMode, antialias,
                                                         padsBegin, padsEnd, cubeCoeff));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::gatherElements(mlir::IntegerAttr axisAttr,
                                                                      mlir::DenseElementsAttr indicesAttr) {
    return addTransformation(Const::GatherElementsAttr::get(axisAttr, indicesAttr));
}
template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::cumSum(mlir::IntegerAttr axis, mlir::BoolAttr exclusive,
                                                              mlir::BoolAttr reverse) {
    return addTransformation(Const::CumSumAttr::get(axis, exclusive, reverse));
}

template <typename T>
SpecializedContentSetup<T> SpecializedContentSetup<T>::logicalNot() {
    return addTransformation(Const::LogicalNotAttr::get(getContext()));
}

mlir::ElementsAttr getDummyBaseContent(mlir::MLIRContext* ctx) {
    // hardcode a very specific element type - insane type
    // that is guaranteed to never be present in reality
    auto elemType = mlir::Float80Type::get(ctx);

    // Zero-shaped tensor: holds no data. Used as a placeholder base for transformations
    // that carry their own data (e.g. ConcatAttr). Accessing its memory is undefined.
    auto tensorType = mlir::RankedTensorType::get({0}, elemType);
    return mlir::DenseElementsAttr::getFromRawBuffer(tensorType, ArrayRef<char>{});
}

bool isDummyBaseContent(mlir::ElementsAttr baseContent) {
    return baseContent == getDummyBaseContent(baseContent.getContext());
}

ContentAttr createConcatContentAttr(mlir::MLIRContext* ctx, mlir::ArrayAttr staticOffsets, int64_t axis,
                                    mlir::ArrayRef<ContentAttr> inputContents) {
    const auto baseContent = getDummyBaseContent(ctx);
    auto axisAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), axis);
    auto concatAttr = Const::ConcatAttr::get(ctx, axisAttr, staticOffsets, ContentArrayAttr::get(ctx, inputContents));
    return ContentAttr::get(baseContent, ArrayRef<TransformAttrInterface>{concatAttr});
}

// Note: this lists explicit template instantiations. we know exactly how many
// versions of this template should exist in the compiler, so we can hide
// certain implementation details in this translation unit, simplifying the the
// C++ compilation process.
template class SpecializedContentSetup<detail::NoopGet>;
template class SpecializedContentSetup<ContentAttr::SpecialSetupCallable>;

}  // namespace vpux::Const

namespace {
void enqueueCache(mlir::MLIRContext* ctx, const vpux::Const::ContentAttr& attr) {
#ifdef BACKGROUND_FOLDING_ENABLED
    auto& cacheManager = vpux::Const::ConstantFoldingCacheManager::getInstance();
    if (cacheManager.contains(ctx)) {
        auto& cache = cacheManager.get(ctx);
        cache.enqueueRequest(vpux::Const::FoldingRequest{attr, nullptr});
    }
#else
    std::ignore = ctx;
    std::ignore = attr;
#endif
}
}  // namespace

vpux::Const::ContentAttr vpux::Const::ContentAttr::get(mlir::ElementsAttr baseContent,
                                                       const Const::detail::ContentSetupBase& setup) {
    auto resultAttr = ContentAttr::get(baseContent, setup.getTransformations());
    enqueueCache(setup.getContext(), resultAttr);
    return resultAttr;
}
