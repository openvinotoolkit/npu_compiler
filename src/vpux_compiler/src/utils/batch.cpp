//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/batch.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/common_string_utils.hpp"

#include <charconv>
#include <functional>
#include <iterator>

using namespace vpux;

bool DebatchedCallOpData::canBeDeserialized(const SmallVector<ValueType>& array) {
    constexpr uint32_t minElementsInArray = 2;
    return array.size() >= minElementsInArray;
}

DebatchedCallOpData DebatchedCallOpData::deserialize(const SmallVector<ValueType>& array) {
    VPUX_THROW_UNLESS(DebatchedCallOpData::canBeDeserialized(array),
                      "Cannot deserialzie DebatchedCallOpData from array. More elements expected, got {1}",
                      array.size());
    return DebatchedCallOpData{array[0], array[1]};
}

SmallVector<DebatchedCallOpData::ValueType> DebatchedCallOpData::serialize() const {
    return SmallVector<ValueType>({callOpIndex, totalBatchSize});
}

std::string DebatchedCallOpData::to_string() const {
    return "call Index: " + std::to_string(callOpIndex) + ", batch size: " + std::to_string(totalBatchSize);
}

const DebatchedCallOpData& DebatchedCallOpAttributeView::getCallData() const {
    return data;
}

std::optional<DebatchedCallOpAttributeView> DebatchedCallOpAttributeView::extract(mlir::func::CallOp callOp) {
    if (!callOp->hasAttr(DebatchedCallOpAttributeView::name())) {
        return {};
    }
    auto attr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(callOp->getAttr(DebatchedCallOpAttributeView::name()));
    VPUX_THROW_UNLESS(attr != nullptr, "Unexpected type for \"{0}\", only \"mlir::ArrayAttr\" supported",
                      DebatchedCallOpAttributeView::name());
    return DebatchedCallOpAttributeView(
            DebatchedCallOpData::deserialize(parseIntArrayAttr<DebatchedCallOpData::ValueType>(attr)));
}

void DebatchedCallOpAttributeView::injectImpl(mlir::func::CallOp callOp) const {
    auto serializedArray = data.serialize();

    auto debatchedAttr = getIntArrayAttr(callOp->getContext(), serializedArray);
    VPUX_THROW_UNLESS(debatchedAttr != nullptr, "Cannot create 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::name());
    callOp->setAttr(DebatchedCallOpAttributeView::name(), debatchedAttr);
}

bool DebatchedCallOpAttributeView::hasAvailableTilesAttr(mlir::func::CallOp callOp) {
    return callOp->hasAttr(DebatchedCallOpAttributeView::availableTilesAttrName());
}

void DebatchedCallOpAttributeView::setAvailableTilesAttr(mlir::func::CallOp callOp,
                                                         DebatchedCallOpData::ValueType val) {
    VPUX_THROW_UNLESS(!hasAvailableTilesAttr(callOp),
                      "Detected existing 'DebatchedCallOpAttributeView' attribute \"{0}\", cannot create new attribute",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    auto newAttr = getIntAttr(callOp->getContext(), val);

    VPUX_THROW_UNLESS(newAttr != nullptr, "Failed to create new 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    callOp->setAttr(DebatchedCallOpAttributeView::availableTilesAttrName(), newAttr);
}

void DebatchedCallOpAttributeView::removeAvailableTilesAttr(mlir::func::CallOp callOp) {
    VPUX_THROW_UNLESS(hasAvailableTilesAttr(callOp),
                      "'DebatchedCallOpAttributeView' attribute \"{0}\" not found, cannot remove",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    callOp->removeAttr(DebatchedCallOpAttributeView::availableTilesAttrName());
}

DebatchedCallOpData::ValueType DebatchedCallOpAttributeView::getAvailableTilesVal(mlir::func::CallOp callOp) {
    VPUX_THROW_UNLESS(hasAvailableTilesAttr(callOp), "'DebatchedCallOpAttributeView' attribute \"{0}\" not found",
                      DebatchedCallOpAttributeView::availableTilesAttrName());
    return static_cast<DebatchedCallOpData::ValueType>(
            mlir::cast<mlir::IntegerAttr>(callOp->getAttr(DebatchedCallOpAttributeView::availableTilesAttrName()))
                    .getValue()
                    .getSExtValue());
}

bool DebatchedCallOpAttributeView::hasReorderingAttr(mlir::func::CallOp callOp) {
    return callOp->hasAttr(DebatchedCallOpAttributeView::reorderingAttrName());
}

void DebatchedCallOpAttributeView::setReorderingAttr(mlir::func::CallOp callOp) {
    auto newAttr = getIntAttr(callOp->getContext(), 1);

    VPUX_THROW_UNLESS(newAttr != nullptr, "Failed to create new 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::reorderingAttrName());

    callOp->setAttr(DebatchedCallOpAttributeView::reorderingAttrName(), newAttr);
}

DebatchCoeffDescription DebatchCoeffDescription::createFromString(std::string_view descr) {
    VPUX_THROW_WHEN(descr.empty(), "DebatchCoeffDescription must be created from a non-empty description");
    VPUX_THROW_WHEN(descr.size() < 2,
                    "DebatchCoeffDescription must be started with \"[\" and finished with \"]\", got: {0}", descr);
    VPUX_THROW_UNLESS(descr[0] == '[' && descr.back() == ']',
                      "DebatchCoeffDescription must be started with \"[\" and finished with \"]\", got: {0}", descr);
    auto cend = descr.cbegin();
    std::advance(cend, descr.size() - 1);
    std::vector<int64_t> parsedValues;
    parsedValues.reserve(2);
    auto cbegin = descr.cbegin();
    cbegin++;
    vpux::splitRangeAndApply(cbegin, cend, '-', [&parsedValues](std::string_view item) {
        int64_t result = 0;
        auto [ptr, ec] = std::from_chars(item.data(), item.data() + item.size(), result);
        VPUX_THROW_UNLESS(ec == std::errc(), "Cannot convert string: {0} to a number", item);
        VPUX_THROW_UNLESS(result >= 0, "Each DebatchCoeffDescription mustn't be a negative: {0}", result);
        parsedValues.push_back(result);
    });
    VPUX_THROW_UNLESS(parsedValues.size() == 2,
                      "DebatchCoeffDescription expects the format \"[BatchPositionInShape-DesiredBatchValue]\"");
    return DebatchCoeffDescription{Dim{parsedValues[0]}, parsedValues[1]};
}

DebatchCoeffDescription DebatchCoeffDescription::createFromShapes(ShapeRef inShape, ShapeRef outShape) {
    VPUX_THROW_UNLESS(
            inShape.size() == outShape.size(),
            "DebatchCoeffDescription must be created from shapes with an equal rank, got shapes in: {0}, out: {1}",
            inShape, outShape);
    std::optional<DebatchCoeffDescription> coeff;
    for (size_t index = 0; index < inShape.size(); index++) {
        if (inShape[Dim{index}] != outShape[Dim{index}]) {
            VPUX_THROW_WHEN(coeff.has_value(),
                            "DebatchCoeffDescription must be created from shapes which differ only in N dimension, got "
                            "shapes in: {0}, out: {1}",
                            inShape, outShape);
            coeff = DebatchCoeffDescription{Dim{index}, outShape[Dim{index}]};
        }
    }

    if (!coeff.has_value()) {
        coeff = DebatchCoeffDescription{Dim{0}, outShape[Dim{0}]};
    }

    try {
        coeff->apply(inShape);
    } catch (std::exception& ex) {
        VPUX_THROW("DebatchCoeffDescription::createFromShapes failed due to the error: {0}", ex.what());
    }
    return *coeff;
}

Shape DebatchCoeffDescription::apply(ShapeRef shape) const {
    // class Shape should have the method "at()" which check a dimension value on correctness by itself
    VPUX_THROW_UNLESS(batchPositionIndex.ind() >= 0 && static_cast<size_t>(batchPositionIndex.ind()) < shape.size(),
                      "Dimension value: {0} is not apt for a shape: {1}", batchPositionIndex, shape);
    VPUX_THROW_UNLESS(shape[batchPositionIndex],
                      "Cannot apply batch conversion because shape: {0} has a zero batch in dimension: {1}", shape,
                      batchPositionIndex);
    VPUX_THROW_UNLESS(desiredBatchValue, "Cannot apply batch conversion because desiredBatchValue is zero");

    Shape retShape{shape};
    if (shape[batchPositionIndex] == mlir::ShapedType::kDynamic) {
        retShape[batchPositionIndex] = desiredBatchValue;
        return retShape;
    }
    int64_t remainder = retShape[batchPositionIndex] > static_cast<int64_t>(desiredBatchValue)
                                ? retShape[batchPositionIndex] % desiredBatchValue
                                : desiredBatchValue % retShape[batchPositionIndex];
    VPUX_THROW_WHEN(
            remainder,
            "Cannot get the desired batch value: {0} from a shape: {1}, where a batch is expected on the position: "
            "{2}. A division operation on those number produces the remainder: {3} which otherwise would be abandoned",
            desiredBatchValue, shape, batchPositionIndex, remainder);
    retShape[batchPositionIndex] = desiredBatchValue;
    return retShape;
}

Shape DebatchCoeffDescription::applyProportionFromShape(ShapeRef fromShape, ShapeRef toShape) const {
    VPUX_THROW_UNLESS(batchPositionIndex.ind() >= 0 && static_cast<size_t>(batchPositionIndex.ind()) < toShape.size(),
                      "Dimension value: {0} must be apt for the shape: {1} to apply proportions from the shape: {2}",
                      batchPositionIndex, toShape, fromShape);
    VPUX_THROW_UNLESS(toShape[batchPositionIndex],
                      "Cannot apply proportions cast because toShape : {0} has a zero batch", toShape);

    // Cannot extract proportion from kDynamic, thus apply the coefficient to the result instead
    if (fromShape[batchPositionIndex] == mlir::ShapedType::kDynamic) {
        return apply(toShape);
    }

    Shape modifiedReferenceShape = apply(fromShape);

    // When coeff asks for conversion to kDynamic, do nothing regardless of whether shapes have dynamic dims or not
    if (modifiedReferenceShape[batchPositionIndex] == mlir::ShapedType::kDynamic) {
        return toShape.raw();
    }

    // In other cases, when we can extract proportion using fromShape and coefficient,
    // lets use proportion cast
    // avoid overflow in case of dynamic
    Shape resShape = toShape.raw();
    if (resShape[batchPositionIndex] == mlir::ShapedType::kDynamic) {
        resShape[batchPositionIndex] = -1;
    }

    Shape origShape = fromShape.raw();
    if (modifiedReferenceShape[batchPositionIndex] >= origShape[batchPositionIndex]) {
        int64_t upcastRatio = modifiedReferenceShape[batchPositionIndex] / origShape[batchPositionIndex];
        VPUX_THROW_WHEN(
                modifiedReferenceShape[batchPositionIndex] % origShape[batchPositionIndex],
                "Cannot apply proportions upcast, because cannot get the desired batch value: {0} "
                "from a shape: {1}, where a batch is expected on the position: "
                "{2}. A division operation on those number produces the remainder: {3} which otherwise would be "
                "abandoned",
                desiredBatchValue, fromShape, batchPositionIndex,
                modifiedReferenceShape[batchPositionIndex] % fromShape[batchPositionIndex]);
        resShape[batchPositionIndex] *= upcastRatio;
    } else {
        int64_t downcastRatio = origShape[batchPositionIndex] / modifiedReferenceShape[batchPositionIndex];
        VPUX_THROW_WHEN(
                origShape[batchPositionIndex] % modifiedReferenceShape[batchPositionIndex],
                "Cannot apply proportions downcast, because cannot get the desired batch value: {0} "
                "from a shape: {1}, where a batch is expected on the position: "
                "{2}. A division operation on those number produces the remainder: {3} which otherwise would be "
                "abandoned",
                desiredBatchValue, fromShape, batchPositionIndex, fromShape[batchPositionIndex] % desiredBatchValue);
        if (downcastRatio < 0 || downcastRatio == mlir::ShapedType::kDynamic) {
            downcastRatio = -1;
        }
        // resShape must remain intact if N is supposed to disappeared after conversion.
        // Proceed only if result is not 0
        bool isResultZero = ((resShape[batchPositionIndex] / downcastRatio) == 0);
        VPUX_THROW_WHEN((resShape[batchPositionIndex] % downcastRatio) && !isResultZero,
                        "Cannot apply proportions downcast to the shape: {0} "
                        "from a shape: {1}, while a batch expected on the given position: "
                        "{2} and the given ratio: {3} cannot coexist without narrowing downcasting",
                        toShape, fromShape, batchPositionIndex, downcastRatio);
        if (!isResultZero) {
            resShape[batchPositionIndex] /= downcastRatio;
        }
    }
    if (resShape[batchPositionIndex] < 0) {
        resShape[batchPositionIndex] = mlir::ShapedType::kDynamic;
    }
    return resShape;
}

std::string DebatchCoeffDescription::to_string() const {
    return llvm::formatv("[{0}-{1}]", batchPositionIndex.ind(), desiredBatchValue).str();
}

std::optional<DebatchCoefficients> DebatchCoefficients::create(std::string_view tensorsCoeffFormatted) {
    if (tensorsCoeffFormatted.empty()) {
        return {};
    }
    DebatchCoefficients ret;
    vpux::splitRangeAndApply(tensorsCoeffFormatted.cbegin(), tensorsCoeffFormatted.cend(), ',',
                             [&ret](std::string_view item) {
                                 auto nameCoeffDivPos = item.find(":");
                                 std::string nodeName;
                                 std::string_view::iterator itBegin = item.begin();
                                 std::string_view::iterator itEnd = item.end();
                                 if (nameCoeffDivPos != std::string::npos) {
                                     nodeName = std::string(item.begin(), item.begin() + nameCoeffDivPos);
                                     itBegin += (nameCoeffDivPos + 1);
                                 }
                                 std::string_view coeffPairStr{&(*itBegin), static_cast<size_t>(itEnd - itBegin)};
                                 ret.orderedInputCoefficients.emplace(
                                         std::move(nodeName), DebatchCoeffDescription::createFromString(coeffPairStr));
                             });

    VPUX_THROW_UNLESS(ret.orderedInputCoefficients.size(),
                      "DebatchCoefficients must contain something once coefficient string: \"{0}\" has been parsed",
                      tensorsCoeffFormatted);
    auto isEmptyNodeName = [](const auto& node) {
        return node.first.empty();
    };
    size_t nonNamedNodeCount =
            std::count_if(ret.orderedInputCoefficients.begin(), ret.orderedInputCoefficients.end(), isEmptyNodeName);
    size_t namedNodeCount = std::count_if(ret.orderedInputCoefficients.begin(), ret.orderedInputCoefficients.end(),
                                          std::not_fn(isEmptyNodeName));
    VPUX_THROW_WHEN(namedNodeCount != 0 && nonNamedNodeCount != 0,
                    "DebatchCoefficients encloses both named: {0} and unnamed: {1} node types, "
                    "which is forbidden. Please specify absent names in nodes, or don't give names to all nodes at all",
                    namedNodeCount, nonNamedNodeCount);
    return ret;
}
Shape DebatchCoefficients::applyDefault(ShapeRef shape) {
    return DebatchCoeffDescription{}.apply(shape);
}

Shape DebatchCoefficients::apply(ShapeRef shape, size_t index) const {
    VPUX_THROW_UNLESS(index < orderedInputCoefficients.size(),
                      "DebatchCoefficients requested index: {0} must be lesser than tensor description count: {1}",
                      index, orderedInputCoefficients.size());
    auto [nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt] = orderedInputCoefficients.equal_range("");
    size_t nodesCount = std::distance(nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt);
    VPUX_THROW_WHEN(nodesCount != orderedInputCoefficients.size(),
                    "DebatchCoefficients must encompass only named nodes only or unnamed");
    std::advance(nodeCoeffDescriptionBeginIt, index);
    return nodeCoeffDescriptionBeginIt->second.apply(shape);
}

Shape DebatchCoefficients::apply(ShapeRef shape, const std::string& nodeName) const {
    VPUX_THROW_WHEN(nodeName.empty(),
                    "DebatchCoefficients cannot apply shape transformation on empty node name. Use index instead");
    auto [nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt] = orderedInputCoefficients.equal_range(nodeName);
    auto nodesCount = std::distance(nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt);
    VPUX_THROW_WHEN(nodesCount == 0, "DebatchCoefficients must contain node name: {0}", nodeName);
    VPUX_THROW_WHEN(nodesCount != 1,
                    "DebatchCoefficients has multiple nodes with the name: {0}, please use index instead", nodeName);
    return nodeCoeffDescriptionBeginIt->second.apply(shape);
}

size_t DebatchCoefficients::size() const {
    return orderedInputCoefficients.size();
}

/**
 * @brief Retrieves the coefficient description at the specified index.
 *
 * This function attempts to access the coefficient description from the
 * `orderedInputCoefficients` container based on the provided index. If the
 * index is out of bounds, an empty `std::optional` is returned.
 *
 * @param index The zero-based index of the coefficient to retrieve.
 * @return std::optional<DebatchCoeffDescription> containing the coefficient
 *         description if the index is valid, or an empty optional if the
 *         index is out of range.
 */
std::optional<DebatchCoeffDescription> DebatchCoefficients::getCoefficient(size_t index) const {
    if (index >= orderedInputCoefficients.size()) {
        return {};
    }

    auto it = orderedInputCoefficients.begin();
    std::advance(it, index);
    return it->second;
}

/**
 * @brief Retrieves the coefficient description associated with a given node name.
 *
 * This function searches for the coefficient description in the `orderedInputCoefficients`
 * map using the provided node name. If the node name is empty or not found in the map,
 * the function returns an empty optional.
 *
 * @param nodeName The name of the node for which the coefficient description is to be retrieved.
 *                 Must be a non-empty string.
 * @return std::optional<DebatchCoeffDescription> An optional containing the coefficient description
 *         if the node name is found in the map, or an empty optional if the node name is empty
 *         or not found.
 */
std::optional<DebatchCoeffDescription> DebatchCoefficients::getCoefficient(const std::string& nodeName) const {
    if (nodeName.empty()) {
        return {};
    }
    auto it = orderedInputCoefficients.find(nodeName);
    if (it == orderedInputCoefficients.end()) {
        return {};
    }
    return it->second;
}

std::string DebatchCoefficients::to_string(bool includeNodeNames) const {
    std::stringstream sstream;
    for (const auto& [nodeName, coeff] : orderedInputCoefficients) {
        if (includeNodeNames) {
            sstream << nodeName << ":";
        }
        sstream << coeff.to_string() << ",";
    }
    std::string ret = sstream.str();
    if (!ret.empty()) {
        ret.pop_back();
    }
    return ret;
}
