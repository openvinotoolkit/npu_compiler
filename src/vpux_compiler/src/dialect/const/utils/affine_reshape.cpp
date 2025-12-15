//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <numeric>

using namespace vpux;

std::optional<mlir::Type> vpux::Const::inferElemTypeAffineReshape(ShapeRef inputShape, mlir::Type inputElementType,
                                                                  const SmallVector<SmallVector<int64_t>>& dimMapping,
                                                                  ArrayRef<int64_t> shapeValue) {
    const auto perAxisQType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>(inputElementType);
    if (perAxisQType == nullptr) {
        return inputElementType;
    }

    const auto inputQAxis = perAxisQType.getQuantizedDimension();
    const auto outputShape = shapeValue;

    // get output dims for input Q axis
    const auto& outputDims = dimMapping[inputQAxis];
    int64_t outQAxis = -1;
    int64_t inputQAxisSize = inputShape.raw()[inputQAxis];

    if (inputQAxisSize == 1) {
        // Convert per-axis quantized type to per-tensor quantized type
        return mlir::quant::UniformQuantizedType::get(
                perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
                perAxisQType.getScales().front(), perAxisQType.getZeroPoints().front(),
                perAxisQType.getStorageTypeMin(), perAxisQType.getStorageTypeMax());
    }

    for (const auto& dim : outputDims) {
        if (inputQAxisSize == outputShape[dim]) {
            // firstly check that element is unique and others == 1
            if (std::find_if(outputDims.begin(), outputDims.end(), [&](int64_t elem) {
                    return (outputShape[elem] != 1 && outputShape[elem] != inputQAxisSize);
                }) != outputDims.end()) {
                return std::nullopt;
            }
            outQAxis = dim;
            break;
        }
    }

    if (outQAxis == -1) {
        return std::nullopt;
    }

    if (const auto perAxisQuantileQType =
                mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(inputElementType)) {
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisQuantileQType.getFlags(), perAxisQuantileQType.getStorageType(),
                perAxisQuantileQType.getQuantileType(), perAxisQuantileQType.getExpressedType(),
                perAxisQuantileQType.getQuantiles(), perAxisQuantileQType.getScales(),
                perAxisQuantileQType.getZeroPoints(), static_cast<int32_t>(outQAxis),
                perAxisQuantileQType.getStorageTypeMin(), perAxisQuantileQType.getStorageTypeMax());
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisQType.getFlags(), perAxisQType.getStorageType(), perAxisQType.getExpressedType(),
            perAxisQType.getScales(), perAxisQType.getZeroPoints(), static_cast<int32_t>(outQAxis),
            perAxisQType.getStorageTypeMin(), perAxisQType.getStorageTypeMax());
}

std::optional<mlir::Type> vpux::Const::backInferElemTypeAffineReshape(
        ShapeRef inShape, mlir::Type outputElemType, const SmallVector<SmallVector<int64_t>>& dimMapping,
        ArrayRef<int64_t> shapeValue) {
    auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(outputElemType);
    if (perAxisType == nullptr) {
        return outputElemType;
    }

    const auto outAxis = static_cast<int64_t>(perAxisType.getQuantizedDimension());
    if (shapeValue[outAxis] == 1) {
        // Corner case: if output quantization axis has 1 value, the quantization is actually
        // per tensor; don't propagate, special handling might be needed.
        return std::nullopt;
    }

    // Find input indices that contain the output quantization axis.
    // There are 2 possible scenarios:
    //    1. in_axis = out_quant_axis * other_dim_val0 * other_dim_val1 * ...
    //    2. out_quant_axis = in_dim_val0 * in_dim_val1 * ...
    // For propagation to occur, we must find input axis in_quant_axis, such that:
    //    in_shape[in_quant_axis] = out_shape[out_quant_axis]
    // Therefore, in the 2 scenarios above we need:
    //    1. other_dim_val0 * other_dim_val1 * ... = 1
    //    2. only one of in_dim_val0, in_dim_val1 * ... can be != 1
    SmallVector<int64_t> inNonOneIndices;
    for (auto inIdx : irange(inShape.size())) {
        const auto inDimVal = inShape[Dim(inIdx)];
        // Skip 1 values for input, they will not be the input quantization axis
        if (inDimVal == 1) {
            continue;
        }

        if (llvm::find(dimMapping[inIdx], outAxis) != dimMapping[inIdx].end()) {
            // Collect all output dims that make up the input dim with out_quant_axis in the mapping
            // Equivalent to collecting other_dim_val0, other_dim_val1 ...
            SmallVector<int64_t> mappedShape;
            std::transform(dimMapping[inIdx].begin(), dimMapping[inIdx].end(), std::back_inserter(mappedShape),
                           [&shapeValue](auto oIdx) {
                               return shapeValue[oIdx];
                           });

            // Ensures that the output sub-shape has the same number of elements as in_shape[in_quant_axis] in
            // total. This ensures that we don't fuse the q dimension with other dimensions that are >1.
            if (std::accumulate(mappedShape.begin(), mappedShape.end(), 1, std::multiplies{}) != inDimVal) {
                return std::nullopt;
            }

            // We know the product of the mapped shape is in_shape[in_quant_axis] and then try to find the single
            // dimension that is in_shape[in_quant_axis]. If we can find it, we automatically know that all other
            // output dimensions are 1.
            if (llvm::find(mappedShape, inDimVal) == mappedShape.end()) {
                return std::nullopt;
            }

            // Collect all input indices that contain the outAxis in their mapping and are != 1
            // Equivalent to collecting in_dim_val0, in_dim_val1 ...
            inNonOneIndices.push_back(static_cast<int64_t>(inIdx));
        }
    }

    // If there is more than one non-one input index:
    //    1. Cannot have more than 1 input index, as we are "merging" output dims into one input dim.
    //    2. There are more than one input dims != 1.
    //  We cannot propagate the quant axis up.
    if (inNonOneIndices.size() != 1) {
        return std::nullopt;
    }

    const auto qInAxis = inNonOneIndices.front();
    if (const auto perAxisQuantileQType =
                mlir::dyn_cast_or_null<mlir::quant::QuantileQuantizedPerAxisType>(outputElemType)) {
        return mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisQuantileQType.getFlags(), perAxisQuantileQType.getStorageType(),
                perAxisQuantileQType.getQuantileType(), perAxisQuantileQType.getExpressedType(),
                perAxisQuantileQType.getQuantiles(), perAxisQuantileQType.getScales(),
                perAxisQuantileQType.getZeroPoints(), static_cast<int32_t>(qInAxis),
                perAxisQuantileQType.getStorageTypeMin(), perAxisQuantileQType.getStorageTypeMax());
    }

    return mlir::quant::UniformQuantizedPerAxisType::get(
            perAxisType.getFlags(), perAxisType.getStorageType(), perAxisType.getExpressedType(),
            perAxisType.getScales(), perAxisType.getZeroPoints(), static_cast<int32_t>(qInAxis),
            perAxisType.getStorageTypeMin(), perAxisType.getStorageTypeMax());
}

std::optional<vpux::DimsOrder> vpux::Const::inferAffineReshapeOutputLayout(const DimArr& inPerm,
                                                                           mlir::ArrayAttr dimMapAttr) {
    VPUX_THROW_UNLESS(dimMapAttr != nullptr, "dimMapAttr is nullptr");
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(dimMapAttr);
    SmallVector<vpux::Dim> perm;

    // Iterate over input dims in the given order and push back corresponding output dims as indicated by the op's
    // dim_mapping. The result is the permutation of output dims.
    bool layoutInferFail = false;
    for (auto pIt = inPerm.begin(); pIt != inPerm.end(); ++pIt) {
        const auto& outputDims = dimMapping[pIt->ind()];
        for (const auto& dim : outputDims) {
            const auto outDim = vpux::Dim(dim);

            // Ensure input dim order is not switched.
            // E.g. nchw -> c'h'w', with n = c', c = h', h * w = w'
            // Layouts 0123 and 0132 would both produce 012 output layout, but
            // the content of w' would not be the same.
            if (!perm.empty() && perm.back() == outDim) {
                layoutInferFail = std::prev(pIt)->ind() > pIt->ind();
                if (layoutInferFail) {
                    return std::nullopt;
                }

                continue;
            }
            perm.push_back(outDim);
        }
    }

    // Check that the resulting output permutation does not have duplicate dims
    SmallVector<vpux::Dim> temp(perm);
    llvm::sort(temp.begin(), temp.end(), [](const vpux::Dim& dim0, const vpux::Dim& dim1) {
        return dim0.ind() < dim1.ind();
    });

    if (std::adjacent_find(temp.begin(), temp.end()) != temp.end()) {
        return std::nullopt;
    }

    return DimsOrder::fromPermutation(ArrayRef(perm));
}

/// minCorner and maxCorner are two subscripts that represent the minimum vertex and the maximum vertex of
/// a hyper-rectangle in some n-dimension space (n = minCorner.size()). This function computes the volume of
/// that hyper-recangle if minCorner[i] <= maxCorner[i] for indices i.
/// If minCorner is not less or equal than maxCorner (i.e. for some index i minCorner[i] > maxCorner[i]),
/// the computed volume will be 0.
int64_t volume(ArrayRef<int64_t> minCorner, ArrayRef<int64_t> maxCorner) {
    assert(minCorner.size() == maxCorner.size());
    int64_t v = 1;
    for (const auto [mn, mx] : llvm::zip(minCorner, maxCorner)) {
        // Clip negative values to 0 resulting in a volume of 0 and signaling invalid values minCorner and maxCorner.
        v *= std::max<int64_t>(0, mx - mn + 1);
    }
    return v;
}

/// Computes the volume of the tensor which is represented by shape.
int64_t volume(ArrayRef<int64_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{});
}

/// This class provides some utility functions to quickly compute linear indices from subscripts and vice versa.
class ShapeHelper {
public:
    ShapeHelper(ArrayRef<int64_t> shape): _shape(shape) {
    }

    /// Returns the linear index in the flattened array given the subscript.
    /// For example for _shape = {2, 3}:
    ///   sub2ind({0, 0}) = 0
    ///   sub2ind({1, 1}) = 4
    /// This function is inspired by Matlab's sub2ind (https://www.mathworks.com/help/matlab/ref/sub2ind.html).
    int64_t sub2ind(ArrayRef<int64_t> sub) const {
        assert(sub.size() == _shape.size());
        return getMemIndex1D(MemShapeRef(sub), MemShapeRef(_shape));
    }

    /// Returns the subscript for a given linear index.
    /// For example for _shape = {2, 3}:
    ///   ind2sub(0) = 0
    ///   ind2sub(4) = {1, 1}
    /// This function is inspired by Matlab's ind2sub (https://www.mathworks.com/help/matlab/ref/ind2sub.html).
    SmallVector<int64_t> ind2sub(int64_t ind) const {
        assert(0 <= ind && ind < getVolume());
        return getMemIndexND(ind, MemShapeRef(_shape)).raw();
    }

    /// Returns the [from, to) slice of the underlying shape.
    /// For example for _shape = {2, 3, 4, 5, 6} and [from, to) = [1, 3) it returns the sub-shape {3, 4} and recomputes
    /// the strides for that shape.
    ShapeHelper slice(size_t from, size_t to) const {
        if (from >= to) {
            return ShapeHelper({});
        }
        return ShapeHelper(_shape.slice(from, to - from));
    }

    int64_t getVolume() const {
        return volume(_shape);
    }

    ArrayRef<int64_t> getShape() const {
        return _shape;
    }

    bool isEmpty() const {
        return _shape.empty();
    }

private:
    // As this class is a utility and only used in this context it does *not* have to own the underlying shapes. This
    // saves us some copies.
    ArrayRef<int64_t> _shape;
};

/// Maps the subscript sub (that lives in the space of srcShape) into the space of dstShape.
SmallVector<int64_t> mapSub(const ShapeHelper& dstShape, const ShapeHelper& srcShape, ArrayRef<int64_t> sub) {
    return dstShape.ind2sub(srcShape.sub2ind(sub));
}

/// Returns true iff the hyper-rectangle defined by minCorner and maxCorner only contains indices that are contiguous in
/// the given shape.
bool isDenseIn(const ShapeHelper& shape, ArrayRef<int64_t> minCorner, ArrayRef<int64_t> maxCorner) {
    const auto delta = shape.sub2ind(maxCorner) - shape.sub2ind(minCorner) + 1;
    return delta == volume(minCorner, maxCorner);
}

struct Indices {
    int64_t fromLo, fromHi;
    int64_t toLo, toHi;
};

/// Given fromShape and toShape this function computes that part in both shapes that differ from each other.
/// Example:
///   For fromShape = {2, 3, 9, 5}, toShape = {2, 27, 5} it would return the indices that select {3, 9} from froShape
///   and that select {27} from toShape.
Indices findNonTrivialSubShape(ArrayRef<int64_t> fromShape, ArrayRef<int64_t> toShape) {
    auto [fromBegin, toBegin] = std::mismatch(fromShape.begin(), fromShape.end(), toShape.begin(), toShape.end());
    auto [fromEnd, toEnd] = std::mismatch(fromShape.rbegin(), fromShape.rend(), toShape.rbegin(), toShape.rend());

    return {std::distance(fromShape.begin(), fromBegin), std::distance(fromShape.begin(), fromEnd.base()),
            std::distance(toShape.begin(), toBegin), std::distance(toShape.begin(), toEnd.base())};
}

/// clang-format off
///    Returns true when minCorner and maxCorner (that are subscripts in toShape) are translated to subscripts in
///    fromShape create a *contiguous rectangular region* (CRR) in fromShape.
///
///    Abbreviations and definitions:
///        min' = minCorner
///        max' = maxCorner
///
///        From hereon we assume that we have a reshape from shape s1 x ... x sm to shape s1' x ... x sn'.
///        min' and max' are subscripts for the shape s'.
///
///        A subset M of the set of tensor subscripts for some shape satisfies the CRR property if there are subscripts
///        mn, mx such that M is equal to the set { m | mn <= m <= mx }. Here the comparison a <= b is true if for all
///        entries a_i <= b_i. This induces a partial ordering on subscripts.
///
///    By definition of the reshape operation the volumes of s and s' are identical. This means we can translate
///    subscripts from s' to s the following way:
///        x = ind2sub(s, sub2ind(s', x'))
///    where x' is some subscript in s'. So the core idea of computing the bounding corners of the CRR in s is
///    by mapping min' and max' to s:
///        min = ind2sub(s, sub2ind(s', min'))
///        max = ind2sub(s, sub2ind(s', max'))
///
///    However, there are some caveats.
///
///    Because indices might not align, we can have wrapping and holes in the source CRR. For example look at:
///        [(0, 0),  (0, 1)*, (0, 2)]  --Reshape--> [(0, 0), (0, 1)*]
///        [(1, 0)*, (1, 1), (1, 2)]                [(1, 0), (1, 1)*]
///                                                 [(2, 0), (2, 1) ]
///    We can clearly see that the CRR (marked by (*)) in the output shape does *not* map to a CRR in the inut shape.
///
///    We can distill some provable conditions that guarantee that the CRR in the output is mappable to a CRR in the
///    input. The other way around is not necessarily true though.
///
///    The CRR in the output is mappable to a CRR in the input if
///        - The CRR in the output is dense
///        - volume(min, max) = volume(min', max')
///
///    This is a very strong condition and prevents us from optimizing a lot of cases. However, a weakened condition
///    only regarding certain sub-shapes of s and s' is enough to reject all invalid cases while extending the
///    applicability to valid cases.
///
///    Given shapes s = s1 x ... x sm and s' = s1' x ... x sn' we first want to find a sub-shape that is non-trivial.
///    This means we want to find k, l, k', l' such that
///      s  = xy abc zw
///             k   l
///      s' = xy de zw
///             k' l'
///
///    I.e. there are some sub parts abc, de in s and s' that are different in each shape (but with the same volume).
///    The sub shapes abc and de are called the non-trivial sub-shapes of s and s'.
///
///    The CRR in the output is mappable to a CRR in the input if
///        - The subset of CRR in the input over sub-shape s(k')' x ... x s(l')' has the CRR property
///
///    Here we can use our previous condition. I.e. to check if we can perform a CRR mapping, we have to check if the
///    CRR over the non-trivial sub-shape is dense AND the min/max corners in the input enclose an equally sized volume.
/// clang-format on
bool doesSatisfyContiguousRectangularRegion(const ShapeHelper& fromShape, const ShapeHelper& toShape,
                                            ArrayRef<int64_t> minCorner, ArrayRef<int64_t> maxCorner) {
    assert(minCorner.size() == maxCorner.size() && minCorner.size() == toShape.getShape().size());
    const auto subShapes = findNonTrivialSubShape(fromShape.getShape(), toShape.getShape());

    const auto fromSubShape = fromShape.slice(subShapes.fromLo, subShapes.fromHi);
    const auto toSubShape = toShape.slice(subShapes.toLo, subShapes.toHi);

    // Sub-shapes of volume 1 are always dense! No need to proceed.
    if (fromSubShape.getVolume() == 1 || toSubShape.getVolume() == 1) {
        return true;
    }

    // Extract the parts of the min and max corners that are admissible in the sub-shape of toShape.
    const auto toMinCorner = minCorner.slice(subShapes.toLo, subShapes.toHi);
    const auto toMaxCorner = maxCorner.slice(subShapes.toLo, subShapes.toHi);

    const auto isOutputDense = isDenseIn(toSubShape, toMinCorner, toMaxCorner);
    if (!isOutputDense) {
        return false;
    }

    const auto fromMinCorner = mapSub(fromSubShape, toSubShape, toMinCorner);
    const auto fromMaxCorner = mapSub(fromSubShape, toSubShape, toMaxCorner);

    // There is a little edge case here: fromMinCorner <= fromMaxCorner is not necessarily true.
    // In such a case volume(fromMinCorner, fromMaxCorner) will be 0 and != volume(subMinCorner, subMaxCorner)
    // because subMinCorner <= subMaxCorner is always true and therefore volume(subMinCorner, subMaxCorner) >= 1.
    assert(volume(toMinCorner, toMaxCorner) >= 1);
    return volume(fromMinCorner, fromMaxCorner) == volume(toMinCorner, toMaxCorner);
}

/// A helper class to manage the clustering of AffineReshape's dimMapping and with a bunch of utilities to extract and
/// manage slices of shapes.
class SwappingHelper {
public:
    SwappingHelper(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
                   const SmallVector<SmallVector<int64_t>>& dimMapping, ArrayRef<int64_t> subViewOffset,
                   ArrayRef<int64_t> subViewShape)
            : _inputShape(inputShape),
              _outputShape(outputShape),
              _subViewOffset(subViewOffset),
              _subViewShape(subViewShape),
              _newOffset(inputShape.size(), -1),
              _newShape(inputShape.size(), -1) {
        constexpr auto identity = [](auto x) {
            return x;
        };

        // 1.) Fill dimsToDims with the initial values (called clusters):
        //       [0] -> [0, 1]
        //       [1] -> [2]
        //       [2] -> [1, 2]
        // 2.) Check if [0] -> [0, 1] can be merged with any other cluster. [2] -> [1, 2] has common elements
        //     with [0] -> [0, 1], namely 1 in the output dimension. These clusters are merged:
        //       [0, 2] -> [0, 1, 2]
        //       [1] -> [2]
        // 2.) Now, [1] -> [2] shares elements with [0, 2] -> [0, 1, 2], namely 2 in the output dimension. These
        //     clusers are merged:
        //       [0, 2, 1] -> [0, 1, 2]
        // 3.) No more clusters can be merged!
        for (size_t dimIndex = 0; dimIndex < inputShape.size(); ++dimIndex) {
            const auto& mapping = dimMapping[dimIndex];
            _dimsToDims.emplace_back(SmallVector<int64_t>{static_cast<int64_t>(dimIndex)}, mapping);
        }

        // merge clusters until all clusters are merged
        auto it = _dimsToDims.begin();
        while (it != _dimsToDims.end()) {
            auto& [fromDims, toDims] = *it;

            bool wereClustersMerged = false;
            for (auto otherIt = std::next(it); otherIt < _dimsToDims.end(); otherIt++) {
                auto& [otherFromDims, otherToDims] = *otherIt;

                // If there is an intersection in the input dimensions set or and intersection in the output
                // dimensions set, then they belong to the same clusters and we merge.
                const auto otherFromContained =
                        llvm::map_range(otherFromDims, [fromDims = std::ref(fromDims)](int64_t index) -> bool {
                            return llvm::is_contained(fromDims.get(), index);
                        });

                const auto otherToContained =
                        llvm::map_range(otherToDims, [toDims = std::ref(toDims)](int64_t index) -> bool {
                            return llvm::is_contained(toDims.get(), index);
                        });

                const auto isSameCluster =
                        llvm::any_of(otherFromContained, identity) || llvm::any_of(otherToContained, identity);

                // merge clusters
                if (isSameCluster) {
                    for (auto [otherFromDim, isContained] : llvm::zip(otherFromDims, otherFromContained)) {
                        if (!isContained) {
                            fromDims.push_back(otherFromDim);
                        }
                    }

                    for (auto [otherToDim, isContained] : llvm::zip(otherToDims, otherToContained)) {
                        if (!isContained) {
                            toDims.push_back(otherToDim);
                        }
                    }

                    _dimsToDims.erase(otherIt);
                    wereClustersMerged = true;
                    break;
                }
            }

            if (wereClustersMerged) {
                it = _dimsToDims.begin();
            } else {
                it++;
            }
        }
    }

    SmallVector<int64_t> getFromShape(size_t clusterIndex) const {
        return select(_inputShape, std::get<0>(_dimsToDims[clusterIndex]));
    }

    SmallVector<int64_t> getToShape(size_t clusterIndex) const {
        return select(_outputShape, std::get<1>(_dimsToDims[clusterIndex]));
    }

    SmallVector<int64_t> getToMinCorner(size_t clusterIndex) const {
        return select(_subViewOffset, std::get<1>(_dimsToDims[clusterIndex]));
    }

    SmallVector<int64_t> getToMaxCorner(size_t clusterIndex) const {
        const auto& to = std::get<1>(_dimsToDims[clusterIndex]);
        SmallVector<int64_t> result(to.size());
        for (size_t i = 0; i < to.size(); ++i) {
            result[i] = _subViewOffset[to[i]] + _subViewShape[to[i]] - 1;
        }
        return result;
    }

    size_t getClusterCount() const {
        return _dimsToDims.size();
    }

    void assignNewOffset(size_t clusterIndex, ArrayRef<int64_t> newMinCorner) {
        const auto& from = std::get<0>(_dimsToDims[clusterIndex]);
        assert(from.size() == newMinCorner.size());
        for (size_t i = 0; i < newMinCorner.size(); ++i) {
            _newOffset[from[i]] = newMinCorner[i];
        }
    }

    void assignNewShape(size_t clusterIndex, ArrayRef<int64_t> newMinCorner, ArrayRef<int64_t> newMaxCorner) {
        const auto& from = std::get<0>(_dimsToDims[clusterIndex]);
        assert(from.size() == newMinCorner.size());
        assert(from.size() == newMaxCorner.size());
        for (size_t i = 0; i < newMaxCorner.size(); ++i) {
            // go from inclusive bounds to size again
            _newShape[from[i]] = newMaxCorner[i] - newMinCorner[i] + 1;
        }
    }

    SmallVector<int64_t> getNewOffset() const {
        VPUX_THROW_UNLESS(llvm::none_of(_newOffset, isMinusOne),
                          "New SubView offset contains a negative value -- this indicates a bug!");
        return _newOffset;
    }

    SmallVector<int64_t> getNewShape() const {
        VPUX_THROW_UNLESS(llvm::none_of(_newShape, isMinusOne),
                          "New SubView offset contains a negative value -- this indicates a bug!");
        return _newShape;
    }

private:
    static constexpr auto isMinusOne = [](int64_t x) -> bool {
        return x == -1;
    };

    /// Returns the values in input according to the indices in selector in order.
    /// For example
    ///   input = [1, 3, 5, 7], selector = [0, 1, 3, 0]
    /// will return
    ///   [1, 3, 7, 1]
    static SmallVector<int64_t> select(ArrayRef<int64_t> input, ArrayRef<int64_t> selector) {
        SmallVector<int64_t> result(selector.size());
        for (size_t i = 0; i < selector.size(); ++i) {
            result[i] = input[selector[i]];
        }
        return result;
    }

    // List of "(from dimension, to dimension)" tuples
    // A dimMapping of [[0, 1], [3], [2], [2]] has an equivalent representation in _dimsToDims:
    // [0]    -> [0, 1]
    // [1]    -> [3]
    // [2, 3] -> [2]
    // The numbers in these vectors are dimension indices. All vectors on the left are the "from"
    // dimension and all vectors on the right are the "to" dimension. This representation
    // makes it much easier to take slices of shapes that are fused or split.
    SmallVector<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>>> _dimsToDims;

    // DimensionClusters does *not* have to own these.
    ArrayRef<int64_t> _inputShape;
    ArrayRef<int64_t> _outputShape;
    ArrayRef<int64_t> _subViewOffset;
    ArrayRef<int64_t> _subViewShape;

    // The new offset and shape if SubView and be ordered before AffineReshape.
    SmallVector<int64_t> _newOffset;
    SmallVector<int64_t> _newShape;
};

/// clang-format off
///    For a #const.AffineReshape, #const.SubView pair this function attemps to order #const.SubView before to help
///    optimizing folding.
///
///    To understand how this function works, it makes sense to break down #const.AffineReshape into smaller pieces. An
///    AffineReshape is made up of a target shape and a dimension mapping. For example:
///        AffineReshape<[[0, 1], [3], [2], [2]], [2, 2, 42, 5]> : 4x5x6x7 -> 2x2x42x5
///    The 'dimMapping' attribute maps the input dimension to the output dimensions:
///        [0] -> [0, 1]
///        [1] -> [3]
///        [2] -> [2]
///        [3] -> [2]
///    If we cluster all dimensions on the left side and all dimensions on the right side together, such that all
///    clusters are disjunctive, we get:
///        [0]    -> [0, 1]
///        [1]    -> [3]
///        [2, 3] -> [2]
///    Or, expressing it using shapes:
///        4   -> 2x2
///        5   -> 5
///        6x7 -> 42
///    This means that AffineReshape can be understood as a "Transpose of little Reshapes".
///
///    Now, because swapping SubView with Transpose is trivial (we just have to swap the dimensions), we can focus on
///    how to solve the problem for these little Reshapes.
///
///    If every reshape can be swapped with the corresponding *part* of the SubView, then the whole SubView can be
///    ordered before the AffineReshape. If the SubView can be ordered before the Reshape is determined by
///    doesSatisfyContiguousRectangularRegion(). Please refer to its documentation to understand how it works.
/// clang-format on
mlir::FailureOr<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>>> vpux::Const::swapAffineReshapeAndSubView(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
        const SmallVector<SmallVector<int64_t>>& dimMapping, ArrayRef<int64_t> subViewOffset,
        ArrayRef<int64_t> subViewShape) {
    SwappingHelper helper(inputShape, outputShape, dimMapping, subViewOffset, subViewShape);

    // We now have computed our clusters as described above.
    //     Dimensions             Shapes
    //   [0]    -> [0, 1]   |   4   -> 2x2
    //   [1]    -> [3]      |   5   -> 5
    //   [2, 3] -> [2]      |   6x7 -> 42
    // We can now begin to look at each cluster and try to map the rectangular region in the output tensor to a
    // rectangular region
    //  in the input tensor.

    for (size_t clusterIndex = 0; clusterIndex < helper.getClusterCount(); ++clusterIndex) {
        const auto fromShape = helper.getFromShape(clusterIndex);
        const auto toShape = helper.getToShape(clusterIndex);
        const auto subViewMinCorner = helper.getToMinCorner(clusterIndex);
        const auto subViewMaxCorner = helper.getToMaxCorner(clusterIndex);

        // check some invariants
        {
            const auto fromVolume = volume(fromShape);
            const auto toVolume = volume(toShape);
            VPUX_THROW_UNLESS(fromVolume == toVolume,
                              "The input and output volume of these shapes is expected to match but got {0} and {1} -- "
                              "this should never happen and indicates bug!",
                              fromVolume, toVolume);
        }

        const ShapeHelper from(fromShape);
        const ShapeHelper to(toShape);

        const auto isCRR = doesSatisfyContiguousRectangularRegion(from, to, subViewMinCorner, subViewMaxCorner);
        if (!isCRR) {
            Logger::global().trace("Encountered non-CRR reshape {0} -> {1} with SubView [{2}, {3}]", fromShape, toShape,
                                   subViewMinCorner, subViewMaxCorner);
            return mlir::failure();
        }

        const auto newMinCorner = mapSub(from, to, subViewMinCorner);
        const auto newMaxCorner = mapSub(from, to, subViewMaxCorner);

        helper.assignNewOffset(clusterIndex, newMinCorner);
        helper.assignNewShape(clusterIndex, newMinCorner, newMaxCorner);
    }

    return std::tuple{helper.getNewOffset(), helper.getNewShape()};
}
