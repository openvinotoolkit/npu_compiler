//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <numeric>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace {
// replaces the -1 value with the inferred split length if it exists
bool inferLastSplitLength(mlir::MutableArrayRef<int64_t> splitLengths, int64_t inputAxisSize) {
    auto negIt = llvm::find(splitLengths, -1);
    if (negIt == splitLengths.end()) {
        return true;
    }

    // splitLengths contains a -1, which we just include in the sum for simplicity. To correct for that we add another 1
    // at the end.
    const auto sum = std::accumulate(splitLengths.begin(), splitLengths.end(), 0) + 1;
    *negIt = inputAxisSize - sum;
    return *negIt > 0;
}

int64_t getInferredAxis(int64_t axis, int64_t inputRank) {
    return (axis < 0) ? (axis + inputRank) : axis;
}
}  // namespace

mlir::LogicalResult vpux::IE::VariadicSplitOp::verify() {
    const auto inputType = getInput().getType();
    const auto rank = inputType.getRank();

    const auto axisAttr = getAxis();
    if (axisAttr < -rank || axisAttr >= rank) {
        return emitOpError(formatv("'axis' must be in the interval [{0}, {1}] but got {2}", -rank, rank - 1, axisAttr));
    }

    // negative values count from the back
    const auto axis = (axisAttr < 0) ? (axisAttr + rank) : axisAttr;

    auto splitLengths = parseIntArrayAttr<int64_t>(getSplitLengths());

    const auto allGreater = llvm::all_of(splitLengths, [](int64_t x) {
        return x >= -1;
    });
    if (!allGreater) {
        return emitOpError(formatv("all values in 'split_lengths' must be -1 or greater"));
    }

    const auto neg1Count = llvm::count(splitLengths, -1);
    if (neg1Count > 1) {
        return emitOpError(formatv("'split_lengths' can contain at most one -1 value"));
    }

    // replace -1 value because it's inconvenient to work with
    if (!inferLastSplitLength(splitLengths, inputType.getShape()[axis])) {
        return emitOpError(formatv("cannot infer a positive value for the -1 value in 'split_lengths'"));
    }

    const auto sum = std::accumulate(splitLengths.begin(), splitLengths.end(), 0l);
    if (sum != inputType.getShape()[axis]) {
        return emitOpError(
                formatv("entries in 'split_lengths' are expected to sum up to axis dimension but got {0}", sum));
    }

    const auto numOutputs = getOutputs().size();
    const auto numSplitLengths = splitLengths.size();
    if (numOutputs != numSplitLengths) {
        return emitOpError(formatv("number of outputs {0} does not match length of 'split_lengths' {1}", numOutputs,
                                   numSplitLengths));
    }

    // check if all output shapes are equal to the input shape except for the axis dimension
    for (size_t outIndex = 0; outIndex < getOutputs().size(); ++outIndex) {
        const auto output = getOutputs()[outIndex];
        const auto ndType = mlir::cast<NDTypeInterface>(output.getType());

        for (int64_t dimIndex = 0; dimIndex < ndType.getRank(); ++dimIndex) {
            const auto expected = (dimIndex != axis) ? inputType.getShape()[dimIndex] : splitLengths[outIndex];
            const auto got = ndType.getShape().raw()[dimIndex];

            if (got != expected) {
                return emitOpError(formatv("output {0} is expected to have size {1} for axis {2} but got {3}", outIndex,
                                           expected, dimIndex, got));
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::VariadicSplitOp::inferReturnTypeComponents(
        ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location, ::mlir::ValueShapeRange operands,
        ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::ShapedTypeComponents>& inferredReturnShapes) {
    IE::VariadicSplitOpAdaptor adaptor(operands, attributes, properties, regions);
    if (mlir::failed(adaptor.verify(location.value_or(mlir::UnknownLoc::get(context))))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<NDTypeInterface>(adaptor.getInput().getType());
    // negative values count from the back
    const auto axis = ::getInferredAxis(adaptor.getAxis(), inputType.getRank());

    auto newShape = SmallVector<int64_t>(inputType.getShape().raw());

    auto splitLengths = parseIntArrayAttr<int64_t>(adaptor.getSplitLengths());
    inferLastSplitLength(splitLengths, inputType.getShape().raw()[axis]);

    const auto outDesc = vpux::getTensorAttr(mlir::cast<mlir::RankedTensorType>(inputType));
    for (const auto splitLength : splitLengths) {
        newShape[axis] = splitLength;
        inferredReturnShapes.emplace_back(newShape, inputType.getElementType(), outDesc);
    }

    return mlir::success();
}

mlir::SmallVector<int64_t> vpux::IE::VariadicSplitOp::getInferredSplitLengths() {
    auto splitLengths = parseIntArrayAttr<int64_t>(getSplitLengths());
    const auto inputType = mlir::cast<NDTypeInterface>(getInput().getType());
    const auto axis = getInferredAxis();
    // validity of this operation was verified by VariadicSplitOp::verify()
    std::ignore = inferLastSplitLength(splitLengths, inputType.getShape().raw()[axis]);
    return splitLengths;
}

int64_t vpux::IE::VariadicSplitOp::getInferredAxis() {
    const auto inputType = mlir::cast<NDTypeInterface>(getInput().getType());
    return ::getInferredAxis(getAxis(), inputType.getRank());
}
