//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pretty_test_arguments.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include <openvino/core/dimension.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <vpux/utils/core/checked_cast.hpp>

#include <set>
#include <vector>

namespace {

ov::PartialShape createPartialShape(const std::vector<BoundedDim>& boundedDims) {
    auto dimensions = std::vector<ov::Dimension>();
    dimensions.reserve(boundedDims.size());

    for (auto boundedDim : boundedDims) {
        if (boundedDim.dim == -1) {
            dimensions.emplace_back(1, boundedDim.bound);
        } else {
            dimensions.emplace_back(boundedDim.dim);
        }
    }

    return ov::PartialShape(dimensions);
}

std::vector<std::vector<int>> expandEachBoundedDimWithCustomRule(
        const std::vector<BoundedDim>& boundedDims, const DynamicShapeGenerationCallback& runtimeShapeCallback) {
    auto staticDimsVector = std::vector<std::vector<int>>();
    staticDimsVector.reserve(boundedDims.size());

    for (const auto& boundedDim : boundedDims) {
        if (boundedDim.dim != -1) {
            staticDimsVector.push_back({boundedDim.dim});
            continue;
        }

        auto rawValues = runtimeShapeCallback(boundedDim);
        VPUX_THROW_WHEN(rawValues.empty(),
                        "runtimeShapeCallback must return at least one value for dynamic dimension with bound {0}",
                        boundedDim.bound);

        auto valuesSet = std::set<int>();
        for (const auto value : rawValues) {
            VPUX_THROW_UNLESS(value > 0 && value <= boundedDim.bound,
                              "Dynamic dimension value must be in range [1, {0}], got: {1}", boundedDim.bound, value);
            valuesSet.insert(value);
        }

        auto values = std::vector<int>(valuesSet.begin(), valuesSet.end());
        staticDimsVector.push_back(values);
    }

    return staticDimsVector;
}

std::vector<ov::Shape> generateShapesFromAllDimPermutations(const std::vector<std::vector<int>>& staticDims) {
    std::vector<int> indices(staticDims.size(), 0);
    std::set<ov::Shape> allShapes;

    auto incrementIndex = [&]() {
        for (auto i : vpux::irange(indices.size()) | vpux::reversed) {
            auto maxDimIndex = static_cast<int>(staticDims[i].size());
            if (++indices[i] < maxDimIndex) {
                return true;
            } else {
                indices[i] = 0;
            }
        }

        return false;
    };

    while (true) {
        ov::Shape currentShape;
        currentShape.reserve(indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            currentShape.push_back(staticDims[i][indices[i]]);
        }

        allShapes.insert(currentShape);

        if (!incrementIndex()) {
            break;
        }
    }

    return std::vector<ov::Shape>(allShapes.begin(), allShapes.end());
}

}  // namespace

// Static shape case
ov::test::InputShape generateTestShape(const ov::Shape& shape) {
    auto partialShape = ov::PartialShape(shape);
    return ov::test::InputShape(std::move(partialShape), {shape});
}

// Dynamic shape case
ov::test::InputShape generateTestShape(const std::vector<BoundedDim>& boundedDims) {
    // Default callback for dynamic dimensions.
    // It matches documented default examples:
    // generateTestShape(5, 20)     -> ov::test::InputShape(PartialShape{5, 20}, std::vector<ov::Shape>{{5, 20}})
    // generateTestShape(5, 20_Dyn) -> ov::test::InputShape(PartialShape{5, 1..20},
    //                                                      std::vector<ov::Shape>{{5, 1}, {5, 10}, {5, 20}})
    const auto defaultRuntimeShapeCallback = [](const BoundedDim& boundedDim) {
        auto valuesSet = std::set<int>{1, (boundedDim.bound + 1) / 2, boundedDim.bound};
        return std::vector<int>(valuesSet.begin(), valuesSet.end());
    };

    return generateTestShape(boundedDims, defaultRuntimeShapeCallback);
}

ov::test::InputShape generateTestShape(const std::vector<BoundedDim>& boundedDims,
                                       const DynamicShapeGenerationCallback& runtimeShapeCallback) {
    auto partialShape = createPartialShape(boundedDims);

    auto staticDims = expandEachBoundedDimWithCustomRule(boundedDims, runtimeShapeCallback);
    auto staticShapes = generateShapesFromAllDimPermutations(staticDims);

    return ov::test::InputShape(partialShape, staticShapes);
}

std::vector<int> hostCompileSmallShapesLimitationCallback(const BoundedDim& boundedDim) {
    // HostCompile does not support shapes with dynamic dimension value below tiling step.
    // Keep generated runtime values in the upper half of the allowed [1, bound] interval.
    // Track: E#170856
    const auto lowerHalf = (boundedDim.bound + 1) / 2;
    return std::vector<int>{(lowerHalf + boundedDim.bound) / 2, boundedDim.bound};
}
