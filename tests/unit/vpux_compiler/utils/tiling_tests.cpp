//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gtest/gtest.h>
#include "vpux/compiler/core/tiling.hpp"

using namespace vpux;

// Imagine shape [1, 8, 8, 9] and divisor [1, 2, 3, 2].
// We'll end up with the following shapes and offsets.

// Shapes   {[1, 4, 3, 5], [1, 4, 3, 5], [1, 4, 3, 5], [1, 4, 3, 5], [1, 4, 2, 5], [1, 4, 2, 5],
//           [1, 4, 3, 4], [1, 4, 3, 4], [1, 4, 3, 4], [1, 4, 3, 4], [1, 4, 2, 4], [1, 4, 2, 4]}
// Offsets  {[1, 0, 0, 0], [1, 4, 0, 0], [1, 0, 3, 0], [1, 4, 3, 0], [1, 0, 6, 0], [1, 4, 6, 0],
//           [1, 0, 0, 5], [1, 4, 0, 5], [1, 0, 3, 5], [1, 4, 3, 5], [1, 0, 6, 0], [1, 4, 6, 5]}

TEST(MLIR_TilingTest, FillDividedTiles) {
    // No alignment single axis tiling
    {
        vpux::Shape shape({1, 8, 8, 17});
        vpux::Shape divisor({1, 1, 1, 4});
        const auto dividedTiles = vpux::fillDividedTiles(divisor, shape, None);

        const auto expectedTiles =
                SmallVector<TileInfo>({TileInfo{Shape({1, 8, 8, 5}), Shape({0, 0, 0, 0}), Shape({1, 1, 1, 4})},
                                       TileInfo{Shape({1, 8, 8, 4}), Shape({0, 0, 0, 5}), Shape({1, 1, 1, 4})},
                                       TileInfo{Shape({1, 8, 8, 4}), Shape({0, 0, 0, 9}), Shape({1, 1, 1, 4})},
                                       TileInfo{Shape({1, 8, 8, 4}), Shape({0, 0, 0, 13}), Shape({1, 1, 1, 4})}});

        for (auto tileInfo : zip(dividedTiles, expectedTiles)) {
            EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
        }
    }

    // No alignment multi axis tiling
    {
        vpux::Shape shape({1, 8, 8, 17});
        vpux::Shape divisor({1, 2, 3, 2});
        const auto dividedTiles = vpux::fillDividedTiles(divisor, shape, None);
        const auto expectedTiles =
                SmallVector<TileInfo>({TileInfo{Shape({1, 4, 3, 9}), Shape({0, 0, 0, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 0, 0, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 9}), Shape({0, 0, 3, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 0, 3, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 9}), Shape({0, 0, 6, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 8}), Shape({0, 0, 6, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 9}), Shape({0, 4, 0, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 4, 0, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 9}), Shape({0, 4, 3, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 4, 3, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 9}), Shape({0, 4, 6, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 8}), Shape({0, 4, 6, 9}), Shape({1, 2, 3, 2})}});

        for (auto tileInfo : zip(dividedTiles, expectedTiles)) {
            EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
        }
    }

    // Dummy alignment multi axis tiling
    {
        vpux::Shape shape({1, 8, 8, 17});
        vpux::Shape divisor({1, 2, 3, 2});
        auto alignment = SmallVector<int64_t>({1, 1, 1, 1});
        auto optionalAlignment = Optional<ArrayRef<int64_t>>(makeArrayRef(alignment));
        const auto dividedTiles = vpux::fillDividedTiles(divisor, shape, optionalAlignment);
        const auto expectedTiles =
                SmallVector<TileInfo>({TileInfo{Shape({1, 4, 3, 9}), Shape({0, 0, 0, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 0, 0, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 9}), Shape({0, 0, 3, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 0, 3, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 9}), Shape({0, 0, 6, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 8}), Shape({0, 0, 6, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 9}), Shape({0, 4, 0, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 4, 0, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 9}), Shape({0, 4, 3, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 3, 8}), Shape({0, 4, 3, 9}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 9}), Shape({0, 4, 6, 0}), Shape({1, 2, 3, 2})},
                                       TileInfo{Shape({1, 4, 2, 8}), Shape({0, 4, 6, 9}), Shape({1, 2, 3, 2})}});

        for (auto tileInfo : zip(dividedTiles, expectedTiles)) {
            EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
        }
    }

    // Single alignment, single axis tiling
    {
        vpux::Shape shape({1, 8, 8, 17});
        vpux::Shape divisor({1, 1, 1, 4});
        auto alignment = SmallVector<int64_t>({1, 1, 1, 5});
        auto optionalAlignment = Optional<ArrayRef<int64_t>>(makeArrayRef(alignment));
        const auto dividedTiles = vpux::fillDividedTiles(divisor, shape, optionalAlignment);

        const auto expectedTiles =
                SmallVector<TileInfo>({TileInfo{Shape({1, 8, 8, 5}), Shape({0, 0, 0, 0}), Shape({1, 1, 1, 4})},
                                       TileInfo{Shape({1, 8, 8, 5}), Shape({0, 0, 0, 5}), Shape({1, 1, 1, 4})},
                                       TileInfo{Shape({1, 8, 8, 5}), Shape({0, 0, 0, 10}), Shape({1, 1, 1, 4})},
                                       TileInfo{Shape({1, 8, 8, 2}), Shape({0, 0, 0, 15}), Shape({1, 1, 1, 4})}});

        for (auto tileInfo : zip(dividedTiles, expectedTiles)) {
            EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
        }
    }

    // Single alignment multi axis tiling
    {
        vpux::Shape shape({1, 8, 8, 17});
        vpux::Shape divisor({1, 2, 3, 4});
        auto alignment = SmallVector<int64_t>({1, 1, 1, 5});
        auto optionalAlignment = Optional<ArrayRef<int64_t>>(makeArrayRef(alignment));
        const auto dividedTiles = vpux::fillDividedTiles(divisor, shape, optionalAlignment);

        const auto expectedTiles =
                SmallVector<TileInfo>({TileInfo{Shape({1, 4, 3, 5}), Shape({0, 0, 0, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 0, 0, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 0, 0, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 2}), Shape({0, 0, 0, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 0, 3, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 0, 3, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 0, 3, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 2}), Shape({0, 0, 3, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 5}), Shape({0, 0, 6, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 5}), Shape({0, 0, 6, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 5}), Shape({0, 0, 6, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 2}), Shape({0, 0, 6, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 4, 0, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 4, 0, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 4, 0, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 2}), Shape({0, 4, 0, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 4, 3, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 4, 3, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 5}), Shape({0, 4, 3, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 3, 2}), Shape({0, 4, 3, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 5}), Shape({0, 4, 6, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 5}), Shape({0, 4, 6, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 5}), Shape({0, 4, 6, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 4, 2, 2}), Shape({0, 4, 6, 15}), Shape({1, 2, 3, 4})}});

        for (auto tileInfo : zip(dividedTiles, expectedTiles)) {
            EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
        }
    }

    // Multi axis alignment multi axis tiling
    {
        vpux::Shape shape({1, 8, 8, 17});
        vpux::Shape divisor({1, 2, 3, 4});
        auto alignment = SmallVector<int64_t>({1, 6, 1, 5});
        auto optionalAlignment = Optional<ArrayRef<int64_t>>(makeArrayRef(alignment));
        const auto dividedTiles = vpux::fillDividedTiles(divisor, shape, optionalAlignment);

        const auto expectedTiles =
                SmallVector<TileInfo>({TileInfo{Shape({1, 6, 3, 5}), Shape({0, 0, 0, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 3, 5}), Shape({0, 0, 0, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 3, 5}), Shape({0, 0, 0, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 3, 2}), Shape({0, 0, 0, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 3, 5}), Shape({0, 0, 3, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 3, 5}), Shape({0, 0, 3, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 3, 5}), Shape({0, 0, 3, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 3, 2}), Shape({0, 0, 3, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 2, 5}), Shape({0, 0, 6, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 2, 5}), Shape({0, 0, 6, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 2, 5}), Shape({0, 0, 6, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 6, 2, 2}), Shape({0, 0, 6, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 5}), Shape({0, 6, 0, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 5}), Shape({0, 6, 0, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 5}), Shape({0, 6, 0, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 2}), Shape({0, 6, 0, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 5}), Shape({0, 6, 3, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 5}), Shape({0, 6, 3, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 5}), Shape({0, 6, 3, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 3, 2}), Shape({0, 6, 3, 15}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 2, 5}), Shape({0, 6, 6, 0}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 2, 5}), Shape({0, 6, 6, 5}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 2, 5}), Shape({0, 6, 6, 10}), Shape({1, 2, 3, 4})},
                                       TileInfo{Shape({1, 2, 2, 2}), Shape({0, 6, 6, 15}), Shape({1, 2, 3, 4})}});

        for (auto tileInfo : zip(dividedTiles, expectedTiles)) {
            EXPECT_EQ(std::get<0>(tileInfo), std::get<1>(tileInfo));
        }
    }
}
