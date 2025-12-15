//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_storage.hpp"

#include "common/utils.hpp"

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

using VFTilingStorage = VPU::VFContainer<size_t, TileInfo>;

using MLIR_VPU_VFStorage = MLIR_UnitBase;

TEST_F(MLIR_VPU_VFStorage, VF_Container_Insert) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    VFTilingStorage storage;

    const auto axis = Shape({1, 1, 2, 1});
    const auto shape = Shape({1, 16, 2, 7});
    const auto shapeMore = Shape({1, 16, 3, 7});
    const auto offset0 = Shape({0, 0, 0, 0});
    const auto offset1 = Shape({0, 0, 2, 0});

    TileInfo tile0(shape, offset0, axis);
    TileInfo tile1(shape, offset1, axis);
    TileInfo tileMore0(shapeMore, offset0, axis);
    storage.insert(0, 0, tile0);
    storage.insert(0, 1, tile1);
    storage.insert(0, 0, tileMore0);

    // check that between two elements for same argument
    // for same VF tiles, max element was chosen
    EXPECT_TRUE(storage.get(0, 0).has_value());
    EXPECT_EQ(storage.get(0, 0).value(), tileMore0);

    // check that element for VF tile 1 was found too
    EXPECT_TRUE(storage.get(0, 1).has_value());
    EXPECT_EQ(storage.get(0, 1).value(), tile1);

    // check that there is no elements for VF tile 2
    EXPECT_FALSE(storage.get(0, 2).has_value());
}

TEST_F(MLIR_VPU_VFStorage, VF_Container_Merge) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    VFTilingStorage dst;

    const auto axis = Shape({1, 1, 2, 1});
    const auto shape = Shape({1, 16, 2, 7});
    const auto shapeMore = Shape({1, 16, 3, 7});
    const auto offset0 = Shape({0, 0, 0, 0});
    const auto offset1 = Shape({0, 0, 2, 0});

    TileInfo tile0(shape, offset0, axis);
    TileInfo tile1(shape, offset1, axis);
    TileInfo tileMore0(shapeMore, offset0, axis);

    dst.insert(0, 0, tile0);

    VFTilingStorage src;

    src.insert(0, 0, tileMore0);
    src.insert(0, 1, tile1);

    dst.merge(src);

    // check that between two elements for same argument
    // for same VF tiles, max element was chosen
    EXPECT_TRUE(dst.get(0, 0).has_value());
    EXPECT_EQ(dst.get(0, 0).value(), tileMore0);

    // check that element for VF tile 1 was found too
    EXPECT_TRUE(dst.get(0, 1).has_value());
    EXPECT_EQ(dst.get(0, 1).value(), tile1);

    // check that there is no elements for VF tile 2
    EXPECT_FALSE(dst.get(0, 2).has_value());
}
