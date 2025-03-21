//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <vpux/compiler/core/attributes/shape.hpp>
#include <vpux/compiler/core/attributes/strides.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <gtest/gtest.h>

using namespace vpux;

struct SEInterpolateAttrParams {
    VPU::NCEInterpolateMode mode;
    IE::InterpolateNearestMode nearestMode;
    IE::InterpolateCoordMode coordMode;
    std::vector<double> scales;
    std::vector<int64_t> offsets;
    std::vector<int64_t> sizes;
    std::vector<int64_t> dataShape;
    std::vector<Bit> dataStrides;
    int64_t dataElemByteSize;
    int64_t seSize;
    std::vector<int32_t> expectedOutput;
};

class SEInterpolateAttrTests : public testing::TestWithParam<SEInterpolateAttrParams> {};

TEST_P(SEInterpolateAttrTests, ComputeSEOffsets) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto params = GetParam();

    auto modeAttr = VPU::NCEInterpolateModeAttr::get(&ctx, params.mode);
    auto nearestModeAttr = IE::InterpolateNearestModeAttr::get(&ctx, params.nearestMode);
    auto coordModeAttr = IE::InterpolateCoordModeAttr::get(&ctx, params.coordMode);
    auto scaleAttr = getFPArrayAttr(&ctx, params.scales);
    auto offsetsAttr = params.offsets.empty() ? nullptr : getIntArrayAttr(&ctx, params.offsets);
    auto sizesAttr = params.sizes.empty() ? nullptr : getIntArrayAttr(&ctx, params.sizes);
    auto interpolateAttr = VPU::SEInterpolateAttr::get(&ctx, modeAttr, coordModeAttr, scaleAttr, nearestModeAttr,
                                                       offsetsAttr, sizesAttr, /*initialInputShapeAttr=*/nullptr,
                                                       /*initialOutputShapeAttr=*/nullptr);

    auto seAttrInterface = interpolateAttr.dyn_cast<VPU::SEAttr>();
    ASSERT_TRUE(seAttrInterface != nullptr);

    Shape dataShape(params.dataShape);
    Strides dataStrides(params.dataStrides);
    Byte elemSize(params.dataElemByteSize);
    const auto seOffsets = seAttrInterface.computeSEOffsets(dataShape, dataStrides, elemSize, params.seSize);
    EXPECT_EQ(seOffsets, params.expectedOutput);
}

// clang-format off

std::vector<SEInterpolateAttrParams> nearestAsymmetricParams = {
    //
    // Nearest modes
    //
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::ROUND_PREFER_FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128, \
                         96, 96, 112, 112, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::ROUND_PREFER_CEIL, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32,  32,  32, \
                         48,  64,  64,  80,  80,  80, \
                         48,  64,  64,  80,  80,  80, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128, \
                         96, 96, 112, 112, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::CEIL, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32,  32,  32, \
                         48,  64,  64,  80,  80,  80, \
                         48,  64,  64,  80,  80,  80, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128, \
                         96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::SIMPLE, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128, \
                         96, 96, 112, 112, 128, 128}},

    //
    // Scales
    //
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 3, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  16,  16,  16,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 4, 4}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128}},

    //
    // Element byte sizes
    //
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/2, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0,  32,  32,  64,  64, \
                           0,   0,  32,  32,  64,  64, \
                          96,  96, 128, 128, 160, 160, \
                          96,  96, 128, 128, 160, 160, \
                         192, 192, 224, 224, 256, 256, \
                         192, 192, 224, 224, 256, 256}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/4, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0, 64,   64, 128, 128, \
                           0,   0, 64,   64, 128, 128, \
                         192, 192, 256, 256, 320, 320, \
                         192, 192, 256, 256, 320, 320, \
                         384, 384, 448, 448, 512, 512, \
                         384, 384, 448, 448, 512, 512}},

    //
    // Storage element sizes
    //
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 32, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{  0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80, \
                           0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 64, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, \
                           0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 96, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, \
                           0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832}},

    //
    // Offsets & sizes
    //
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 0, 0}, /*sizes=*/{1, 16, 5, 6},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32, \
                          0,  0,  16,  16,  32,  32, \
                         48, 48,  64,  64,  80,  80, \
                         48, 48,  64,  64,  80,  80, \
                         96, 96, 112, 112, 128, 128}},
    {VPU::NCEInterpolateMode::NEAREST, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 1, 1}, /*sizes=*/{1, 16, 4, 4},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32, \
                         48,  64,  64,  80, \
                         48,  64,  64,  80, \
                         96, 112, 112, 128}},
};

std::vector<SEInterpolateAttrParams> bilinearAsymmetricParams = {
    //
    // Scales
    //
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32,  32, \
                          0,  0,  16,  16,  32,  32,  32, \
                         48, 48,  64,  64,  80,  80,  80, \
                         48, 48,  64,  64,  80,  80,  80, \
                         96, 96, 112, 112, 128, 128, 128, \
                         96, 96, 112, 112, 128, 128, 128, \
                         96, 96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 3, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  16,  16,  16,  32,  32,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32,  32,  32, \
                          0,  0,  0,  16,  16,  16,  32,  32,  32,  32,  32, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80,  80,  80, \
                         48, 48, 48,  64,  64,  64,  80,  80,  80,  80,  80, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128, \
                         96, 96, 96, 112, 112, 112, 128, 128, 128, 128, 128}},
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 4, 4}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                          0,  0,  0,  0,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,  32, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         48, 48, 48, 48,  64,  64,  64,  64,  80,  80,  80,  80,  80,  80,  80, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128, \
                         96, 96, 96, 96, 112, 112, 112, 112, 128, 128, 128, 128, 128, 128, 128}},

    //
    // Element byte sizes
    //
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/2, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0,  32,  32,  64,  64, 64, \
                           0,   0,  32,  32,  64,  64, 64, \
                          96,  96, 128, 128, 160, 160, 160, \
                          96,  96, 128, 128, 160, 160, 160, \
                         192, 192, 224, 224, 256, 256, 256, \
                         192, 192, 224, 224, 256, 256, 256, \
                         192, 192, 224, 224, 256, 256, 256}},
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/4, /*seSize=*/16,
     /*expectedOutput=*/{  0,   0, 64,   64, 128, 128, 128, \
                           0,   0, 64,   64, 128, 128, 128, \
                         192, 192, 256, 256, 320, 320, 320, \
                         192, 192, 256, 256, 320, 320, 320, \
                         384, 384, 448, 448, 512, 512, 512, \
                         384, 384, 448, 448, 512, 512, 512, \
                         384, 384, 448, 448, 512, 512, 512}},

    //
    // Storage element sizes
    //
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 32, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{  0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80,  64,  80, \
                           0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80,  64,  80, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, 160, 176, \
                          96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, 160, 176, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, 256, 272, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, 256, 272, \
                         192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, 256, 272}},
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 64, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, 128, 160, \
                           0,  32,   0,  32,  64,  96,  64,  96, 128, 160, 128, 160, 128, 160, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, 320, 352, \
                         192, 224, 192, 224, 256, 288, 256, 288, 320, 352, 320, 352, 320, 352, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, 512, 544, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, 512, 544, \
                         384, 416, 384, 416, 448, 480, 448, 480, 512, 544, 512, 544, 512, 544}},
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 96, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/32,
     /*expectedOutput=*/{  0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, 192, 224, 256, \
                           0,  32,  64,   0,  32,  64,  96, 128, 160,  96, 128, 160, 192, 224, 256, 192, 224, 256, 192, 224, 256, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, 480, 512, 544, \
                         288, 320, 352, 288, 320, 352, 384, 416, 448, 384, 416, 448, 480, 512, 544, 480, 512, 544, 480, 512, 544, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, 768, 800, 832, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, 768, 800, 832, \
                         576, 608, 640, 576, 608, 640, 672, 704, 736, 672, 704, 736, 768, 800, 832, 768, 800, 832, 768, 800, 832}},

    //
    // Offsets & sizes
    //
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 0, 0}, /*sizes=*/{1, 16, 6, 7},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  0,  16,  16,  32,  32,  32, \
                          0,  0,  16,  16,  32,  32,  32, \
                         48, 48,  64,  64,  80,  80,  80, \
                         48, 48,  64,  64,  80,  80,  80, \
                         96, 96, 112, 112, 128, 128, 128, \
                         96, 96, 112, 112, 128, 128, 128}},
    {VPU::NCEInterpolateMode::BILINEAR, IE::InterpolateNearestMode::FLOOR, IE::InterpolateCoordMode::ASYMMETRIC,
     /*scales=*/{1, 1, 2, 2}, /*offsets=*/{0, 0, 1, 1}, /*sizes=*/{1, 16, 5, 5},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*expectedOutput=*/{ 0,  16,  16,  32,  32, \
                         48,  64,  64,  80,  80, \
                         48,  64,  64,  80,  80, \
                         96, 112, 112, 128, 128, \
                         96, 112, 112, 128, 128}},
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(NearestAsymmetric, SEInterpolateAttrTests, testing::ValuesIn(nearestAsymmetricParams));
INSTANTIATE_TEST_SUITE_P(BilinearAsymmetric, SEInterpolateAttrTests, testing::ValuesIn(bilinearAsymmetricParams));

//
// SEUpsamplingAttr
//

struct SEUpsamplingAttrParams {
    // SEUpsamplingAttr parameters
    SmallVector<int64_t> factors;
    SmallVector<int64_t> padding;
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    // Input data
    SmallVector<int64_t> dataShape;
    SmallVector<Bit> dataStrides;
    int64_t dataElemByteSize;
    int64_t seSize;
    SmallVector<int64_t> outputTileOffset;
    SmallVector<int64_t> outputTileShape;

    // Expected results
    SmallVector<int64_t> expectedOutputShape;
    SmallVector<int64_t> expectedBackInferredInputShape;
    SmallVector<int64_t> expectedInputTileOffset;
    SmallVector<int64_t> expectedInputTileShape;
    SmallVector<int64_t> expectedAttrOffsets;
    SmallVector<int64_t> expectedAttrSizes;
    std::vector<std::vector<int64_t>> expectedInputCoords;
    std::vector<int32_t> expectedSEOffsets;
};

class SEUpsamplingAttrTests : public testing::TestWithParam<SEUpsamplingAttrParams> {};

TEST_P(SEUpsamplingAttrTests, SEAttrInterface) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto params = GetParam();

    auto factorsAttr = getIntArrayAttr(&ctx, params.factors);
    auto paddingAttr = params.padding.empty() ? nullptr : getIntArrayAttr(&ctx, params.padding);
    auto offsetsAttr = params.offsets.empty() ? nullptr : getIntArrayAttr(&ctx, params.offsets);
    auto sizesAttr = params.sizes.empty() ? nullptr : getIntArrayAttr(&ctx, params.sizes);
    auto upsamplingAttr = VPU::SEUpsamplingAttr::get(&ctx, factorsAttr, paddingAttr, offsetsAttr, sizesAttr);

    auto seAttrInterface = upsamplingAttr.dyn_cast<VPU::SEAttr>();
    ASSERT_TRUE(seAttrInterface != nullptr);

    // inferOutputShape
    const Shape dataShape(params.dataShape);
    const auto outputShape = seAttrInterface.inferOutputShape(dataShape);
    EXPECT_EQ(outputShape.raw(), params.expectedOutputShape);

    // backInferInputShape
    const auto inputShape = seAttrInterface.backInferInputShape(outputShape);
    EXPECT_EQ(inputShape.raw(), params.expectedBackInferredInputShape);

    // backInferInputCoord
    const auto offsets = params.offsets.empty() ? SmallVector<int64_t>(outputShape.size(), 0) : params.offsets;
    const auto sizes = params.sizes.empty() ? SmallVector<int64_t>(outputShape.raw()) : params.sizes;

    const auto outputH = sizes[Dims4D::Act::H.ind()];
    const auto outputW = sizes[Dims4D::Act::W.ind()];
    ASSERT_EQ(params.expectedInputCoords.size(), outputH * outputW);
    for (auto h : irange(outputH)) {
        for (auto w : irange(outputW)) {
            const auto actualH = h + offsets[Dims4D::Act::H.ind()];
            const auto actualW = w + offsets[Dims4D::Act::W.ind()];
            const Shape outputCoord({0, 0, actualH, actualW});
            const auto inputCoord = seAttrInterface.backInferInputCoord(outputCoord, dataShape);
            const auto& expectedCoord = params.expectedInputCoords[h * outputW + w];
            const Shape expectedInputCoord({0, 0, expectedCoord[0], expectedCoord[1]});
            EXPECT_EQ(inputCoord, expectedInputCoord)
                    << "Invalid input coordinates for output coordinate [" << actualH << ", " << actualW << "]";
        }
    }

    // extractTile
    if (!params.outputTileOffset.empty() && !params.outputTileShape.empty()) {
        const Shape outputTileOffset(params.outputTileOffset);
        const Shape outputTileShape(params.outputTileShape);
        Shape inputTileOffset{};
        Shape inputTileShape{};
        auto newSEAttr = seAttrInterface.extractTile(outputTileOffset, outputTileShape, dataShape, inputTileOffset,
                                                     inputTileShape);
        auto newSEUpsamplingAttr = newSEAttr.dyn_cast_or_null<VPU::SEUpsamplingAttr>();
        ASSERT_TRUE(newSEUpsamplingAttr != nullptr);
        EXPECT_EQ(inputTileOffset.raw(), params.expectedInputTileOffset);
        EXPECT_EQ(inputTileShape.raw(), params.expectedInputTileShape);
        const auto newSEUpsamplingAttrOffsets = parseIntArrayAttr<int64_t>(newSEUpsamplingAttr.getOffsets());
        const auto newSEUpsamplingAttrSizes = parseIntArrayAttr<int64_t>(newSEUpsamplingAttr.getSizes());
        const auto newSEUpsamplingAttrPadding = parseIntArrayAttr<int64_t>(newSEUpsamplingAttr.getPadding());
        EXPECT_EQ(newSEUpsamplingAttrOffsets, params.expectedAttrOffsets);
        EXPECT_EQ(newSEUpsamplingAttrSizes, params.expectedAttrSizes);
    }

    // computeSEOffsets
    const Strides dataStrides(params.dataStrides);
    const Byte elemSize(params.dataElemByteSize);
    const auto seOffsets = seAttrInterface.computeSEOffsets(dataShape, dataStrides, elemSize, params.seSize);
    EXPECT_EQ(seOffsets, params.expectedSEOffsets);
}

// clang-format off

std::vector<SEUpsamplingAttrParams> upsamplingParams = {
    {/*factors=*/{1, 1}, /*padding=*/{0, 0, 0, 0}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 1}, /*outputTileShape=*/{1, 16, 3, 3},
     /*expectedOutputShape*/{1, 16, 5, 5}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 2, 2},
     /*expectedAttrOffsets=*/{0, 0, 1, 1}, /*expectedAttrSizes=*/{1, 16, 3, 3},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, \
                             {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,  0,  16,  16,  32, \
                             0,  0,  16,  16,  32, \
                            48, 48,  64,  64,  80, \
                            48, 48,  64,  64,  80, \
                            96, 96, 112, 112, 128}},
    {/*factors=*/{1, 1}, /*padding=*/{1, 1, 1, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 0, 3}, /*outputTileShape=*/{1, 16, 2, 3},
     /*expectedOutputShape*/{1, 16, 7, 7},
     /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 1}, /*expectedInputTileShape=*/{1, 16, 1, 2},
     /*expectedAttrOffsets=*/{0, 0, 0, 1}, /*expectedAttrSizes=*/{1, 16, 2, 3},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,  0,   0,  16,  16,  32,  32, \
                             0,  0,   0,  16,  16,  32,  32, \
                             0,  0,   0,  16,  16,  32,  32, \
                            48, 48,  48,  64,  64,  80,  80, \
                            48, 48,  48,  64,  64,  80,  80, \
                            96, 96,  96, 112, 112, 128, 128, \
                            96, 96,  96, 112, 112, 128, 128}},
    {/*factors=*/{1, 1}, /*padding=*/{0, 2, 2, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 5, 3}, /*outputTileShape=*/{1, 16, 4, 5},
     /*expectedOutputShape*/{1, 16, 10, 9},
     /*expectedBackInferredInputShape=*/{1, 16, 4, 4},
     /*expectedInputTileOffset=*/{0, 0, 1, 1}, /*expectedInputTileShape=*/{1, 16, 3, 3},
     /*expectedAttrOffsets=*/{0, 0, 3, 1}, /*expectedAttrSizes=*/{1, 16, 4, 5},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 3}, {0, 3}, {0, 3}, \
                             {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 3}, {0, 3}, {0, 3}, \
                             {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 3}, {0, 3}, {0, 3}, \
                             {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 3}, {0, 3}, {0, 3}, \
                             {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, {1, 3}, {1, 3}, {1, 3}, \
                             {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, {1, 3}, {1, 3}, {1, 3}, \
                             {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}, {2, 3}, {2, 3}, {2, 3}, \
                             {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}, {2, 3}, {2, 3}, {2, 3}, \
                             {3, 0}, {3, 0}, {3, 1}, {3, 1}, {3, 2}, {3, 2}, {3, 3}, {3, 3}, {3, 3}, \
                             {3, 0}, {3, 0}, {3, 1}, {3, 1}, {3, 2}, {3, 2}, {3, 3}, {3, 3}, {3, 3}},
     /*expectedSEOffsets=*/{ 0,   0,  16,  16,  32,  32,  48,  48,  48, \
                             0,   0,  16,  16,  32,  32,  48,  48,  48, \
                             0,   0,  16,  16,  32,  32,  48,  48,  48, \
                             0,   0,  16,  16,  32,  32,  48,  48,  48, \
                            64,  64,  80,  80,  96,  96, 112, 112, 112, \
                            64,  64,  80,  80,  96,  96, 112, 112, 112, \
                           128, 128, 144, 144, 160, 160, 176, 176, 176, \
                           128, 128, 144, 144, 160, 160, 176, 176, 176, \
                           192, 192, 208, 208, 224, 224, 240, 240, 240, \
                           192, 192, 208, 208, 224, 224, 240, 240, 240}},
    {/*factors=*/{2, 2}, /*padding=*/{1, 1, 1, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 0}, /*outputTileShape=*/{1, 16, 8, 9},
     /*expectedOutputShape*/{1, 16, 9, 9},
     /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 3, 3},
     /*expectedAttrOffsets=*/{0, 0, 1, 0}, /*expectedAttrSizes=*/{1, 16, 8, 9},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 1}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 1}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,   0,   0,   0,  16,  16,  16,  32,  32, \
                             0,   0,   0,   0,  16,  16,  16,  32,  32, \
                             0,   0,   0,   0,  16,  16,  16,  32,  32, \
                             0,   0,   0,   0,  16,  16,  16,  32,  32, \
                            48,  48,  48,  48,  64,  64,  64,  80,  80, \
                            48,  48,  48,  48,  64,  64,  64,  80,  80, \
                            48,  48,  48,  48,  64,  64,  64,  80,  80, \
                            96,  96,  96,  96, 112, 112, 112, 128, 128, \
                            96,  96,  96,  96, 112, 112, 112, 128, 128}},

    //
    // Element byte sizes
    //
    {/*factors=*/{1, 1}, /*padding=*/{1, 1, 1, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/2, /*seSize=*/16,
     /*outputTileOffset=*/{}, /*outputTileShape=*/{},
     /*expectedOutputShape*/{1, 16, 7, 7},
     /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{}, /*expectedInputTileShape=*/{},
     /*expectedAttrOffsets=*/{}, /*expectedAttrSizes=*/{},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{  0,   0,    0,  32,  32,  64,  64, \
                              0,   0,    0,  32,  32,  64,  64, \
                              0,   0,    0,  32,  32,  64,  64, \
                             96,  96,   96, 128, 128, 160, 160, \
                             96,  96,   96, 128, 128, 160, 160, \
                            192, 192,  192, 224, 224, 256, 256, \
                            192, 192,  192, 224, 224, 256, 256}},

    //
    // Storage element sizes
    //
    {/*factors=*/{1, 1}, /*padding=*/{1, 1, 1, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 32, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{}, /*outputTileShape=*/{},
     /*expectedOutputShape*/{1, 32, 7, 7},
     /*expectedBackInferredInputShape=*/{1, 32, 3, 3},
     /*expectedInputTileOffset=*/{}, /*expectedInputTileShape=*/{},
     /*expectedAttrOffsets=*/{}, /*expectedAttrSizes=*/{},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,  16,   0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80, \
                             0,  16,   0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80, \
                             0,  16,   0,  16,   0,  16,  32,  48,  32,  48,  64,  80,  64,  80, \
                            96, 112,  96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, \
                            96, 112,  96, 112,  96, 112, 128, 144, 128, 144, 160, 176, 160, 176, \
                           192, 208, 192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272, \
                           192, 208, 192, 208, 192, 208, 224, 240, 224, 240, 256, 272, 256, 272}},

    //
    // Offsets & sizes
    //
    {/*factors=*/{1, 1}, /*padding=*/{1, 1, 1, 1}, /*offsets=*/{0, 0, 1, 2}, /*sizes=*/{1, 16, 5, 5},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{}, /*outputTileShape=*/{},
     /*expectedOutputShape*/{1, 16, 5, 5},
     /*expectedBackInferredInputShape=*/{1, 16, 2, 2},
     /*expectedInputTileOffset=*/{}, /*expectedInputTileShape=*/{},
     /*expectedAttrOffsets=*/{}, /*expectedAttrSizes=*/{},
     /*expectedInputCoords*/{{0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {1, 0}, {1, 1}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 1}, {2, 1}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,  16,  16,  32,  32, \
                             0,  16,  16,  32,  32, \
                            48,  64,  64,  80,  80, \
                            48,  64,  64,  80,  80, \
                            96, 112, 112, 128, 128}},
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(unit, SEUpsamplingAttrTests, testing::ValuesIn(upsamplingParams));

//
// SEPaddingAttr
//

struct SEPaddingAttrParams {
    // SEPaddingAttr parameters
    IE::PadMode padMode;
    SmallVector<int64_t> padding;
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    // Input data
    SmallVector<int64_t> dataShape;
    SmallVector<Bit> dataStrides;
    int64_t dataElemByteSize;
    int64_t seSize;
    SmallVector<int64_t> outputTileOffset;
    SmallVector<int64_t> outputTileShape;

    // Expected results
    SmallVector<int64_t> expectedOutputShape;
    SmallVector<int64_t> expectedBackInferredInputShape;
    SmallVector<int64_t> expectedInputTileOffset;
    SmallVector<int64_t> expectedInputTileShape;
    SmallVector<int64_t> expectedAttrOffsets;
    SmallVector<int64_t> expectedAttrSizes;
    SmallVector<int64_t> expectedAttrPadding;
    std::vector<std::vector<int64_t>> expectedInputCoords;
    std::vector<int32_t> expectedSEOffsets;
};

class SEPaddingAttrTests : public testing::TestWithParam<SEPaddingAttrParams> {};

TEST_P(SEPaddingAttrTests, SEAttrInterface) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto params = GetParam();

    auto padModeAttr = IE::PadModeAttr::get(&ctx, params.padMode);
    auto paddingAttr = getIntArrayAttr(&ctx, params.padding);
    auto offsetsAttr = params.offsets.empty() ? nullptr : getIntArrayAttr(&ctx, params.offsets);
    auto sizesAttr = params.sizes.empty() ? nullptr : getIntArrayAttr(&ctx, params.sizes);
    auto PaddingAttr = VPU::SEPaddingAttr::get(&ctx, padModeAttr, paddingAttr, offsetsAttr, sizesAttr);

    auto seAttrInterface = PaddingAttr.dyn_cast<VPU::SEAttr>();
    ASSERT_TRUE(seAttrInterface != nullptr);

    // inferOutputShape
    const Shape dataShape(params.dataShape);
    const auto outputShape = seAttrInterface.inferOutputShape(dataShape);
    EXPECT_EQ(outputShape.raw(), params.expectedOutputShape);

    // backInferInputShape
    if (offsetsAttr == nullptr && sizesAttr == nullptr) {
        const auto inputShape = seAttrInterface.backInferInputShape(outputShape);
        EXPECT_EQ(inputShape.raw(), params.expectedBackInferredInputShape);
    }

    // backInferInputCoord
    const auto offsets = params.offsets.empty() ? SmallVector<int64_t>(outputShape.size(), 0) : params.offsets;
    const auto sizes = params.sizes.empty() ? SmallVector<int64_t>(outputShape.raw()) : params.sizes;

    const auto outputH = sizes[Dims4D::Act::H.ind()];
    const auto outputW = sizes[Dims4D::Act::W.ind()];
    ASSERT_EQ(params.expectedInputCoords.size(), outputH * outputW);
    for (auto h : irange(outputH)) {
        for (auto w : irange(outputW)) {
            const auto actualH = h + offsets[Dims4D::Act::H.ind()];
            const auto actualW = w + offsets[Dims4D::Act::W.ind()];
            const Shape outputCoord({0, 0, actualH, actualW});
            const auto inputCoord = seAttrInterface.backInferInputCoord(outputCoord, dataShape);
            const auto& expectedCoord = params.expectedInputCoords[h * outputW + w];
            const Shape expectedInputCoord({0, 0, expectedCoord[0], expectedCoord[1]});
            EXPECT_EQ(inputCoord, expectedInputCoord)
                    << "Invalid input coordinates for output coordinate [" << actualH << ", " << actualW << "]";
        }
    }

    // extractTile
    if (!params.outputTileOffset.empty() && !params.outputTileShape.empty()) {
        const Shape outputTileOffset(params.outputTileOffset);
        const Shape outputTileShape(params.outputTileShape);
        Shape inputTileOffset{};
        Shape inputTileShape{};
        auto newSEAttr = seAttrInterface.extractTile(outputTileOffset, outputTileShape, dataShape, inputTileOffset,
                                                     inputTileShape);
        auto newSEPaddingAttr = newSEAttr.dyn_cast_or_null<VPU::SEPaddingAttr>();
        ASSERT_TRUE(newSEPaddingAttr != nullptr);
        EXPECT_EQ(inputTileOffset.raw(), params.expectedInputTileOffset);
        EXPECT_EQ(inputTileShape.raw(), params.expectedInputTileShape);
        const auto newSEPaddingAttrOffsets = parseIntArrayAttr<int64_t>(newSEPaddingAttr.getOffsets());
        const auto newSEPaddingAttrSizes = parseIntArrayAttr<int64_t>(newSEPaddingAttr.getSizes());
        const auto newSEPaddingAttrPadding = parseIntArrayAttr<int64_t>(newSEPaddingAttr.getPadding());
        EXPECT_EQ(newSEPaddingAttrOffsets, params.expectedAttrOffsets);
        EXPECT_EQ(newSEPaddingAttrSizes, params.expectedAttrSizes);
        EXPECT_EQ(newSEPaddingAttrPadding, params.expectedAttrPadding);
    }

    // computeSEOffsets
    const Strides dataStrides(params.dataStrides);
    const Byte elemSize(params.dataElemByteSize);
    const auto seOffsets = seAttrInterface.computeSEOffsets(dataShape, dataStrides, elemSize, params.seSize);
    EXPECT_EQ(seOffsets, params.expectedSEOffsets);
}

// clang-format off

std::vector<SEPaddingAttrParams> paddingParams = {
    // REFLECT: H(CoordLocation::padBegin -> CoordLocation::inData); W(CoordLocation::inData -> CoordLocation::inData)
    {/*padMode=*/IE::PadMode::REFLECT, /*padding=*/{1, 2, 2, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 1}, /*outputTileShape=*/{1, 16, 3, 3},
     /*expectedOutputShape*/{1, 16, 6, 6}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 2, 3},
     /*expectedAttrOffsets=*/{0, 0, 1, 1}, /*expectedAttrSizes=*/{1, 16, 3, 3}, /*expectedAttrPadding=*/{1, 2, 2, 1},
     /*expectedInputCoords*/{{2, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 1}, {2, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, \
                             {0, 1}, {0, 0}, {0, 1}, {0, 2}, {0, 1}, {0, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, \
                             {2, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 1}, {2, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}},
     /*expectedSEOffsets=*/{ 112, 96, 112, 128, 112, 96, \
                              64, 48,  64,  80,  64, 48, \
                              16,  0,  16,  32,  16,  0, \
                              64, 48,  64,  80,  64, 48, \
                             112, 96, 112, 128, 112, 96, \
                              64, 48,  64,  80,  64, 48}},

    // REFLECT: H(CoordLocation::padBegin -> CoordLocation::padBegin); W(CoordLocation::padEnd -> CoordLocation::padEnd)
    {/*padMode=*/IE::PadMode::REFLECT, /*padding=*/{1, 2, 2, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 4}, /*outputTileShape=*/{1, 16, 1, 2},
     /*expectedOutputShape*/{1, 16, 6, 6}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 2, 3},
     /*expectedAttrOffsets=*/{0, 0, 3, 4}, /*expectedAttrSizes=*/{1, 16, 1, 2}, /*expectedAttrPadding=*/{1, 2, 2, 1},
     /*expectedInputCoords*/{{2, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 1}, {2, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, \
                             {0, 1}, {0, 0}, {0, 1}, {0, 2}, {0, 1}, {0, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, \
                             {2, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 1}, {2, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}},
     /*expectedSEOffsets=*/{ 112, 96, 112, 128, 112, 96, \
                              64, 48,  64,  80,  64, 48, \
                              16,  0,  16,  32,  16,  0, \
                              64, 48,  64,  80,  64, 48, \
                             112, 96, 112, 128, 112, 96, \
                              64, 48,  64,  80,  64, 48}},

    // REFLECT: H(CoordLocation::padBegin -> CoordLocation::padEnd); W(CoordLocation::inData -> CoordLocation::padEnd)
    {/*padMode=*/IE::PadMode::REFLECT, /*padding=*/{1, 2, 2, 1}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 2}, /*outputTileShape=*/{1, 16, 5, 3},
     /*expectedOutputShape*/{1, 16, 6, 6}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 1}, /*expectedInputTileShape=*/{1, 16, 3, 2},
     /*expectedAttrOffsets=*/{0, 0, 1, 1}, /*expectedAttrSizes=*/{1, 16, 5, 3}, /*expectedAttrPadding=*/{1, 2, 2, 1},
     /*expectedInputCoords*/{{2, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 1}, {2, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, \
                             {0, 1}, {0, 0}, {0, 1}, {0, 2}, {0, 1}, {0, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, \
                             {2, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 1}, {2, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}},
     /*expectedSEOffsets=*/{ 112, 96, 112, 128, 112, 96, \
                              64, 48,  64,  80,  64, 48, \
                              16,  0,  16,  32,  16,  0, \
                              64, 48,  64,  80,  64, 48, \
                             112, 96, 112, 128, 112, 96, \
                              64, 48,  64,  80,  64, 48}},

    // REFLECT with offsetsAttr and sizesAttr
    {/*padMode=*/IE::PadMode::REFLECT, /*padding=*/{1, 2, 2, 1}, /*offsets=*/{0, 0, 2, 0}, /*sizes=*/{1, 16, 3, 6},
     /*dataShape=*/{1, 16, 2, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{}, /*outputTileShape=*/{},
     /*expectedOutputShape*/{1, 16, 3, 6}, /*expectedBackInferredInputShape=*/{},
     /*expectedInputTileOffset=*/{}, /*expectedInputTileShape=*/{},
     /*expectedAttrOffsets=*/{}, /*expectedAttrSizes=*/{}, /*expectedAttrPadding=*/{},
     /*expectedInputCoords*/{{0, 1}, {0, 0}, {0, 1}, {0, 2}, {0, 1}, {0, 0}, \
                             {1, 1}, {1, 0}, {1, 1}, {1, 2}, {1, 1}, {1, 0}, \
                             {0, 1}, {0, 0}, {0, 1}, {0, 2}, {0, 1}, {0, 0}},
     /*expectedSEOffsets=*/{16,  0, 16, 32, 16,  0, \
                            64, 48, 64, 80, 64, 48, \
                            16,  0, 16, 32, 16,  0}},

    // SYMMETRIC: H(CoordLocation::inData -> CoordLocation::padEnd); W(CoordLocation::padBegin -> CoordLocation::inData)
    {/*padMode=*/IE::PadMode::SYMMETRIC, /*padding=*/{2, 1, 1, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 2, 0}, /*outputTileShape=*/{1, 16, 3, 4},
     /*expectedOutputShape*/{1, 16, 6, 6}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 1, 0}, /*expectedInputTileShape=*/{1, 16, 2, 2},
     /*expectedAttrOffsets=*/{0, 0, 1, 0}, /*expectedAttrSizes=*/{1, 16, 3, 4}, /*expectedAttrPadding=*/{2, 1, 1, 2},
     /*expectedInputCoords*/{{0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 1}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, \
                             {2, 1}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}},
     /*expectedSEOffsets=*/{ 16,  0,  0,  16,  32,  32, \
                             16,  0,  0,  16,  32,  32, \
                             64, 48, 48,  64,  80,  80, \
                            112, 96, 96, 112, 128, 128, \
                            112, 96, 96, 112, 128, 128, \
                             64, 48, 48,  64,  80,  80}},

    // SYMMETRIC: H(CoordLocation::padEnd -> CoordLocation::padEnd); W(CoordLocation::padBegin -> CoordLocation::inData)
    {/*padMode=*/IE::PadMode::SYMMETRIC, /*padding=*/{2, 1, 1, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 4, 1}, /*outputTileShape=*/{1, 16, 2, 3},
     /*expectedOutputShape*/{1, 16, 6, 6}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 1, 0}, /*expectedInputTileShape=*/{1, 16, 2, 2},
     /*expectedAttrOffsets=*/{0, 0, 3, 1}, /*expectedAttrSizes=*/{1, 16, 2, 3}, /*expectedAttrPadding=*/{2, 1, 1, 2},
     /*expectedInputCoords*/{{0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 1}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, \
                             {2, 1}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}},
     /*expectedSEOffsets=*/{ 16,  0,  0,  16,  32,  32, \
                             16,  0,  0,  16,  32,  32, \
                             64, 48, 48,  64,  80,  80, \
                            112, 96, 96, 112, 128, 128, \
                            112, 96, 96, 112, 128, 128, \
                             64, 48, 48,  64,  80,  80}},

    // SYMMETRIC: H(CoordLocation::padBegin -> CoordLocation::padEnd); W(CoordLocation::inData -> CoordLocation::inData)
    {/*padMode=*/IE::PadMode::SYMMETRIC, /*padding=*/{2, 1, 1, 2}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 0, 4}, /*outputTileShape=*/{1, 16, 5, 1},
     /*expectedOutputShape*/{1, 16, 6, 6}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 2}, /*expectedInputTileShape=*/{1, 16, 3, 1},
     /*expectedAttrOffsets=*/{0, 0, 0, 2}, /*expectedAttrSizes=*/{1, 16, 5, 1}, /*expectedAttrPadding=*/{2, 1, 1, 2},
     /*expectedInputCoords*/{{0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, \
                             {0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, \
                             {2, 1}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, \
                             {2, 1}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}},
     /*expectedSEOffsets=*/{ 16,  0,  0,  16,  32,  32, \
                             16,  0,  0,  16,  32,  32, \
                             64, 48, 48,  64,  80,  80, \
                            112, 96, 96, 112, 128, 128, \
                            112, 96, 96, 112, 128, 128, \
                             64, 48, 48,  64,  80,  80}},

    // SYMMETRIC with offsetsAttr and sizesAttr
    {/*padMode=*/IE::PadMode::SYMMETRIC, /*padding=*/{2, 1, 1, 2}, /*offsets=*/{0, 0, 1, 0}, /*sizes=*/{1, 16, 4, 6},
     /*dataShape=*/{1, 16, 2, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{}, /*outputTileShape=*/{},
     /*expectedOutputShape*/{1, 16, 4, 6}, /*expectedBackInferredInputShape=*/{},
     /*expectedInputTileOffset=*/{}, /*expectedInputTileShape=*/{},
     /*expectedAttrOffsets=*/{}, /*expectedAttrSizes=*/{}, /*expectedAttrPadding=*/{},
     /*expectedInputCoords*/{{0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, \
                             {1, 1}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, \
                             {0, 1}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}},
     /*expectedSEOffsets=*/{16,  0,  0, 16, 32, 32, \
                            64, 48, 48, 64, 80, 80, \
                            64, 48, 48, 64, 80, 80, \
                            16,  0,  0, 16, 32, 32,}},

    // EDGE: H(CoordLocation::padBegin -> CoordLocation::padEnd); W(CoordLocation::padBegin -> CoordLocation::inData)
    {/*padMode=*/IE::PadMode::EDGE, /*padding=*/{2, 1, 3, 4}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 0, 0}, /*outputTileShape=*/{1, 16, 8, 4},
     /*expectedOutputShape*/{1, 16, 8, 8}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 3, 2},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 16, 8, 4}, /*expectedAttrPadding=*/{2, 1, 3, 4},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,  0,  0,  16,  32,  32,  32,  32, \
                             0,  0,  0,  16,  32,  32,  32,  32, \
                            48, 48, 48,  64,  80,  80,  80,  80, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128}},

    // EDGE: H(CoordLocation::padBegin -> CoordLocation::padEnd); W(CoordLocation::padBegin -> CoordLocation::padEnd)
    {/*padMode=*/IE::PadMode::EDGE, /*padding=*/{2, 1, 3, 4}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 2, 3}, /*outputTileShape=*/{1, 16, 4, 4},
     /*expectedOutputShape*/{1, 16, 8, 8}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 1, 1}, /*expectedInputTileShape=*/{1, 16, 2, 2},
     /*expectedAttrOffsets=*/{0, 0, 1, 2}, /*expectedAttrSizes=*/{1, 16, 4, 4}, /*expectedAttrPadding=*/{2, 1, 3, 4},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,  0,  0,  16,  32,  32,  32,  32, \
                             0,  0,  0,  16,  32,  32,  32,  32, \
                            48, 48, 48,  64,  80,  80,  80,  80, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128}},

    // EDGE: H(CoordLocation::padEnd -> CoordLocation::padEnd); W(CoordLocation::padEnd -> CoordLocation::padEnd)
    {/*padMode=*/IE::PadMode::EDGE, /*padding=*/{2, 1, 3, 4}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 3, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 5, 5}, /*outputTileShape=*/{1, 16, 3, 3},
     /*expectedOutputShape*/{1, 16, 8, 8}, /*expectedBackInferredInputShape=*/{1, 16, 3, 3},
     /*expectedInputTileOffset=*/{0, 0, 2, 2}, /*expectedInputTileShape=*/{1, 16, 1, 1},
     /*expectedAttrOffsets=*/{0, 0, 2, 3}, /*expectedAttrSizes=*/{1, 16, 3, 3}, /*expectedAttrPadding=*/{2, 1, 3, 4},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {1, 0}, {1, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 2}, {1, 2}, {1, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, \
                             {2, 0}, {2, 0}, {2, 0}, {2, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}},
     /*expectedSEOffsets=*/{ 0,  0,  0,  16,  32,  32,  32,  32, \
                             0,  0,  0,  16,  32,  32,  32,  32, \
                            48, 48, 48,  64,  80,  80,  80,  80, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128, \
                            96, 96, 96, 112, 128, 128, 128, 128}},

    // EDGE with offsetsAttr and sizesAttr
    {/*padMode=*/IE::PadMode::EDGE, /*padding=*/{2, 1, 3, 4}, /*offsets=*/{0, 0, 1, 0}, /*sizes=*/{1, 16, 3, 8},
     /*dataShape=*/{1, 16, 1, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{}, /*outputTileShape=*/{},
     /*expectedOutputShape*/{1, 16, 3, 8}, /*expectedBackInferredInputShape=*/{},
     /*expectedInputTileOffset=*/{}, /*expectedInputTileShape=*/{},
     /*expectedAttrOffsets=*/{}, /*expectedAttrSizes=*/{}, /*expectedAttrPadding=*/{},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}},
     /*expectedSEOffsets=*/{ 0,  0,  0,  16,  32,  32,  32,  32, \
                             0,  0,  0,  16,  32,  32,  32,  32, \
                             0,  0,  0,  16,  32,  32,  32,  32}},

    // CONSTANT with offsetsAttr and sizesAttr
    {/*padMode=*/IE::PadMode::CONSTANT, /*padding=*/{2, 1, 3, 4}, /*offsets=*/{0, 0, 1, 0}, /*sizes=*/{1, 16, 3, 8},
     /*dataShape=*/{1, 16, 1, 3}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{}, /*outputTileShape=*/{},
     /*expectedOutputShape*/{1, 16, 3, 8}, /*expectedBackInferredInputShape=*/{},
     /*expectedInputTileOffset=*/{}, /*expectedInputTileShape=*/{},
     /*expectedAttrOffsets=*/{}, /*expectedAttrSizes=*/{}, /*expectedAttrPadding=*/{},
     /*expectedInputCoords*/{{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}, \
                             {0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 2}, {0, 2}, {0, 2}},
     /*expectedSEOffsets=*/{ 0,  0,  0,  16,  32,  32,  32,  32, \
                             0,  0,  0,  16,  32,  32,  32,  32, \
                             0,  0,  0,  16,  32,  32,  32,  32}},
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(unit, SEPaddingAttrTests, testing::ValuesIn(paddingParams));

//
// SERollAttr
//

struct SERollAttrParams {
    // SERollAttr parameters
    SmallVector<int64_t> shifts;
    SmallVector<int64_t> axes;
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    // Input data
    SmallVector<int64_t> dataShape;
    SmallVector<Bit> dataStrides;
    int64_t dataElemByteSize;
    int64_t seSize;
    SmallVector<int64_t> outputTileOffset;
    SmallVector<int64_t> outputTileShape;

    // Expected results
    SmallVector<int64_t> expectedOutputShape;
    SmallVector<int64_t> expectedBackInferredInputShape;
    SmallVector<int64_t> expectedInputTileOffset;
    SmallVector<int64_t> expectedInputTileShape;
    SmallVector<int64_t> expectedAttrShifts;
    SmallVector<int64_t> expectedAttrAxes;
    SmallVector<int64_t> expectedAttrOffsets;
    SmallVector<int64_t> expectedAttrSizes;
    std::vector<std::vector<int64_t>> expectedInputCoords;
    std::vector<int32_t> expectedSEOffsets;
};

class SERollAttrTests : public testing::TestWithParam<SERollAttrParams> {};

TEST_P(SERollAttrTests, SEAttrInterface) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto params = GetParam();

    auto shiftsAttr = getIntArrayAttr(&ctx, params.shifts);
    auto axesAttr = getIntArrayAttr(&ctx, params.axes);
    auto offsetsAttr = params.offsets.empty() ? nullptr : getIntArrayAttr(&ctx, params.offsets);
    auto sizesAttr = params.sizes.empty() ? nullptr : getIntArrayAttr(&ctx, params.sizes);
    auto rollAttr = VPU::SERollAttr::get(&ctx, shiftsAttr, axesAttr, offsetsAttr, sizesAttr);

    auto seAttrInterface = rollAttr.dyn_cast<VPU::SEAttr>();
    ASSERT_TRUE(seAttrInterface != nullptr);

    // inferOutputShape
    const Shape dataShape(params.dataShape);
    const auto outputShape = seAttrInterface.inferOutputShape(dataShape);
    EXPECT_EQ(outputShape.raw(), params.expectedOutputShape);

    // backInferInputShape
    if (offsetsAttr == nullptr && sizesAttr == nullptr) {
        const auto inputShape = seAttrInterface.backInferInputShape(outputShape);
        EXPECT_EQ(inputShape.raw(), params.expectedBackInferredInputShape);
    }

    // backInferInputCoord
    const auto offsets = params.offsets.empty() ? SmallVector<int64_t>(outputShape.size(), 0) : params.offsets;
    const auto sizes = params.sizes.empty() ? SmallVector<int64_t>(outputShape.raw()) : params.sizes;

    const auto outputH = sizes[Dims4D::Act::H.ind()];
    const auto outputW = sizes[Dims4D::Act::W.ind()];
    ASSERT_EQ(params.expectedInputCoords.size(), outputH * outputW);

    for (auto h : irange(outputH)) {
        for (auto w : irange(outputW)) {
            const auto actualH = h + offsets[Dims4D::Act::H.ind()];
            const auto actualW = w + offsets[Dims4D::Act::W.ind()];
            const Shape outputCoord({0, 0, actualH, actualW});
            const auto inputCoord = seAttrInterface.backInferInputCoord(outputCoord, dataShape);
            const auto& expectedCoord = params.expectedInputCoords[h * outputW + w];
            const Shape expectedInputCoord({0, 0, expectedCoord[0], expectedCoord[1]});
            EXPECT_EQ(inputCoord, expectedInputCoord)
                    << "Invalid input coordinates for output coordinate [" << actualH << ", " << actualW << "]";
        }
    }

    // extractTile
    if (!params.outputTileOffset.empty() && !params.outputTileShape.empty()) {
        const Shape outputTileOffset(params.outputTileOffset);
        const Shape outputTileShape(params.outputTileShape);
        Shape inputTileOffset{};
        Shape inputTileShape{};
        auto newSEAttr = seAttrInterface.extractTile(outputTileOffset, outputTileShape, dataShape, inputTileOffset,
                                                     inputTileShape);
        auto newSERollAttr = newSEAttr.dyn_cast_or_null<VPU::SERollAttr>();
        ASSERT_TRUE(newSERollAttr != nullptr);
        EXPECT_EQ(inputTileOffset.raw(), params.expectedInputTileOffset);
        EXPECT_EQ(inputTileShape.raw(), params.expectedInputTileShape);
        const auto newAttrShifts = parseIntArrayAttr<int64_t>(newSERollAttr.getShift());
        const auto newAttrAxes = parseIntArrayAttr<int64_t>(newSERollAttr.getAxes());
        EXPECT_EQ(newAttrShifts, params.expectedAttrShifts);
        EXPECT_EQ(newAttrAxes, params.expectedAttrAxes);

        const auto newSERollAttrOffsets = parseIntArrayAttr<int64_t>(newSERollAttr.getOffsets());
        const auto newSERollAttrSizes = parseIntArrayAttr<int64_t>(newSERollAttr.getSizes());
        EXPECT_EQ(newSERollAttrOffsets, params.expectedAttrOffsets);
        EXPECT_EQ(newSERollAttrSizes, params.expectedAttrSizes);
    }

    // computeSEOffsets
    const Strides dataStrides(params.dataStrides);
    const Byte elemSize(params.dataElemByteSize);
    const auto seOffsets = seAttrInterface.computeSEOffsets(dataShape, dataStrides, elemSize, params.seSize);
    EXPECT_EQ(seOffsets, params.expectedSEOffsets);
}

// clang-format off

std::vector<SERollAttrParams> rollParams = {
    {/*shifts=*/{2, 2}, /*axes=*/{2, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 1}, /*outputTileShape=*/{1, 16, 3, 3},
     /*expectedOutputShape*/{1, 16, 4, 4}, /*expectedBackInferredInputShape=*/{1, 16, 4, 4},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 4, 4},
     /*expectedAttrShifts=*/{1, 1}, /*expectedAttrAxes=*/{2, 3},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 16, 3, 3},
     /*expectedInputCoords*/{{2, 2}, {2, 3}, {2, 0}, {2, 1}, \
                             {3, 2}, {3, 3}, {3, 0}, {3, 1}, \
                             {0, 2}, {0, 3}, {0, 0}, {0, 1}, \
                             {1, 2}, {1, 3}, {1, 0}, {1, 1}},
     /*expectedSEOffsets=*/{ 160, 176, 128, 144, \
                             224, 240, 192, 208, \
                              32,  48,   0,  16,  \
                              96, 112,  64,  80}},

    {/*shifts=*/{3, 3}, /*axes=*/{2, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 0, 0}, /*outputTileShape=*/{1, 16, 2, 2},
     /*expectedOutputShape*/{1, 16, 4, 4}, /*expectedBackInferredInputShape=*/{1, 16, 4, 4},
     /*expectedInputTileOffset=*/{0, 0, 1, 1}, /*expectedInputTileShape=*/{1, 16, 2, 2},
     /*expectedAttrShifts=*/{3, 3}, /*expectedAttrAxes=*/{2, 3},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 16, 2, 2},
     /*expectedInputCoords*/{{1, 1}, {1, 2}, {1, 3}, {1, 0}, \
                             {2, 1}, {2, 2}, {2, 3}, {2, 0}, \
                             {3, 1}, {3, 2}, {3, 3}, {3, 0}, \
                             {0, 1}, {0, 2}, {0, 3}, {0, 0}},
     /*expectedSEOffsets=*/{  80,  96, 112,  64, \
                             144, 160, 176, 128, \
                             208, 224, 240, 192, \
                              16,  32,  48,   0}},

    {/*shifts=*/{1, 1}, /*axes=*/{2, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 2, 2}, /*outputTileShape=*/{1, 16, 2, 2},
     /*expectedOutputShape*/{1, 16, 4, 4}, /*expectedBackInferredInputShape=*/{1, 16, 4, 4},
     /*expectedInputTileOffset=*/{0, 0, 1, 1}, /*expectedInputTileShape=*/{1, 16, 2, 2},
     /*expectedAttrShifts=*/{0, 0}, /*expectedAttrAxes=*/{2, 3},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 16, 2, 2},
     /*expectedInputCoords*/{{3, 3}, {3, 0}, {3, 1}, {3, 2}, \
                             {0, 3}, {0, 0}, {0, 1}, {0, 2}, \
                             {1, 3}, {1, 0}, {1, 1}, {1, 2}, \
                             {2, 3}, {2, 0}, {2, 1}, {2, 2}},
     /*expectedSEOffsets=*/{ 240, 192, 208, 224, \
                              48,   0,  16,  32, \
                             112,  64,  80,  96,  \
                             176, 128, 144, 160}},

    // offsets & sizes
    {/*shifts=*/{1, 1}, /*axes=*/{2, 3}, /*offsets=*/{0, 0, 0, 0}, /*sizes=*/{1, 16, 4, 4},
     /*dataShape=*/{1, 16, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 0}, /*outputTileShape=*/{1, 16, 2, 2},
     /*expectedOutputShape*/{1, 16, 4, 4}, /*expectedBackInferredInputShape=*/{},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 2, 4},
     /*expectedAttrShifts=*/{0, 1}, /*expectedAttrAxes=*/{2, 3},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 16, 2, 2},
     /*expectedInputCoords*/{{3, 3}, {3, 0}, {3, 1}, {3, 2}, \
                             {0, 3}, {0, 0}, {0, 1}, {0, 2}, \
                             {1, 3}, {1, 0}, {1, 1}, {1, 2}, \
                             {2, 3}, {2, 0}, {2, 1}, {2, 2}},
     /*expectedSEOffsets=*/{ 240, 192, 208, 224, \
                              48,   0,  16,  32, \
                             112,  64,  80,  96,  \
                             176, 128, 144, 160}},
    //
    // Element byte sizes
    //
    {/*shifts=*/{1, 1}, /*axes=*/{2, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/2, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 0, 1}, /*outputTileShape=*/{1, 16, 4, 3},
     /*expectedOutputShape*/{1, 16, 4, 4}, /*expectedBackInferredInputShape=*/{1, 16, 4, 4},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 16, 4, 3},
     /*expectedAttrShifts=*/{1, 0}, /*expectedAttrAxes=*/{2, 3},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 16, 4, 3},
     /*expectedInputCoords*/{{3, 3}, {3, 0}, {3, 1}, {3, 2}, \
                             {0, 3}, {0, 0}, {0, 1}, {0, 2}, \
                             {1, 3}, {1, 0}, {1, 1}, {1, 2}, \
                             {2, 3}, {2, 0}, {2, 1}, {2, 2}},
     /*expectedSEOffsets=*/{ 480, 384, 416, 448, \
                              96,   0,  32,  64, \
                             224, 128, 160, 192,  \
                             352, 256, 288, 320}},


    //
    // Storage element sizes
    //
    {/*shifts=*/{2, 2}, /*axes=*/{2, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 32, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 1}, /*outputTileShape=*/{1, 32, 3, 3},
     /*expectedOutputShape*/{1, 32, 4, 4}, /*expectedBackInferredInputShape=*/{1, 32, 4, 4},
     /*expectedInputTileOffset=*/{0, 0, 0, 0}, /*expectedInputTileShape=*/{1, 32, 4, 4},
     /*expectedAttrShifts=*/{1, 1}, /*expectedAttrAxes=*/{2, 3},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 32, 3, 3},
     /*expectedInputCoords*/{{2, 2}, {2, 3}, {2, 0}, {2, 1}, \
                             {3, 2}, {3, 3}, {3, 0}, {3, 1}, \
                             {0, 2}, {0, 3}, {0, 0}, {0, 1}, \
                             {1, 2}, {1, 3}, {1, 0}, {1, 1}},
     /*expectedSEOffsets=*/{ 320, 336, 352, 368, 256, 272, 288, 304, \
                             448, 464, 480, 496, 384, 400, 416, 432, \
                              64,  80,  96, 112,   0,  16,  32,  48, \
                             192, 208, 224, 240, 128, 144, 160, 176}},

    //
    // Shifts contain Zero
    //
    {/*shifts=*/{0, 2}, /*axes=*/{2, 3}, /*offsets=*/{}, /*sizes=*/{},
     /*dataShape=*/{1, 16, 4, 4}, /*dataStrides=*/{}, /*dataElemByteSize=*/1, /*seSize=*/16,
     /*outputTileOffset=*/{0, 0, 1, 0}, /*outputTileShape=*/{1, 16, 3, 3},
     /*expectedOutputShape*/{1, 16, 4, 4}, /*expectedBackInferredInputShape=*/{1, 16, 4, 4},
     /*expectedInputTileOffset=*/{0, 0, 1, 0}, /*expectedInputTileShape=*/{1, 16, 3, 4},
     /*expectedAttrShifts=*/{0, 2}, /*expectedAttrAxes=*/{2, 3},
     /*expectedAttrOffsets=*/{0, 0, 0, 0}, /*expectedAttrSizes=*/{1, 16, 3, 3},
     /*expectedInputCoords*/{{0, 2}, {0, 3}, {0, 0}, {0, 1}, \
                             {1, 2}, {1, 3}, {1, 0}, {1, 1}, \
                             {2, 2}, {2, 3}, {2, 0}, {2, 1}, \
                             {3, 2}, {3, 3}, {3, 0}, {3, 1}},
     /*expectedSEOffsets=*/{  32,  48,   0,  16, \
                              96, 112,  64,  80, \
                             160, 176, 128, 144, \
                             224, 240, 192, 208}}
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(unit, SERollAttrTests, testing::ValuesIn(rollParams));

// SEDilatedConvAttr
struct SEDilatedConvAttrParams {
    // SEDilatedConvAttr parameters
    SmallVector<int64_t> dilation;
    SmallVector<int64_t> kernelStride;
    SmallVector<int64_t> kernelSize;
    SmallVector<int64_t> dataOffset;
    SmallVector<int64_t> dataSizes;

    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;

    // Input data
    SmallVector<int64_t> dataShape;
    SmallVector<Bit> dataStrides;
    int64_t dataElemByteSize;
    int64_t seSize;
    SmallVector<int64_t> outputTileOffset;
    SmallVector<int64_t> outputTileShape;

    // Expected results
    SmallVector<int64_t> expectedOutputShape;
    SmallVector<int64_t> expectedBackInferredInputShape;
    SmallVector<int64_t> expectedInputTileOffset;
    SmallVector<int64_t> expectedInputTileShape;
    SmallVector<int64_t> expectedAttrOffsets;
    SmallVector<int64_t> expectedAttrSizes;
    std::vector<std::vector<int64_t>> expectedInputCoords;
    std::vector<int32_t> expectedSEOffsets;
};

class SEDilatedConvAttrTests : public testing::TestWithParam<SEDilatedConvAttrParams> {};

TEST_P(SEDilatedConvAttrTests, SEAttrInterface) {
    auto registry = vpux::createDialectRegistry();

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPU::VPUDialect>();

    const auto params = GetParam();

    auto dilationAttr = getIntArrayAttr(&ctx, params.dilation);

    auto kernelStrideAttr = params.kernelStride.empty() ? nullptr : getIntArrayAttr(&ctx, params.kernelStride);
    auto kernelSizeAttr = params.kernelSize.empty() ? nullptr : getIntArrayAttr(&ctx, params.kernelSize);

    auto dataOffsetAttr = params.dataOffset.empty() ? nullptr : getIntArrayAttr(&ctx, params.dataOffset);
    auto dataSizesAttr = params.dataSizes.empty() ? nullptr : getIntArrayAttr(&ctx, params.dataSizes);

    auto offsetsAttr = params.offsets.empty() ? nullptr : getIntArrayAttr(&ctx, params.offsets);
    auto sizesAttr = params.sizes.empty() ? nullptr : getIntArrayAttr(&ctx, params.sizes);

    auto dilatedConvAttr = VPU::SEDilatedConvAttr::get(&ctx, dilationAttr, kernelStrideAttr, kernelSizeAttr,
                                                       dataOffsetAttr, dataSizesAttr, offsetsAttr, sizesAttr);

    auto seAttrInterface = dilatedConvAttr.dyn_cast<VPU::SEAttr>();
    ASSERT_TRUE(seAttrInterface != nullptr);

    // inferOutputShape
    const Shape dataShape(params.dataShape);
    const auto outputShape = seAttrInterface.inferOutputShape(dataShape);
    EXPECT_EQ(outputShape.raw(), params.expectedOutputShape);

    // backInferInputShape
    if (offsetsAttr == nullptr && sizesAttr == nullptr) {
        const auto inputShape = seAttrInterface.backInferInputShape(outputShape);
        EXPECT_EQ(inputShape.raw(), params.expectedBackInferredInputShape);
    }

    // backInferInputCoord
    const auto offsets = params.offsets.empty() ? SmallVector<int64_t>(outputShape.size(), 0) : params.offsets;
    const auto sizes = params.sizes.empty() ? SmallVector<int64_t>(outputShape.raw()) : params.sizes;

    const auto outputH = sizes[Dims4D::Act::H.ind()];
    const auto outputW = sizes[Dims4D::Act::W.ind()];

    ASSERT_EQ(params.expectedInputCoords.size(), outputH * outputW);

    for (auto h : irange(outputH)) {
        for (auto w : irange(outputW)) {
            const auto actualH = h + offsets[Dims4D::Act::H.ind()];
            const auto actualW = w + offsets[Dims4D::Act::W.ind()];

            const Shape outputCoord({0, 0, actualH, actualW});

            const auto inputCoord = seAttrInterface.backInferInputCoord(outputCoord, dataShape);

            const auto& expectedCoord = params.expectedInputCoords[h * outputW + w];
            const Shape expectedInputCoord({0, 0, expectedCoord[0], expectedCoord[1]});

            EXPECT_EQ(inputCoord, expectedInputCoord)
                    << "Invalid input coordinates for output coordinate [" << actualH << ", " << actualW << "]";
        }
    }

    // extractTile
    if (!params.outputTileOffset.empty() && !params.outputTileShape.empty()) {
        const Shape outputTileOffset(params.outputTileOffset);
        const Shape outputTileShape(params.outputTileShape);

        Shape inputTileOffset{};
        Shape inputTileShape{};

        auto newSEAttr = seAttrInterface.extractTile(outputTileOffset, outputTileShape, dataShape, inputTileOffset,
                                                     inputTileShape);

        auto newSEDilatedConvAttr = newSEAttr.dyn_cast_or_null<VPU::SEDilatedConvAttr>();
        ASSERT_TRUE(newSEDilatedConvAttr != nullptr);

        EXPECT_EQ(inputTileOffset.raw(), params.expectedInputTileOffset);
        EXPECT_EQ(inputTileShape.raw(), params.expectedInputTileShape);
    }

    // computeSEOffsets
    const Strides dataStrides(params.dataStrides);
    const Byte elemSize(params.dataElemByteSize);

    const auto seOffsets = seAttrInterface.computeSEOffsets(dataShape, dataStrides, elemSize, params.seSize);

    EXPECT_EQ(seOffsets, params.expectedSEOffsets);
}

// clang-format off

std::vector<SEDilatedConvAttrParams> dilatedConvParams = {
//                                                 EXAMPLE SUB-GRAPH

//
//                          NOTE: We use HxW and YxX format to align with other SEAttrs.
//

//
//      Input                     Subviews                     Sub-convolution        Outputs     Re-constructed Output
//
//                              N = 4  (Dy * Dx)               N = 4  (Dy * Dx)
//
//     Size = 8x8         Size = 8x8        Size = 8x7                              Size = 2x2       Size = 4x4
//   Dilation = 2,2      Offset = 0,0      Offset = 0,1
//                                                              4x4        4x4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2      1 1 1 1    2 2 2 2     1 1  2 2         1 2 1 2
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4      1 1 1 1    2 2 2 2     1 1  2 2         3 4 3 4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2      1 1 1 1    2 2 2 2                      1 2 1 2
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4      1 1 1 1    2 2 2 2     3 3  4 4         3 4 3 4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2                             3 3  4 4
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4        4x4        4x4
//  1 2 1 2 1 2 1 2     1 2 1 2 1 2 1 2    2 1 2 1 2 1 2      3 3 3 3    4 4 4 4
//  3 4 3 4 3 4 3 4     3 4 3 4 3 4 3 4    4 3 4 3 4 3 4      3 3 3 3    4 4 4 4
//                                                            3 3 3 3    4 4 4 4
//      Kernel            Size = 7x8        Size = 7x7        3 3 3 3    4 4 4 4
//                       Offset = 1,0      Offset = 1,1
//    Size = 3x3
//                      3 4 3 4 3 4 3 4    4 3 4 3 4 3 4
//      1 2 3           1 2 1 2 1 2 1 2    2 1 2 1 2 1 2
//      4 5 6           3 4 3 4 3 4 3 4    4 3 4 3 4 3 4
//      7 8 9           1 2 1 2 1 2 1 2    2 1 2 1 2 1 2
//                      3 4 3 4 3 4 3 4    4 3 4 3 4 3 4
//                      1 2 1 2 1 2 1 2    2 1 2 1 2 1 2
//                      3 4 3 4 3 4 3 4    4 3 4 3 4 3 4


    // TEST-CASE 1

    {/* dilation = */ { 2, 2 },

     /* kernelStride = */ { 1, 1 },
     /* kernelSize */     { 3, 3 },

     /* dataOffset = */ { 0, 0,  0, 1 },
     /* dataSizes */    { 1, 16, 8, 7 },

     /* offsets = */ {},
     /* sizes = */ {},

     /* dataShape = */ { 1, 16, 8, 7 },
     /* dataStrides = */ {},
     /* dataElemByteSize = */ 1,
     /* seSize = */ 16,

     /* outputTileOffset = */ { 0, 0,  0, 0 },
     /* outputTileShape = */  { 1, 16, 4, 4 },

     /* expectedOutputShape = */            { 1, 16, 4, 3 },
     /* expectedBackInferredInputShape = */ { 1, 16, 8, 6 },

     /* expectedInputTileOffset = */ { 0, 0,  0, 0 },
     /* expectedInputTileShape = */  { 1, 16, 8, 7 },

     /* expectedAttrOffsets = */ { 0, 0,  0, 0 },
     /* expectedAttrSizes = */   { 1, 16, 8, 7 },

     /* expectedInputCoords = */ { {0, 1}, {0, 3}, {0, 5},
                                   {2, 1}, {2, 3}, {2, 5},
                                   {4, 1}, {4, 3}, {4, 5},
                                   {6, 1}, {6, 3}, {6, 5} },

     /* expectedSEOffsets = */ { 16,  48,  80,
                                 240, 272, 304,
                                 464, 496, 528,
                                 688, 720, 752 }},


    // TEST-CASE 2

    {/* dilation = */ { 2, 2 },

     /* kernelStride = */ { 1, 1 },
     /* kernelSize */     { 2, 2 },

     /* dataOffset = */ { 0, 0,  0,  0 },
     /* dataSizes */    { 1, 16, 12, 12 },

     /* offsets = */ {},
     /* sizes = */ {},

     /* dataShape = */ { 1, 16, 12, 12 },
     /* dataStrides = */ {},
     /* dataElemByteSize = */ 1,
     /* seSize = */ 16,

     /* outputTileOffset = */ { 0, 0,  0, 0 },
     /* outputTileShape = */  { 1, 16, 6, 6 },

     /* expectedOutputShape = */            { 1, 16, 6,  6 },
     /* expectedBackInferredInputShape = */ { 1, 16, 12, 12 },

     /* expectedInputTileOffset = */ { 0, 0,  0,  0 },
     /* expectedInputTileShape = */  { 1, 16, 12, 12 },

     /* expectedAttrOffsets = */ { 0, 0,  0,  0 },
     /* expectedAttrSizes = */   { 1, 16, 12, 12 },

     /* expectedInputCoords = */ { {0,  0}, {0,  2}, {0,  4}, {0, 6},  {0, 8},  {0,  10},
                                   {2,  0}, {2,  2}, {2,  4}, {2, 6},  {2, 8},  {2,  10},
                                   {4,  0}, {4,  2}, {4,  4}, {4, 6},  {4, 8},  {4,  10},
                                   {6,  0}, {6,  2}, {6,  4}, {6, 6},  {6, 8},  {6,  10},
                                   {8,  0}, {8,  2}, {8,  4}, {8, 6},  {8, 8},  {8,  10},
                                   {10, 0}, {10, 2}, {10, 4}, {10, 6}, {10, 8}, {10, 10} },

     /* expectedSEOffsets = */ { 0,    32,   64,   96,   128,  160,
                                 384,  416,  448,  480,  512,  544,
                                 768,  800,  832,  864,  896,  928,
                                 1152, 1184, 1216, 1248, 1280, 1312,
                                 1536, 1568, 1600, 1632, 1664, 1696,
                                 1920, 1952, 1984, 2016, 2048, 2080 }},


    // TEST-CASE 3

    {/* dilation = */ { 2, 1 },

     /* kernelStride = */ { 1, 1 },
     /* kernelSize */     { 1, 1 },

     /* dataOffset = */ { 0, 0,  0,  0 },
     /* dataSizes */    { 1, 16, 32, 1 },

     /* offsets = */ {},
     /* sizes = */ {},

     /* dataShape = */ { 1, 16, 32, 1 },
     /* dataStrides = */ {},
     /* dataElemByteSize = */ 1,
     /* seSize = */ 16,

     /* outputTileOffset = */ { 0, 0,  0,  0 },
     /* outputTileShape = */  { 1, 16, 16, 1 },

     /* expectedOutputShape = */            { 1, 16, 16, 1 },
     /* expectedBackInferredInputShape = */ { 1, 16, 32, 1 },

     /* expectedInputTileOffset = */ { 0, 0,  0,  0 },
     /* expectedInputTileShape = */  { 1, 16, 32, 1 },

     /* expectedAttrOffsets = */ { 0, 0,  0,  0 },
     /* expectedAttrSizes = */   { 1, 16, 16, 1 },

     /* expectedInputCoords = */ { {0,  0}, {2,  0}, {4,  0}, {6,  0},
                                   {8,  0}, {10, 0}, {12, 0}, {14, 0},
                                   {16, 0}, {18, 0}, {20, 0}, {22, 0},
                                   {24, 0}, {26, 0}, {28, 0}, {30, 0} },

     /* expectedSEOffsets = */ { 0,   32,  64,  96,
                                 128, 160, 192, 224,
                                 256, 288, 320, 352,
                                 384, 416, 448, 480 }},

    // TEST-CASE 4

    {/* dilation = */ { 2, 2 },

     /* kernelStride = */ { 1, 1 },
     /* kernelSize */     { 3, 3 },

     /* dataOffset = */ { 0, 0,  1, 1 },
     /* dataSizes */    { 1, 16, 7, 7 },

     /* offsets = */ {},
     /* sizes = */ {},

     /* dataShape = */ { 1, 16, 8, 8 },
     /* dataStrides = */ {},
     /* dataElemByteSize = */ 1,
     /* seSize = */ 16,

     /* outputTileOffset = */ { 0, 0,  0, 0 },
     /* outputTileShape = */  { 1, 16, 4, 4 },

     /* expectedOutputShape = */            { 1, 16, 4, 4 },
     /* expectedBackInferredInputShape = */ { 1, 16, 8, 8 },

     /* expectedInputTileOffset = */ { 0, 0,  0, 0 },
     /* expectedInputTileShape = */  { 1, 16, 8, 8 },

     /* expectedAttrOffsets = */ { 0, 0,  0, 0 },
     /* expectedAttrSizes = */   { 1, 16, 8, 7 },

     /* expectedInputCoords = */ { {1, 1}, {1, 3}, {1, 5}, {1, 7},
                                   {3, 1}, {3, 3}, {3, 5}, {3, 7},
                                   {5, 1}, {5, 3}, {5, 5}, {5, 7},
                                   {7, 1}, {7, 3}, {7, 5}, {7, 7} },

     /* expectedSEOffsets = */ { 144, 176, 208, 240,
                                 400, 432, 464, 496,
                                 656, 688, 720, 752,
                                 912, 944, 976, 1008}},
};

// clang-format on

INSTANTIATE_TEST_SUITE_P(unit, SEDilatedConvAttrTests, testing::ValuesIn(dilatedConvParams));
