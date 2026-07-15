//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::VPU;

// Fixture providing an MLIRContext with arith dialect loaded, for IR-building tests.
class SCFEncodingWithContext : public MLIR_UnitBase {
public:
    SCFEncodingWithContext() {
        ctx = std::make_unique<mlir::MLIRContext>(registry);
        ctx->loadDialect<mlir::arith::ArithDialect>();
    }

protected:
    std::unique_ptr<mlir::MLIRContext> ctx;
};

// ---------------------------------------------------------------------------
// getTilePositionShift
// ---------------------------------------------------------------------------

TEST(SCFTilePositionEncoding, GetShift_AllNCHWDims) {
    EXPECT_EQ(getTilePositionShift(Dims4D::Act::N), 6);
    EXPECT_EQ(getTilePositionShift(Dims4D::Act::C), 4);
    EXPECT_EQ(getTilePositionShift(Dims4D::Act::H), 2);
    EXPECT_EQ(getTilePositionShift(Dims4D::Act::W), 0);
}

TEST(SCFTilePositionEncoding, GetShift_InvalidDim_Throws) {
    EXPECT_ANY_THROW(getTilePositionShift(Dim(4)));
}

// ---------------------------------------------------------------------------
// getTilePaddingMask
// ---------------------------------------------------------------------------

TEST(SCFTilePositionEncoding, PaddingMask_AllPositions) {
    EXPECT_EQ(getTilePaddingMask(TilePosition::MIDDLE), (std::pair<bool, bool>{false, false}));
    EXPECT_EQ(getTilePaddingMask(TilePosition::END), (std::pair<bool, bool>{false, true}));
    EXPECT_EQ(getTilePaddingMask(TilePosition::START), (std::pair<bool, bool>{true, false}));
    EXPECT_EQ(getTilePaddingMask(TilePosition::FULLBLK), (std::pair<bool, bool>{true, true}));
}

// ---------------------------------------------------------------------------
// getEncodedPointPosition / getDimPosition — round-trip
// ---------------------------------------------------------------------------

TEST(SCFTilePositionEncoding, PointEncode_EmptyInput) {
    EXPECT_EQ(getEncodedPointPosition({}), 0);
}

TEST(SCFTilePositionEncoding, PointEncode_SingleDim_AllPositions) {
    for (auto pos : {TilePosition::MIDDLE, TilePosition::END, TilePosition::START, TilePosition::FULLBLK}) {
        for (auto dim : {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}) {
            int64_t encoded = getEncodedPointPosition({{dim, pos}});
            EXPECT_EQ(getDimPosition(encoded, dim), pos);
        }
    }
}

TEST(SCFTilePositionEncoding, PointEncode_SingleDim_OtherDimsAreMiddle) {
    int64_t encoded = getEncodedPointPosition({{Dims4D::Act::H, TilePosition::START}});
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::N), TilePosition::MIDDLE);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::C), TilePosition::MIDDLE);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::W), TilePosition::MIDDLE);
}

TEST(SCFTilePositionEncoding, PointEncode_AllDims_NoOverlap) {
    int64_t encoded = getEncodedPointPosition({
            {Dims4D::Act::N, TilePosition::START},
            {Dims4D::Act::C, TilePosition::END},
            {Dims4D::Act::H, TilePosition::MIDDLE},
            {Dims4D::Act::W, TilePosition::FULLBLK},
    });
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::N), TilePosition::START);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::C), TilePosition::END);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::H), TilePosition::MIDDLE);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::W), TilePosition::FULLBLK);
}

// ---------------------------------------------------------------------------
// setDimPosition
// ---------------------------------------------------------------------------

TEST(SCFTilePositionEncoding, SetDimPosition_DoesNotCorruptOtherDims) {
    int64_t encoded = getEncodedPointPosition({
            {Dims4D::Act::N, TilePosition::FULLBLK},
            {Dims4D::Act::C, TilePosition::FULLBLK},
            {Dims4D::Act::H, TilePosition::FULLBLK},
            {Dims4D::Act::W, TilePosition::FULLBLK},
    });
    encoded = setDimPosition(encoded, Dims4D::Act::H, TilePosition::MIDDLE);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::N), TilePosition::FULLBLK);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::C), TilePosition::FULLBLK);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::H), TilePosition::MIDDLE);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::W), TilePosition::FULLBLK);
}

TEST(SCFTilePositionEncoding, SetDimPosition_Overwrite) {
    int64_t encoded = getEncodedPointPosition({{Dims4D::Act::W, TilePosition::START}});
    encoded = setDimPosition(encoded, Dims4D::Act::W, TilePosition::END);
    EXPECT_EQ(getDimPosition(encoded, Dims4D::Act::W), TilePosition::END);
}

TEST(SCFTilePositionEncoding, SetDimPosition_AllDimsAllPositions) {
    for (auto dim : {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}) {
        for (auto pos : {TilePosition::MIDDLE, TilePosition::END, TilePosition::START, TilePosition::FULLBLK}) {
            int64_t encoded = setDimPosition(0, dim, pos);
            EXPECT_EQ(getDimPosition(encoded, dim), pos);
        }
    }
}

// ---------------------------------------------------------------------------
// decodePointPosition — round-trip with getEncodedPointPosition
// ---------------------------------------------------------------------------

TEST(SCFTilePositionEncoding, DecodePointPosition_RoundTrip) {
    int64_t encoded = getEncodedPointPosition({
            {Dims4D::Act::N, TilePosition::START},
            {Dims4D::Act::H, TilePosition::END},
    });
    SmallVector<vpux::Dim> dims = {Dims4D::Act::N, Dims4D::Act::H};
    auto positions = decodePointPosition(encoded, dims);
    ASSERT_EQ(positions.size(), 2u);
    EXPECT_EQ(positions[0], TilePosition::START);
    EXPECT_EQ(positions[1], TilePosition::END);
}

TEST(SCFTilePositionEncoding, DecodePointPosition_EmptyDims) {
    auto positions = decodePointPosition(0xFF, {});
    EXPECT_TRUE(positions.empty());
}

// ---------------------------------------------------------------------------
// encodeRangePosition / decodeRangePosition — round-trip
// ---------------------------------------------------------------------------

TEST(SCFTilePositionEncoding, RangeEncode_SingleDim_AllCombinations) {
    for (auto dim : {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W}) {
        for (auto start : {TilePosition::MIDDLE, TilePosition::END, TilePosition::START, TilePosition::FULLBLK}) {
            for (auto end : {TilePosition::MIDDLE, TilePosition::END, TilePosition::START, TilePosition::FULLBLK}) {
                int64_t encoded = encodeRangePosition(dim, 0, start, end);
                auto [decodedStart, decodedEnd] = decodeRangePosition(encoded, dim);
                EXPECT_EQ(decodedStart, start) << "start mismatch for dim=" << dim.ind();
                EXPECT_EQ(decodedEnd, end) << "end mismatch for dim=" << dim.ind();
            }
        }
    }
}

TEST(SCFTilePositionEncoding, RangeEncode_AllDims_NoOverlap) {
    int64_t encoded = 0;
    encoded = encodeRangePosition(Dims4D::Act::N, encoded, TilePosition::START, TilePosition::END);
    encoded = encodeRangePosition(Dims4D::Act::C, encoded, TilePosition::END, TilePosition::START);
    encoded = encodeRangePosition(Dims4D::Act::H, encoded, TilePosition::MIDDLE, TilePosition::FULLBLK);
    encoded = encodeRangePosition(Dims4D::Act::W, encoded, TilePosition::FULLBLK, TilePosition::MIDDLE);

    auto [ns, ne] = decodeRangePosition(encoded, Dims4D::Act::N);
    auto [cs, ce] = decodeRangePosition(encoded, Dims4D::Act::C);
    auto [hs, he] = decodeRangePosition(encoded, Dims4D::Act::H);
    auto [ws, we] = decodeRangePosition(encoded, Dims4D::Act::W);

    EXPECT_EQ(ns, TilePosition::START);
    EXPECT_EQ(ne, TilePosition::END);
    EXPECT_EQ(cs, TilePosition::END);
    EXPECT_EQ(ce, TilePosition::START);
    EXPECT_EQ(hs, TilePosition::MIDDLE);
    EXPECT_EQ(he, TilePosition::FULLBLK);
    EXPECT_EQ(ws, TilePosition::FULLBLK);
    EXPECT_EQ(we, TilePosition::MIDDLE);
}

TEST(SCFTilePositionEncoding, RangeEncode_ZeroBaseValue_Preserved) {
    // encodeRangePosition ORs into currentValue — starting from 0 should give clean encoding
    int64_t encoded = encodeRangePosition(Dims4D::Act::W, 0, TilePosition::START, TilePosition::END);
    auto [s, e] = decodeRangePosition(encoded, Dims4D::Act::W);
    EXPECT_EQ(s, TilePosition::START);
    EXPECT_EQ(e, TilePosition::END);
}

// ---------------------------------------------------------------------------
// Negative tests — invalid dim propagation
// ---------------------------------------------------------------------------

TEST(SCFTilePositionEncoding, GetShift_NegativeDim_Throws) {
    // Covers the dim.ind() < 0 branch, distinct from the >= 4 branch already tested.
    EXPECT_ANY_THROW(getTilePositionShift(Dim(-1)));
}

TEST(SCFTilePositionEncoding, GetEncodedPointPosition_InvalidDim_Throws) {
    EXPECT_ANY_THROW(getEncodedPointPosition({{Dim(5), TilePosition::START}}));
}

TEST(SCFTilePositionEncoding, GetDimPosition_InvalidDim_Throws) {
    EXPECT_ANY_THROW(getDimPosition(0, Dim(5)));
}

TEST(SCFTilePositionEncoding, SetDimPosition_InvalidDim_Throws) {
    EXPECT_ANY_THROW(setDimPosition(0, Dim(5), TilePosition::START));
}

TEST(SCFTilePositionEncoding, DecodePointPosition_InvalidDimInList_Throws) {
    SmallVector<vpux::Dim> dims = {Dims4D::Act::H, Dim(5)};
    EXPECT_ANY_THROW(decodePointPosition(0, dims));
}

TEST(SCFTilePositionEncoding, EncodeRangePosition_InvalidDim_Throws) {
    EXPECT_ANY_THROW(encodeRangePosition(Dim(5), 0, TilePosition::START, TilePosition::END));
}

TEST(SCFTilePositionEncoding, DecodeRangePosition_InvalidDim_Throws) {
    EXPECT_ANY_THROW(decodeRangePosition(0, Dim(5)));
}

// ---------------------------------------------------------------------------
// Negative test — encodePointPosition with empty values (validates Issue 1 fix)
// ---------------------------------------------------------------------------

TEST_F(SCFEncodingWithContext, EncodePointPosition_EmptyValues_Throws) {
    mlir::OpBuilder builder(ctx.get());
    auto loc = builder.getUnknownLoc();
    EXPECT_ANY_THROW(encodePointPosition(builder, loc, {}, {}));
}

// ---------------------------------------------------------------------------
// Tests proving potential bugs in the added encoding functions
// ---------------------------------------------------------------------------

// Bug 3: encodePointPosition requires values.size() == shiftAmounts.size().
// Guard added to catch mismatch before llvm::zip would silently truncate.
TEST_F(SCFEncodingWithContext, EncodePointPosition_ShiftAmountsShorterThanValues_Throws) {
    // Use a stack-allocated Block to own the ops so they are freed on unwind.
    mlir::Block block;
    mlir::OpBuilder builder(ctx.get());
    builder.setInsertionPointToStart(&block);
    auto loc = builder.getUnknownLoc();

    auto hVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 2).getResult();
    auto wVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 1).getResult();
    auto hShift = builder.create<mlir::arith::ConstantIndexOp>(loc, getTilePositionShift(Dims4D::Act::H)).getResult();
    SmallVector<mlir::Value> values = {hVal, wVal};
    SmallVector<mlir::Value> shiftAmounts = {hShift};  // missing wShift

    EXPECT_ANY_THROW(encodePointPosition(builder, loc, values, shiftAmounts));
}
