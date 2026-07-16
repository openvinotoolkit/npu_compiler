//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/precomputed_strategy_table_cache.hpp"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace vpux::VPU {
uint64_t computeOpStrategyHash(mlir::Operation* op, StringRef device, int64_t numTiles);
}  // namespace vpux::VPU

namespace {

// ── Minimal PTC binary builder ─────────────────────────────────────────────────
// Layout matches the format documented in precomputed_strategy_table_cache.cpp:
//   [PtcHeader 16B][HT: ht_slots × uint32][PtcRecord × n_entries 64B]

constexpr char kMagic[4] = {'V', 'P', 'X', 'T'};
constexpr uint16_t kVersion = 2;
constexpr uint16_t kNoneU16 = 0xFFFFu;
constexpr uint32_t kEmptySlot = 0xFFFFFFFFu;

template <typename T>
void writeLE(std::vector<uint8_t>& buf, size_t offset, T val) {
    std::memcpy(buf.data() + offset, &val, sizeof(T));
}

struct TestRecord {
    uint64_t key;
    uint16_t spatialIdx = kNoneU16;
    int32_t temporal[5] = {};
};

std::vector<uint8_t> buildPtcBuffer(const std::vector<TestRecord>& records) {
    const uint32_t n = static_cast<uint32_t>(records.size());
    uint32_t htSlots = 2;
    while (htSlots < 2 * n) {
        htSlots <<= 1;
    }

    const size_t hdrSz = 16;
    const size_t htSz = htSlots * 4;
    std::vector<uint8_t> buf(hdrSz + htSz + n * 64, 0);

    std::memcpy(buf.data(), kMagic, 4);
    writeLE<uint16_t>(buf, 4, kVersion);
    writeLE<uint32_t>(buf, 8, n);
    writeLE<uint32_t>(buf, 12, htSlots);

    for (uint32_t i = 0; i < htSlots; ++i) {
        writeLE<uint32_t>(buf, hdrSz + i * 4, kEmptySlot);
    }

    for (uint32_t i = 0; i < n; ++i) {
        const auto& r = records[i];
        const size_t base = hdrSz + htSz + i * 64;

        writeLE<uint64_t>(buf, base + 0, r.key);
        writeLE<uint16_t>(buf, base + 8, r.spatialIdx);
        writeLE<uint16_t>(buf, base + 10, kNoneU16);  // pipeline_idx (unused)
        for (int d = 0; d < 5; ++d) {
            writeLE<int32_t>(buf, base + 16 + d * 4, r.temporal[d]);
        }

        // Insert into hash table with linear probing.
        uint32_t probe = static_cast<uint32_t>(r.key) & (htSlots - 1u);
        while (true) {
            uint32_t slot;
            std::memcpy(&slot, buf.data() + hdrSz + probe * 4, 4);
            if (slot == kEmptySlot) {
                writeLE<uint32_t>(buf, hdrSz + probe * 4, i);
                break;
            }
            probe = (probe + 1u) & (htSlots - 1u);
        }
    }
    return buf;
}

// 1x16x16x16 convolution with a 1x1 kernel; no pre-assigned strategies.
constexpr vpux::StringLiteral kTestIR = R"(
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    module @test attributes {config.platform = #config.platform<NPU5010>} {
        config.Resources 3 of @NCE {
            config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        }
        func.func @main(%arg0:    tensor<1x16x16x16xf16, {order = #NHWC}>,
                        %weights: tensor<16x16x1x1xf16,  {order = #NHWC}>,
                        %wt:      tensor<16x1x1x4xsi32>)
              -> tensor<1x16x16x16xf16, {order = #NHWC}> {
            %0 = VPU.NCE.Convolution(%arg0, %weights, %wt)
                rawFilterShape [16, 16, 1, 1]
                {
                    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    strides = [1, 1]
                } : tensor<1x16x16x16xf16, {order = #NHWC}>,
                   tensor<16x16x1x1xf16,  {order = #NHWC}>,
                   tensor<16x1x1x4xsi32>
              -> tensor<1x16x16x16xf16, {order = #NHWC}>
            return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>
        }
    })";

// Parses kTestIR and returns the first NCE op found, or nullptr on failure.
mlir::Operation* parseAndGetNceOp(mlir::MLIRContext& ctx, mlir::OwningOpRef<mlir::ModuleOp>& module,
                                  mlir::func::FuncOp& func) {
    module = mlir::parseSourceString<mlir::ModuleOp>(kTestIR, &ctx);
    if (!module) {
        return nullptr;
    }
    func = module->lookupSymbol<mlir::func::FuncOp>("main");
    if (!func) {
        return nullptr;
    }

    mlir::Operation* nceOp = nullptr;
    func->walk([&](mlir::Operation* op) {
        if (mlir::isa<vpux::VPU::NCEOpInterface>(op)) {
            nceOp = op;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return nceOp;
}

}  // namespace

using PtcBinaryTest = vpux::VPU::arch50xx::UnitTest;

TEST_F(PtcBinaryTest, InvalidBuffer_NoCrash) {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp func;
    mlir::Operation* nceOp = parseAndGetNceOp(ctx, module, func);
    ASSERT_TRUE(nceOp != nullptr);

    const std::vector<uint8_t> tiny = {0x00, 0x01};
    vpux::VPU::applyPrecomputedStrategyTableBinary(tiny, func, vpux::Logger::global());

    EXPECT_FALSE(nceOp->hasAttr(vpux::VPU::pinnedStrategy));
}

TEST_F(PtcBinaryTest, EmptyTable_AllMiss) {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp func;
    mlir::Operation* nceOp = parseAndGetNceOp(ctx, module, func);
    ASSERT_TRUE(nceOp != nullptr);

    const auto buf = buildPtcBuffer({});
    vpux::VPU::applyPrecomputedStrategyTableBinary(buf, func, vpux::Logger::global());

    EXPECT_FALSE(nceOp->hasAttr(vpux::VPU::pinnedStrategy));
}

TEST_F(PtcBinaryTest, Hit_TemporalTiling) {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp func;
    mlir::Operation* nceOp = parseAndGetNceOp(ctx, module, func);
    ASSERT_TRUE(nceOp != nullptr);

    TestRecord rec;
    rec.key = vpux::VPU::computeOpStrategyHash(nceOp, "NPU50XX", /*numTiles=*/3);
    rec.temporal[0] = 2;

    const auto buf = buildPtcBuffer({rec});
    vpux::VPU::applyPrecomputedStrategyTableBinary(buf, func, vpux::Logger::global());

    EXPECT_TRUE(nceOp->hasAttr(vpux::VPU::pinnedStrategy));
    EXPECT_TRUE(nceOp->hasAttr("tilingStrategy"));
}

TEST_F(PtcBinaryTest, Hit_SpatialStrategy) {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::func::FuncOp func;
    mlir::Operation* nceOp = parseAndGetNceOp(ctx, module, func);
    ASSERT_TRUE(nceOp != nullptr);

    const auto targetStrategy = vpux::VPU::MultiClusterStrategy::SplitOverHeight;

    TestRecord rec;
    rec.key = vpux::VPU::computeOpStrategyHash(nceOp, "NPU50XX", /*numTiles=*/3);
    rec.spatialIdx = static_cast<uint16_t>(static_cast<uint32_t>(targetStrategy));

    const auto buf = buildPtcBuffer({rec});
    vpux::VPU::applyPrecomputedStrategyTableBinary(buf, func, vpux::Logger::global());

    EXPECT_TRUE(nceOp->hasAttr(vpux::VPU::pinnedStrategy));
    auto clusteredOp = mlir::dyn_cast<vpux::VPU::ClusteredOpInterface>(nceOp);
    ASSERT_TRUE(clusteredOp);
    ASSERT_TRUE(clusteredOp.getMultiClusterStrategy().has_value());
    EXPECT_EQ(clusteredOp.getMultiClusterStrategy().value(), targetStrategy);
}
