//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/feasible_scheduler_utils.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>

using namespace vpux;

using MLIR_ControlEdgeGenerator = MLIR_UnitBase;

TEST_F(MLIR_ControlEdgeGenerator, TestMemOverlapEdges) {
    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(1, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(3, 0, 50, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(4, 51, 100, ScheduledOpOneResource::EResRelation::PRODUCER),
            ScheduledOpOneResource(5, 0, 50, ScheduledOpOneResource::EResRelation::CONSUMER),
            ScheduledOpOneResource(6, 51, 100, ScheduledOpOneResource::EResRelation::CONSUMER),
    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 1,2
    // 1,2 -> 3
    // 1,2 -> 4
    // 3 -> 5
    // 4 -> 6
    SmallVector<ControlEdge> expectedControlEdges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {1, 4}, {2, 4}, {3, 5}, {4, 6}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST_F(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest1) {
    mlir::Value dummyBuffer = nullptr;

    ScheduledOpOneResource::ResourceView resView0({{dummyBuffer}, {0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{dummyBuffer}, {1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView0),
            ScheduledOpOneResource(3, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView1)};

    // For above configuration expected inserted memory control edges are:
    // 0 -> 2
    // 1 -> 3
    SmallVector<ControlEdge> expectedControlEdges = {{0, 2}, {1, 3}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST_F(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest2) {
    mlir::Value dummyBuffer = nullptr;

    ScheduledOpOneResource::ResourceView resView0({{dummyBuffer}, {0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{dummyBuffer}, {1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView0),
            ScheduledOpOneResource(3, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView1),
            ScheduledOpOneResource(4, 50, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),

    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 2
    // 1 -> 3
    // 2,3 -> 4
    SmallVector<ControlEdge> expectedControlEdges = {{0, 2}, {1, 3}, {2, 4}, {3, 4}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST_F(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest3) {
    mlir::Value dummyBuffer = nullptr;

    ScheduledOpOneResource::ResourceView resView0({{dummyBuffer}, {0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{dummyBuffer}, {1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(2, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 1
    // 0 -> 2
    SmallVector<ControlEdge> expectedControlEdges = {{0, 1}, {0, 2}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST_F(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest4) {
    mlir::Value dummyBuffer = nullptr;

    ScheduledOpOneResource::ResourceView resView0({{dummyBuffer}, {0}, {1}, {1}});
    ScheduledOpOneResource::ResourceView resView1({{dummyBuffer}, {1}, {1}, {1}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 101, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(2, 0, 100, ScheduledOpOneResource::EResRelation::CONSUMER, resView0),
            ScheduledOpOneResource(3, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(4, 0, 150, ScheduledOpOneResource::EResRelation::PRODUCER, resView1),
    };

    // For above configuration expected inserted memory control edges are:
    // 0 -> 2
    // 1,2 -> 3
    // 1,2 -> 4
    SmallVector<ControlEdge> expectedControlEdges = {{0, 2}, {2, 3}, {1, 3}, {1, 4}, {2, 4}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}

TEST_F(MLIR_ControlEdgeGenerator, TestMemOverlapEdgesWithSubViewTest5) {
    // auto registry = createDialectRegistry();
    // mlir::MLIRContext ctx(registry);
    // ctx.loadDialect<mlir::quant::QuantDialect>();
    // ctx.loadDialect<Const::ConstDialect>();

    // mlir::OpBuilder builder(&ctx);
    // auto declareOp = builder.create<Const::DeclareOp>(mlir::UnknownLoc::get(&ctx), quantDataType, contentAttr);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();
    ctx.loadDialect<VPURT::VPURTDialect>();

    const Shape shape{100};
    const DimsOrder order = DimsOrder::C;
    const mlir::AffineMapAttr layout = mlir::AffineMapAttr::get(order.toAffineMap(&ctx));
    const IndexedSymbolAttr memSpace = IndexedSymbolAttr::get(&ctx, "CMX_NN");
    const auto memrefType = mlir::MemRefType::get(shape.raw(), mlir::Float16Type::get(&ctx), layout, memSpace);

    mlir::OpBuilder builder(&ctx);
    mlir::OwningOpRef<VPURT::DeclareBufferOp> dummyBuffer0 = builder.create<VPURT::DeclareBufferOp>(
            mlir::UnknownLoc::get(&ctx), memrefType, VPURT::BufferSection::CMX_NN, /*byte_offset=*/0);

    mlir::OwningOpRef<VPURT::DeclareBufferOp> dummyBuffer1 = builder.create<VPURT::DeclareBufferOp>(
            mlir::UnknownLoc::get(&ctx), memrefType, VPURT::BufferSection::CMX_NN, /*byte_offset=*/0);

    ScheduledOpOneResource::ResourceView resView0({{dummyBuffer0->getBuffer()}, {0}, {50}, {100}});
    ScheduledOpOneResource::ResourceView resView1({{dummyBuffer1->getBuffer()}, {50}, {50}, {100}});

    // Create example schedule where operations execute in sequence and either produce
    // or consume certain range of memory
    std::vector<ScheduledOpOneResource> scheduledOpsResources = {
            ScheduledOpOneResource(0, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView0),
            ScheduledOpOneResource(1, 0, 100, ScheduledOpOneResource::EResRelation::PRODUCER, resView1)};

    // For above configuration expected inserted memory control edges are:
    // 0 -> 1
    // Even though subview themselves define non overlapping ranges, buffers themselves are different thus edge is
    // needed
    SmallVector<ControlEdge> expectedControlEdges = {{0, 1}};

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    ASSERT_EQ(controlEdges.size(), expectedControlEdges.size());

    for (size_t i = 0; i < controlEdges.size(); i++) {
        EXPECT_EQ(controlEdges[i]._source, expectedControlEdges[i]._source);
        EXPECT_EQ(controlEdges[i]._sink, expectedControlEdges[i]._sink);
    }
}
