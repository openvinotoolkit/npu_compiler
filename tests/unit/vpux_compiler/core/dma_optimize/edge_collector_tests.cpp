//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Run cmd: npuUnitTests --gtest_filter="MLIR_EdgeCollector.*"
//
// Unit tests for the EdgeCollector data model. The snippets below keep the IR
// small while preserving representative Normal, FanIn, FanOut, and Many2Many
// DMA/view topologies seen in model IR.

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_collector.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/edge_model.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_optimization/loop_region.hpp"

#include "common/utils.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

using BoundaryKind = VPUIP::DmaOptimization::EdgeBoundary::Kind;
using EdgeType = VPUIP::DmaOptimization::EdgeType;

void expectEdgeShape(const VPUIP::DmaOptimization::Edge& edge, EdgeType type, size_t sources, size_t targets,
                     size_t nodes) {
    EXPECT_EQ(edge.type, type);
    ASSERT_EQ(edge.sources.size(), sources);
    ASSERT_EQ(edge.targets.size(), targets);
    ASSERT_EQ(edge.nodes.size(), nodes);
}

void expectLoopDomain(const VPUIP::DmaOptimization::LoopRegionInfo& info, VPUIP::DmaOptimization::LoopRegionKind kind,
                      int64_t loopIndex, int64_t tileIndex = -1) {
    EXPECT_TRUE(info.isValid());
    EXPECT_EQ(info.kind, kind);
    EXPECT_EQ(info.loopIndex, loopIndex);
    if (tileIndex >= 0) {
        EXPECT_EQ(info.tileIndex, tileIndex);
    }
}

bool hasTargetLoopDomain(const VPUIP::DmaOptimization::Edge& edge, VPUIP::DmaOptimization::LoopRegionKind kind,
                         int64_t loopIndex, int64_t tileIndex) {
    return llvm::any_of(edge.targets, [&](const VPUIP::DmaOptimization::EdgeBoundary& target) {
        const auto& info = target.loopRegionInfo;
        return info.kind == kind && info.loopIndex == loopIndex && info.tileIndex == tileIndex;
    });
}

class MLIR_EdgeCollector : public MLIR_UnitBase {
public:
    MLIR_EdgeCollector() {
        _ctx.appendDialectRegistry(registry);
        _ctx.loadAllAvailableDialects();
    }

protected:
    mlir::MLIRContext _ctx;
    Logger _log = Logger::global();

    mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef ir) {
        auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &_ctx);
        EXPECT_NE(module.get(), nullptr) << "Failed to parse test IR";
        return module;
    }

    mlir::func::FuncOp lookupMainFunction(mlir::ModuleOp module) {
        if (module == nullptr) {
            return nullptr;
        }
        auto mainFunction = module.lookupSymbol<mlir::func::FuncOp>("main");
        EXPECT_NE(mainFunction, nullptr) << "Module has no @main function";
        return mainFunction;
    }

    VPUIP::DmaOptimization::EdgeCollection collectEdges(mlir::func::FuncOp func) {
        if (func == nullptr) {
            return {};
        }
        VPUIP::DmaOptimization::EdgeCollector collector(_log);
        return collector.collect(func);
    }

    VPUIP::DmaOptimization::EdgeCollection collectEdges(
            mlir::func::FuncOp func, const VPUIP::DmaOptimization::LoopRegionBoundaryInfo& boundaries) {
        if (func == nullptr) {
            return {};
        }
        VPUIP::DmaOptimization::EdgeCollector collector(_log);
        return collector.collect(func, boundaries);
    }

    VPUIP::DmaOptimization::EdgeCollection collectMainEdges(mlir::ModuleOp module) {
        return collectEdges(lookupMainFunction(module));
    }

    VPUIP::DmaOptimization::EdgeCollection collectMainEdges(
            mlir::ModuleOp module, const VPUIP::DmaOptimization::LoopRegionBoundaryInfo& boundaries) {
        return collectEdges(lookupMainFunction(module), boundaries);
    }

    VPUIP::DmaOptimization::EdgeCollection collectMainEdgesWithLoopRegionInfo(mlir::ModuleOp module) {
        auto main = lookupMainFunction(module);
        if (main == nullptr) {
            return {};
        }
        VPUIP::DmaOptimization::LoopRegionBuilder builder(_log);
        const auto boundaries = builder.build(main);
        return collectEdges(main, boundaries);
    }

    static size_t countBoundariesOfKind(llvm::ArrayRef<VPUIP::DmaOptimization::EdgeBoundary> boundaries,
                                        BoundaryKind kind) {
        size_t count = 0;
        for (const auto& boundary : boundaries) {
            if (boundary.kind == kind) {
                ++count;
            }
        }
        return count;
    }

    static size_t countSourcesOfKind(const VPUIP::DmaOptimization::Edge& edge, BoundaryKind kind) {
        return countBoundariesOfKind(edge.sources, kind);
    }

    static size_t countTargetsOfKind(const VPUIP::DmaOptimization::Edge& edge, BoundaryKind kind) {
        return countBoundariesOfKind(edge.targets, kind);
    }

    static size_t countConcatNodes(const VPUIP::DmaOptimization::Edge& edge) {
        size_t count = 0;
        for (const auto& node : edge.nodes) {
            if (node.op != nullptr && mlir::isa<VPUIP::ConcatViewOp>(node.op)) {
                ++count;
            }
        }
        return count;
    }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Normal: 1 source, 1 target. Single Copy lane, FuncArg to FuncReturn.
//
// Diagram:
//   source(FuncArg %arg0)
//          |
//   edgeNode(Copy)
//          |
//   target(FuncReturn operand 0)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, Normal_FuncArgToFuncReturn) {
    constexpr llvm::StringLiteral IR = R"(
        func.func @main(%arg0: memref<1x1x16x16xf16, @DDR>,
                        %dst: memref<1x1x16x16xf16, @DDR>)
                -> memref<1x1x16x16xf16, @DDR> {
            %0 = VPUIP.Copy
                inputs(%arg0 : memref<1x1x16x16xf16, @DDR>)
                outputs(%dst : memref<1x1x16x16xf16, @DDR>)
                -> memref<1x1x16x16xf16, @DDR>
            return %0 : memref<1x1x16x16xf16, @DDR>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    ASSERT_EQ(edges.edges.size(), 1u);

    const auto& edge = edges.edges.front();
    ASSERT_NO_FATAL_FAILURE(expectEdgeShape(edge, EdgeType::Normal, 1u, 1u, 1u));

    EXPECT_EQ(edge.sources[0].kind, BoundaryKind::FuncArg);
    EXPECT_EQ(edge.targets[0].kind, BoundaryKind::FuncReturn);
    ASSERT_NE(edge.targets[0].op, nullptr);
    EXPECT_TRUE(mlir::isa<mlir::func::ReturnOp>(edge.targets[0].op));
    EXPECT_EQ(edge.targets[0].operandIdx, 0u);
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// Write-only uses of an edge result are not data-flow successors. The second
// Copy below uses %copy0 only as its output buffer; the two Copy nodes still
// share one edge through their common source, but %copy0 must not point to
// %copy1 as a node successor.
//
// Diagram:
//             source(FuncArg %arg0)
//                |              |
//   edgeNode(Copy %copy0)  edgeNode(Copy %copy1)
//             |                  |
//   target(FuncReturn 0)   target(FuncReturn 1)
//   %copy0 is also Copy %copy1's output buffer, which is not a data-flow edge.
//
// Note: this might not be a realistic scenario, but it is a valid IR pattern that
// the collector must handle.
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, CopyOutputBufferUse_IsNotDataFlowSuccessor) {
    constexpr llvm::StringLiteral IR = R"(
        func.func @main(%arg0: memref<1x1x16x16xf16, @DDR>,
                        %dst0: memref<1x1x16x16xf16, @DDR>)
                -> (memref<1x1x16x16xf16, @DDR>, memref<1x1x16x16xf16, @DDR>) {
            %copy0 = VPUIP.Copy
                inputs(%arg0 : memref<1x1x16x16xf16, @DDR>)
                outputs(%dst0 : memref<1x1x16x16xf16, @DDR>)
                -> memref<1x1x16x16xf16, @DDR>
            %copy1 = VPUIP.Copy
                inputs(%arg0 : memref<1x1x16x16xf16, @DDR>)
                outputs(%copy0 : memref<1x1x16x16xf16, @DDR>)
                -> memref<1x1x16x16xf16, @DDR>
            return %copy0, %copy1
                : memref<1x1x16x16xf16, @DDR>,
                  memref<1x1x16x16xf16, @DDR>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    ASSERT_EQ(edges.edges.size(), 1u);

    const auto& edge = edges.edges.front();
    ASSERT_EQ(edge.sources.size(), 1u);
    ASSERT_EQ(edge.targets.size(), 2u);
    ASSERT_EQ(edge.nodes.size(), 2u);

    EXPECT_EQ(countTargetsOfKind(edge, BoundaryKind::FuncReturn), 2u);
    bool foundReturnOperand0 = false;
    bool foundReturnOperand1 = false;
    for (const auto& target : edge.targets) {
        EXPECT_EQ(target.kind, BoundaryKind::FuncReturn);
        ASSERT_NE(target.op, nullptr);
        EXPECT_TRUE(mlir::isa<mlir::func::ReturnOp>(target.op));
        if (target.operandIdx == 0) {
            foundReturnOperand0 = true;
        } else if (target.operandIdx == 1) {
            foundReturnOperand1 = true;
        } else {
            ADD_FAILURE() << "Unexpected func.return operand index " << target.operandIdx;
        }
    }
    EXPECT_TRUE(foundReturnOperand0);
    EXPECT_TRUE(foundReturnOperand1);
    ASSERT_EQ(edge.nodes[0].succs.size(), 1u);
    EXPECT_LT(edge.nodes[0].succs.front(), 0);
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// A Copy whose result is never consumed has a source boundary but no target
// boundary. The collector drops such incomplete components.
//
// Diagram:
//   source(FuncArg %arg0)
//          |
//   edgeNode(Copy %copy)
//          |
//   no data-flow consumer
//
//   FuncReturn consumes %arg0 directly, outside this edge.
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, CopyResultWithoutTarget_DropsIncompleteComponent) {
    constexpr llvm::StringLiteral IR = R"(
        func.func @main(%arg0: memref<1x1x16x16xf16, @DDR>,
                        %dst: memref<1x1x16x16xf16, @DDR>)
                -> memref<1x1x16x16xf16, @DDR> {
            %copy = VPUIP.Copy
                inputs(%arg0 : memref<1x1x16x16xf16, @DDR>)
                outputs(%dst : memref<1x1x16x16xf16, @DDR>)
                -> memref<1x1x16x16xf16, @DDR>
            return %arg0 : memref<1x1x16x16xf16, @DDR>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    EXPECT_TRUE(edges.edges.empty());
}

//===----------------------------------------------------------------------===//
// FanOut: 1 source, N targets. A Copy result is consumed by two ComputeOp
// (ConvertDMA) boundaries so the edge has two distinct target endpoints.
//
// Diagram:
//   source(FuncArg %arg0)
//          |
//   edgeNode(Copy %copy)
//        +-> target(ComputeOp ConvertDMA %d0)
//        +-> target(ComputeOp ConvertDMA %d1)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, FanOut_OneSourceTwoComputeTargets) {
    constexpr llvm::StringLiteral IR = R"(
        func.func @main(%arg0: memref<1x1x16x16xf16, @DDR>)
                -> (memref<1x1x16x16xf16, [@CMX_NN, 0]>,
                    memref<1x1x16x16xf16, [@CMX_NN, 0]>) {
            %ddr_buf = memref.alloc() : memref<1x1x16x16xf16, @DDR>
            %copy = VPUIP.Copy
                inputs(%arg0 : memref<1x1x16x16xf16, @DDR>)
                outputs(%ddr_buf : memref<1x1x16x16xf16, @DDR>)
                -> memref<1x1x16x16xf16, @DDR>

            %cmx0 = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
            %d0 = VPUIP.ConvertDMA
                inputs(%copy : memref<1x1x16x16xf16, @DDR>)
                outputs(%cmx0 : memref<1x1x16x16xf16, [@CMX_NN, 0]>)
                -> memref<1x1x16x16xf16, [@CMX_NN, 0]>

            %cmx1 = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
            %d1 = VPUIP.ConvertDMA
                inputs(%copy : memref<1x1x16x16xf16, @DDR>)
                outputs(%cmx1 : memref<1x1x16x16xf16, [@CMX_NN, 0]>)
                -> memref<1x1x16x16xf16, [@CMX_NN, 0]>

            return %d0, %d1
                : memref<1x1x16x16xf16, [@CMX_NN, 0]>,
                  memref<1x1x16x16xf16, [@CMX_NN, 0]>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    ASSERT_EQ(edges.edges.size(), 1u);

    const auto& edge = edges.edges.front();
    // Only the Copy is an EdgeOp; ConvertDMAs are boundaries.
    ASSERT_NO_FATAL_FAILURE(expectEdgeShape(edge, EdgeType::FanOut, 1u, 2u, 1u));

    EXPECT_EQ(edge.sources[0].kind, BoundaryKind::FuncArg);
    EXPECT_EQ(countTargetsOfKind(edge, BoundaryKind::ComputeOp), 2u);
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// FanIn: N sources, 1 target. Two scatter Copies converge into a ConcatView
// junction that returns to the function. The shared output buffer (head_buf)
// also enters the edge as a ConcatView data-in operand.
//
// Diagram:
//                      source(FuncArg %head_buf)
//                         |                  |
//               edgeNode(SubView %sv0) edgeNode(SubView %sv1)
//                         |                  |
//   source(FuncArg %arg0) -> edgeNode(Copy %c0)   edgeNode(Copy %c1) <- source(FuncArg %arg1)
//                                +------------------+
//                                 edgeNode(ConcatView %cv)
//                                           |
//                                  target(FuncReturn)
//   SubViews between %head_buf and the two Copies are edge nodes too.
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, FanIn_TwoSourcesIntoConcatView) {
    constexpr llvm::StringLiteral IR = R"(
        func.func @main(%arg0: memref<1x1x4x4xf16, @DDR>,
                        %arg1: memref<1x1x4x4xf16, @DDR>,
                        %head_buf: memref<1x2x4x4xf16, @DDR>)
                -> memref<1x2x4x4xf16, @DDR> {
            %sv0 = VPUIP.SubView %head_buf [0, 0, 0, 0] [1, 1, 4, 4]
                : memref<1x2x4x4xf16, @DDR>
                to memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>
            %c0 = VPUIP.Copy
                inputs(%arg0 : memref<1x1x4x4xf16, @DDR>)
                outputs(%sv0 : memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>)
                -> memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>

            %sv1 = VPUIP.SubView %head_buf [0, 1, 0, 0] [1, 1, 4, 4]
                : memref<1x2x4x4xf16, @DDR>
                to memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>
            %c1 = VPUIP.Copy
                inputs(%arg1 : memref<1x1x4x4xf16, @DDR>)
                outputs(%sv1 : memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>)
                -> memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>

            %cv = VPUIP.ConcatView
                inputs(%c0, %c1 :
                    memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>,
                    memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>)
                outputs(%head_buf : memref<1x2x4x4xf16, @DDR>)
                -> memref<1x2x4x4xf16, @DDR>

            return %cv : memref<1x2x4x4xf16, @DDR>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    ASSERT_EQ(edges.edges.size(), 1u);

    const auto& edge = edges.edges.front();
    // 3 sources: head_buf from ConcatView output_buff plus arg0 and arg1.
    // 5 nodes: sv0, c0, sv1, c1, cv.
    ASSERT_NO_FATAL_FAILURE(expectEdgeShape(edge, EdgeType::FanIn, 3u, 1u, 5u));

    EXPECT_EQ(countSourcesOfKind(edge, BoundaryKind::FuncArg), 3u);
    EXPECT_EQ(edge.targets[0].kind, BoundaryKind::FuncReturn);
    EXPECT_TRUE(edge.hasConcatNode());
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// Many2Many: N sources, M targets. A ConcatView gathers two scatter Copies
// AND fans out to two terminal Copies. A single junction simultaneously
// collects and distributes.
// The ConcatView result (%cv) is not a source. It connects to the successor
// Copy edge nodes. The ConcatView output buffer operand (%head_buf) is the
// additional source, same as in the FanIn-only case above.
//
// Diagram:
//                      source(FuncArg %head_buf)
//                         |                  |
//               edgeNode(SubView %sv0) edgeNode(SubView %sv1)
//                         |                  |
//   source(FuncArg %arg0) -> edgeNode(Copy %c0)   edgeNode(Copy %c1) <- source(FuncArg %arg1)
//                                +------------------+
//                                 edgeNode(ConcatView %cv)
//                                  |              |
//                 edgeNode(Copy %d0)       edgeNode(Copy %d1)
//                         |                              |
//                 target(FuncReturn 0)           target(FuncReturn 1)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, Many2Many_ConcatJunctionCollectsAndDistributes) {
    constexpr llvm::StringLiteral IR = R"(
        func.func @main(%arg0: memref<1x1x4x4xf16, @DDR>,
                        %arg1: memref<1x1x4x4xf16, @DDR>,
                        %head_buf: memref<1x2x4x4xf16, @DDR>,
                        %dst0: memref<1x2x4x4xf16, @DDR>,
                        %dst1: memref<1x2x4x4xf16, @DDR>)
                -> (memref<1x2x4x4xf16, @DDR>, memref<1x2x4x4xf16, @DDR>) {
            %sv0 = VPUIP.SubView %head_buf [0, 0, 0, 0] [1, 1, 4, 4]
                : memref<1x2x4x4xf16, @DDR>
                to memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>
            %c0 = VPUIP.Copy
                inputs(%arg0 : memref<1x1x4x4xf16, @DDR>)
                outputs(%sv0 : memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>)
                -> memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>

            %sv1 = VPUIP.SubView %head_buf [0, 1, 0, 0] [1, 1, 4, 4]
                : memref<1x2x4x4xf16, @DDR>
                to memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>
            %c1 = VPUIP.Copy
                inputs(%arg1 : memref<1x1x4x4xf16, @DDR>)
                outputs(%sv1 : memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>)
                -> memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>

            %cv = VPUIP.ConcatView
                inputs(%c0, %c1 :
                    memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>,
                    memref<1x1x4x4xf16, {order = affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, strides = [32, 16, 4, 1]}, @DDR>)
                outputs(%head_buf : memref<1x2x4x4xf16, @DDR>)
                -> memref<1x2x4x4xf16, @DDR>

            %d0 = VPUIP.Copy
                inputs(%cv : memref<1x2x4x4xf16, @DDR>)
                outputs(%dst0 : memref<1x2x4x4xf16, @DDR>)
                -> memref<1x2x4x4xf16, @DDR>
            %d1 = VPUIP.Copy
                inputs(%cv : memref<1x2x4x4xf16, @DDR>)
                outputs(%dst1 : memref<1x2x4x4xf16, @DDR>)
                -> memref<1x2x4x4xf16, @DDR>

            return %d0, %d1 : memref<1x2x4x4xf16, @DDR>, memref<1x2x4x4xf16, @DDR>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    ASSERT_EQ(edges.edges.size(), 1u);

    const auto& edge = edges.edges.front();
    ASSERT_NO_FATAL_FAILURE(expectEdgeShape(edge, EdgeType::Many2Many, 3u, 2u, 7u));
    EXPECT_EQ(countSourcesOfKind(edge, BoundaryKind::FuncArg), edge.sources.size());
    EXPECT_EQ(countTargetsOfKind(edge, BoundaryKind::FuncReturn), edge.targets.size());
    EXPECT_TRUE(edge.hasConcatNode());
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// One constant Copy is shared by two compute targets in different tiles
// of the same VF region. The shared Copy remains one FanOut edge node, while the
// target boundaries retain their distinct tile domains.
//
// Diagram:
//   source(Constant weight_table)
//              |
//       edgeNode(Copy)
//          +-> target(NCE tile 0)
//          +-> target(NCE tile 1)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, FanOut_SharedCopyAcrossVfTiles) {
    constexpr llvm::StringLiteral IR = R"(
        #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        !In = memref<1x128x64x4xf16, #NHWC, @CMX_NN>
        !Weights = memref<1024x128x1x1xf16, #NHWC, @CMX_NN>
        !OutBuffer = memref<1x1024x64x4xf16, #NHWC, @CMX_NN>
        !LogicalOut = memref<1x2048x64x4xf16, #NHWC, @CMX_NN>
        !WT = memref<1024x1x1x4xsi32, #NCHW, @CMX_NN>

        func.func @main(%input: !In, %weights: !Weights) -> !LogicalOut {
            %wt = const.Declare memref<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
            %wt_alloc = memref.alloc() : !WT
            %wt_cmx = VPUIP.Copy inputs(%wt : memref<1024x1x1x4xsi32>) outputs(%wt_alloc : !WT) -> !WT
            %out0_alloc = memref.alloc() : !OutBuffer
            %out0 = VPUIP.NCEClusterTask {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>},
                                          vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64} <{
                    resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, is_zero_offset_weights_table,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}>
                input(%input : !In) weights(%weights : !Weights) weight_table(%wt_cmx : !WT)
                parent_input(%input : !In) parent_output(%out0_alloc : !OutBuffer) outputs(%out0_alloc : !OutBuffer) -> !OutBuffer
                variants : {DPUTask {cluster_id = 0 : i64, inEnd = [3, 63, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 63, 1023], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}} PPE : {}
            %out1_alloc = memref.alloc() : !OutBuffer
            %out1 = VPUIP.NCEClusterTask {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>},
                                          vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64} <{
                    resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, is_zero_offset_weights_table,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}>
                input(%input : !In) weights(%weights : !Weights) weight_table(%wt_cmx : !WT)
                parent_input(%input : !In) parent_output(%out1_alloc : !OutBuffer) outputs(%out1_alloc : !OutBuffer) -> !OutBuffer
                variants : {DPUTask {cluster_id = 0 : i64, inEnd = [3, 63, 127], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [3, 63, 1023], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}} PPE : {}
            %concat_buf = memref.alloc() : !LogicalOut
            %concat = VPUIP.ConcatView inputs(%out0, %out1 : !OutBuffer, !OutBuffer) outputs(%concat_buf : !LogicalOut) -> !LogicalOut
            return %concat : !LogicalOut
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);
    auto edges = collectMainEdgesWithLoopRegionInfo(module.get());

    const auto fanOutIt = llvm::find_if(edges.edges, [](const VPUIP::DmaOptimization::Edge& edge) {
        return edge.type == EdgeType::FanOut && edge.sources.size() == 1 &&
               edge.sources.front().kind == BoundaryKind::Constant;
    });
    ASSERT_NE(fanOutIt, edges.edges.end());
    const auto& edge = *fanOutIt;
    ASSERT_NO_FATAL_FAILURE(expectEdgeShape(edge, EdgeType::FanOut, 1u, 2u, 1u));
    EXPECT_EQ(edge.sources[0].kind, BoundaryKind::Constant);
    EXPECT_TRUE(hasTargetLoopDomain(edge, VPUIP::DmaOptimization::LoopRegionKind::VF, 0, 0));
    EXPECT_TRUE(hasTargetLoopDomain(edge, VPUIP::DmaOptimization::LoopRegionKind::VF, 0, 1));
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// Two sibling Copies read one external value and feed compute targets in
// different tiles of the same VF region. Shared-source component merging makes
// this one FanOut edge with two Copy nodes and two tile-aware targets.
//
// Diagram:
//                      source(FuncArg shared_value)
//                         +-> edgeNode(Copy 0) -> target(SW tile 0)
//                         +-> edgeNode(Copy 1) -> target(SW tile 1)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, FanOut_SiblingCopiesAcrossVfTiles) {
    constexpr llvm::StringLiteral IR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        func.func @main(%activation0: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                        %activation1: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                        %weight_table_ddr: memref<64x1x1x4xsi32, #NCHW, @DDR>)
            -> memref<1x1x1x128xf16, #NCHW, [@CMX_NN, 0]> {
            %wt_cmx0 = memref.alloc() : memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>
            %wt_copy0 = VPUIP.Copy inputs(%weight_table_ddr : memref<64x1x1x4xsi32, #NCHW, @DDR>)
                                  outputs(%wt_cmx0 : memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>)
                         -> memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>
            %wt_cmx1 = memref.alloc() : memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>
            %wt_copy1 = VPUIP.Copy inputs(%weight_table_ddr : memref<64x1x1x4xsi32, #NCHW, @DDR>)
                                  outputs(%wt_cmx1 : memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>)
                         -> memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>

            %out0 = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %consumer0 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>, vf_loop_index = 0 : i64, vf_loop_tile_index = 0 : i64} @VPU.SW::@builtin_Gather
                inputs(%activation0 as %act0: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>, %wt_copy0 as %table0: memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>)
                outputs(%out0 as %res0: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                on tile 0 -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]> {
                    VPUIP.SW.Kernel.run {attrs = [1, 0]}(%act0, %table0, %res0) : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>, memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
                }
            %out1 = memref.alloc() : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
            %consumer1 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>, vf_loop_index = 0 : i64, vf_loop_tile_index = 1 : i64} @VPU.SW::@builtin_Gather
                inputs(%activation1 as %act1: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>, %wt_copy1 as %table1: memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>)
                outputs(%out1 as %res1: memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                on tile 0 -> memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]> {
                    VPUIP.SW.Kernel.run {attrs = [1, 0]}(%act1, %table1, %res1) : memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>, memref<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0]>, memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>
                }
            %concat_buf = memref.alloc() : memref<1x1x1x128xf16, #NCHW, [@CMX_NN, 0]>
            %concat = VPUIP.ConcatView inputs(%consumer0, %consumer1 :
                    memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>,
                    memref<1x1x1x64xf16, #NCHW, [@CMX_NN, 0]>)
                outputs(%concat_buf : memref<1x1x1x128xf16, #NCHW, [@CMX_NN, 0]>)
                -> memref<1x1x1x128xf16, #NCHW, [@CMX_NN, 0]>
            return %concat : memref<1x1x1x128xf16, #NCHW, [@CMX_NN, 0]>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);
    auto edges = collectMainEdgesWithLoopRegionInfo(module.get());

    const auto fanOutIt = llvm::find_if(edges.edges, [](const VPUIP::DmaOptimization::Edge& edge) {
        return edge.type == EdgeType::FanOut && edge.sources.size() == 1 &&
               edge.sources.front().kind == BoundaryKind::FuncArg;
    });
    ASSERT_NE(fanOutIt, edges.edges.end());
    const auto& edge = *fanOutIt;
    ASSERT_NO_FATAL_FAILURE(expectEdgeShape(edge, EdgeType::FanOut, 1u, 2u, 2u));
    EXPECT_EQ(edge.sources[0].kind, BoundaryKind::FuncArg);
    expectLoopDomain(edge.targets[0].loopRegionInfo, VPUIP::DmaOptimization::LoopRegionKind::VF, 0, 0);
    expectLoopDomain(edge.targets[1].loopRegionInfo, VPUIP::DmaOptimization::LoopRegionKind::VF, 0, 1);
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// Two VF producers write ConcatView slots, then two slices of the assembled
// buffer are copied to VF consumers. The result is a Many2Many edge whose
// source and target boundaries preserve their producer/consumer tile domains.
//
// Diagram:
//   source(Compute tile 5) -> edgeNode(Copy 0) --+
//   source(Compute tile 5) -> edgeNode(Copy 1) --+-> edgeNode(ConcatView)
//   source(Alloc concat_buffer) -> edgeNode(SubViews)    |          |
//                                                   SubView 0   SubView 1
//                                                       |          |
//                                                    Copy 0     Copy 1
//                                                       |          |
//                                             target(tile 5) target(tile 6)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, ManyToMany_VfTilePairingThroughConcat) {
    constexpr llvm::StringLiteral IR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        func.func @main(%input0: memref<1x4x1x1xf16, #NCHW, @CMX_NN>,
                        %input1: memref<1x4x1x1xf16, #NCHW, @CMX_NN>)
            -> memref<1x8x1x1xf16, #NCHW, @CMX_NN> {
            %src0_buf = memref.alloc() : memref<1x4x1x1xf16, #NCHW, @CMX_NN>
            %src0 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>, vf_loop_index = 8 : i64, vf_loop_tile_index = 5 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%input0 as %src0_input: memref<1x4x1x1xf16, #NCHW, @CMX_NN>) outputs(%src0_buf as %src0_output: memref<1x4x1x1xf16, #NCHW, @CMX_NN>)
                on tile 0 -> memref<1x4x1x1xf16, #NCHW, @CMX_NN> { VPUIP.SW.Kernel.run {attrs = [0, 0]}(%src0_input, %src0_output) : memref<1x4x1x1xf16, #NCHW, @CMX_NN>, memref<1x4x1x1xf16, #NCHW, @CMX_NN> }
            %src1_buf = memref.alloc() : memref<1x4x1x1xf16, #NCHW, @CMX_NN>
            %src1 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>, vf_loop_index = 8 : i64, vf_loop_tile_index = 5 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%input1 as %src1_input: memref<1x4x1x1xf16, #NCHW, @CMX_NN>) outputs(%src1_buf as %src1_output: memref<1x4x1x1xf16, #NCHW, @CMX_NN>)
                on tile 0 -> memref<1x4x1x1xf16, #NCHW, @CMX_NN> { VPUIP.SW.Kernel.run {attrs = [0, 0]}(%src1_input, %src1_output) : memref<1x4x1x1xf16, #NCHW, @CMX_NN>, memref<1x4x1x1xf16, #NCHW, @CMX_NN> }

            %concat_buf = memref.alloc() : memref<1x8x1x1xf16, #NCHW, @DDR>
            %sv0 = VPUIP.SubView %concat_buf [0, 0, 0, 0] [1, 4, 1, 1]
                : memref<1x8x1x1xf16, #NCHW, @DDR> to memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>
            %sv1 = VPUIP.SubView %concat_buf [0, 4, 0, 0] [1, 4, 1, 1]
                : memref<1x8x1x1xf16, #NCHW, @DDR> to memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>
            %copy0 = VPUIP.Copy inputs(%src0 : memref<1x4x1x1xf16, #NCHW, @CMX_NN>) outputs(%sv0 : memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>) -> memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>
            %copy1 = VPUIP.Copy inputs(%src1 : memref<1x4x1x1xf16, #NCHW, @CMX_NN>) outputs(%sv1 : memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>) -> memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>
            %concat = VPUIP.ConcatView inputs(%copy0, %copy1 : memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>, memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>) outputs(%concat_buf : memref<1x8x1x1xf16, #NCHW, @DDR>) -> memref<1x8x1x1xf16, #NCHW, @DDR>
            %out_slot0 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 4, 1, 1] : memref<1x8x1x1xf16, #NCHW, @DDR> to memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>
            %out_slot1 = VPUIP.SubView %concat [0, 4, 0, 0] [1, 4, 1, 1] : memref<1x8x1x1xf16, #NCHW, @DDR> to memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>
            %consumer0_in = memref.alloc() : memref<1x4x1x1xf16, #NCHW, @CMX_NN>
            %out_copy0 = VPUIP.Copy inputs(%out_slot0 : memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>) outputs(%consumer0_in : memref<1x4x1x1xf16, #NCHW, @CMX_NN>) -> memref<1x4x1x1xf16, #NCHW, @CMX_NN>
            %consumer1_in = memref.alloc() : memref<1x4x1x1xf16, #NCHW, @CMX_NN>
            %out_copy1 = VPUIP.Copy inputs(%out_slot1 : memref<1x4x1x1xf16, {order = #NCHW, strides = [8, 1, 1, 1]}, @DDR>) outputs(%consumer1_in : memref<1x4x1x1xf16, #NCHW, @CMX_NN>) -> memref<1x4x1x1xf16, #NCHW, @CMX_NN>

            %consumer0_buf = memref.alloc() : memref<1x4x1x1xf16, #NCHW, @CMX_NN>
            %consumer0 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>, vf_loop_index = 8 : i64, vf_loop_tile_index = 5 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%out_copy0 as %consumer0_input: memref<1x4x1x1xf16, #NCHW, @CMX_NN>) outputs(%consumer0_buf as %consumer0_output: memref<1x4x1x1xf16, #NCHW, @CMX_NN>)
                on tile 0 -> memref<1x4x1x1xf16, #NCHW, @CMX_NN> { VPUIP.SW.Kernel.run {attrs = [0, 0]}(%consumer0_input, %consumer0_output) : memref<1x4x1x1xf16, #NCHW, @CMX_NN>, memref<1x4x1x1xf16, #NCHW, @CMX_NN> }
            %consumer1_buf = memref.alloc() : memref<1x4x1x1xf16, #NCHW, @CMX_NN>
            %consumer1 = VPUIP.SW.Kernel {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, resultSegmentSizes = array<i32: 1, 0, 0>, vf_loop_index = 8 : i64, vf_loop_tile_index = 6 : i64} @VPU.SW::@builtin_LogSoftmax
                inputs(%out_copy1 as %consumer1_input: memref<1x4x1x1xf16, #NCHW, @CMX_NN>) outputs(%consumer1_buf as %consumer1_output: memref<1x4x1x1xf16, #NCHW, @CMX_NN>)
                on tile 0 -> memref<1x4x1x1xf16, #NCHW, @CMX_NN> { VPUIP.SW.Kernel.run {attrs = [0, 0]}(%consumer1_input, %consumer1_output) : memref<1x4x1x1xf16, #NCHW, @CMX_NN>, memref<1x4x1x1xf16, #NCHW, @CMX_NN> }

            %consumer_concat_buf = memref.alloc() : memref<1x8x1x1xf16, #NCHW, @CMX_NN>
            %consumer_concat = VPUIP.ConcatView inputs(%consumer0, %consumer1 :
                    memref<1x4x1x1xf16, #NCHW, @CMX_NN>,
                    memref<1x4x1x1xf16, #NCHW, @CMX_NN>)
                outputs(%consumer_concat_buf : memref<1x8x1x1xf16, #NCHW, @CMX_NN>)
                -> memref<1x8x1x1xf16, #NCHW, @CMX_NN>
            return %consumer_concat : memref<1x8x1x1xf16, #NCHW, @CMX_NN>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);
    auto edges = collectMainEdgesWithLoopRegionInfo(module.get());

    const auto manyIt = llvm::find_if(edges.edges, [](const VPUIP::DmaOptimization::Edge& edge) {
        return edge.type == EdgeType::Many2Many;
    });
    ASSERT_NE(manyIt, edges.edges.end());
    EXPECT_TRUE(manyIt->hasConcatNode());
    const auto concatNodeIt = llvm::find_if(manyIt->nodes, [](const VPUIP::DmaOptimization::EdgeNode& node) {
        return mlir::isa<VPUIP::ConcatViewOp>(node.op);
    });
    ASSERT_NE(concatNodeIt, manyIt->nodes.end());
    EXPECT_FALSE(concatNodeIt->loopRegionInfo.hasPort &&
                 concatNodeIt->loopRegionInfo.portKind == VPUIP::DmaOptimization::LoopRegionPortKind::Output);
    EXPECT_TRUE(llvm::any_of(manyIt->sources, [](const VPUIP::DmaOptimization::EdgeBoundary& source) {
        return source.loopRegionInfo.kind == VPUIP::DmaOptimization::LoopRegionKind::VF &&
               source.loopRegionInfo.loopIndex == 8 && source.loopRegionInfo.tileIndex == 5;
    }));
    EXPECT_TRUE(llvm::any_of(manyIt->targets, [](const VPUIP::DmaOptimization::EdgeBoundary& target) {
        return target.loopRegionInfo.kind == VPUIP::DmaOptimization::LoopRegionKind::VF &&
               target.loopRegionInfo.loopIndex == 8 && target.loopRegionInfo.tileIndex == 6;
    }));
    EXPECT_TRUE(manyIt->isLoopRegion());
}

//===----------------------------------------------------------------------===//
// Boundary-only IR: ConvertDMA is a ComputeOp boundary, not an EdgeOp. With no
// Copy/SubView/ConcatView/view-like node to collect, the collector emits no
// edge even though the ConvertDMA reads an allocated buffer.
//
// Diagram:
//   memref.alloc %ddr_buf
//          |
//   target(ComputeOp ConvertDMA)
//          |
//   FuncReturn
//   No edgeNode exists, so no Edge is collected.
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, BoundaryOnlyConvertDMA_ProducesNoEdge) {
    constexpr llvm::StringLiteral IR = R"(
        func.func @main()
                -> memref<1x1x16x16xf16, [@CMX_NN, 0]> {
            %ddr_buf = memref.alloc() : memref<1x1x16x16xf16, @DDR>
            %cmx_buf = memref.alloc() : memref<1x1x16x16xf16, [@CMX_NN, 0]>
            %d0 = VPUIP.ConvertDMA
                inputs(%ddr_buf : memref<1x1x16x16xf16, @DDR>)
                outputs(%cmx_buf : memref<1x1x16x16xf16, [@CMX_NN, 0]>)
                -> memref<1x1x16x16xf16, [@CMX_NN, 0]>
            return %d0 : memref<1x1x16x16xf16, [@CMX_NN, 0]>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    // No EdgeOps (Copy / SubView / ConcatView / view-like) in this IR. The
    // ConvertDMA is a boundary. The collector therefore produces no edges.
    EXPECT_TRUE(edges.edges.empty());
}

//===----------------------------------------------------------------------===//
// Many2Many: central junction with N sources and M targets.
//
// This reduced IR keeps the shape of a head-gathering region:
//   * 3 producers, reduced to ConvertDMA ComputeOp boundaries for parser simplicity.
//   * 3 strided Copies scatter into a central 1x1x1024x1536 buffer.
//   * 1 central ConcatView (the "D" junction) merges the 3 tiles.
//   * 3 SubView+Copy chains fan out along the channel dim (Q/K/V split).
//   * 3 ConvertDMA ComputeOp consumers terminate the edge.
//
// EdgeCollector contract being exercised:
//   * Every EdgeOp on the data path becomes a node (3 sv_in + 3 c_in + 1
//     concat + 3 sv_out + 3 c_out = 13 nodes).
//   * `type == Many2Many` because both source and target counts are >= 2.
//   * `hasConcatNode()` is true (central ConcatView D).
//   * `validate` holds (sentinel ranges, entry/exit indices consistent).
//
// The edge's `sources` array currently contains both the 3 ComputeOp producers
// and the AllocBuffer feeding the central ConcatView output buffer.
//
// Diagram:
//                         source(AllocBuffer center)
//                         |           |             |
//          edgeNode(SubView in0) edgeNode(SubView in1) edgeNode(SubView in2)
//                   |                   |                   |
//   source(ComputeOp prod0) -> edgeNode(Copy c0)
//   source(ComputeOp prod1) -> edgeNode(Copy c1)
//   source(ComputeOp prod2) -> edgeNode(Copy c2)
//                   +-------------------+-------------------+
//                         edgeNode(ConcatView)
//                         |           |             |
//         edgeNode(SubView out0) edgeNode(SubView out1) edgeNode(SubView out2)
//                   |                   |                   |
//          edgeNode(Copy co0)   edgeNode(Copy co1)   edgeNode(Copy co2)
//                   |                   |                   |
//   target(ComputeOp cons0) target(ComputeOp cons1) target(ComputeOp cons2)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, Many2Many_CentralHeadGatherJunction) {
    constexpr llvm::StringLiteral IR = R"(
        #strided_in = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        func.func @main(
            %src_cmx0: memref<1x1x342x1536xf16, [@CMX_NN, 0]>,
            %src_cmx1: memref<1x1x341x1536xf16, [@CMX_NN, 0]>,
            %src_cmx2: memref<1x1x341x1536xf16, [@CMX_NN, 0]>,
            %dst_cmx0: memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
            %dst_cmx1: memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
            %dst_cmx2: memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                -> (memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                    memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                    memref<1x1x1024x512xf16, [@CMX_NN, 0]>) {

            // ---------- Producer side: 3 ComputeOp (ConvertDMA) -> DDR tiles ----------
            %prod_buf0 = memref.alloc() : memref<1x1x342x1536xf16, @DDR>
            %prod0 = VPUIP.ConvertDMA
                inputs(%src_cmx0 : memref<1x1x342x1536xf16, [@CMX_NN, 0]>)
                outputs(%prod_buf0 : memref<1x1x342x1536xf16, @DDR>)
                -> memref<1x1x342x1536xf16, @DDR>

            %prod_buf1 = memref.alloc() : memref<1x1x341x1536xf16, @DDR>
            %prod1 = VPUIP.ConvertDMA
                inputs(%src_cmx1 : memref<1x1x341x1536xf16, [@CMX_NN, 0]>)
                outputs(%prod_buf1 : memref<1x1x341x1536xf16, @DDR>)
                -> memref<1x1x341x1536xf16, @DDR>

            %prod_buf2 = memref.alloc() : memref<1x1x341x1536xf16, @DDR>
            %prod2 = VPUIP.ConvertDMA
                inputs(%src_cmx2 : memref<1x1x341x1536xf16, [@CMX_NN, 0]>)
                outputs(%prod_buf2 : memref<1x1x341x1536xf16, @DDR>)
                -> memref<1x1x341x1536xf16, @DDR>

            // ---------- Central scatter buffer + 3 strided Copies ----------
            %center = memref.alloc() : memref<1x1x1024x1536xf16, @DDR>

            %sv_in0 = VPUIP.SubView %center [0, 0, 0, 0] [1, 1, 342, 1536]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x342x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %c0 = VPUIP.Copy
                inputs(%prod0 : memref<1x1x342x1536xf16, @DDR>)
                outputs(%sv_in0 : memref<1x1x342x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                -> memref<1x1x342x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>

            %sv_in1 = VPUIP.SubView %center [0, 0, 342, 0] [1, 1, 341, 1536]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %c1 = VPUIP.Copy
                inputs(%prod1 : memref<1x1x341x1536xf16, @DDR>)
                outputs(%sv_in1 : memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                -> memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>

            %sv_in2 = VPUIP.SubView %center [0, 0, 683, 0] [1, 1, 341, 1536]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %c2 = VPUIP.Copy
                inputs(%prod2 : memref<1x1x341x1536xf16, @DDR>)
                outputs(%sv_in2 : memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                -> memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>

            // ---------- Central ConcatView (junction "D") ----------
            %concat = VPUIP.ConcatView
                inputs(%c0, %c1, %c2 :
                    memref<1x1x342x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>,
                    memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>,
                    memref<1x1x341x1536xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                outputs(%center : memref<1x1x1024x1536xf16, @DDR>)
                -> memref<1x1x1024x1536xf16, @DDR>

            // ---------- Fan-out: 3 SubView+Copy along channel dim ----------
            %sv_out0 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 1, 1024, 512]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x1024x512xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %out_buf0 = memref.alloc() : memref<1x1x1024x512xf16, @DDR>
            %co0 = VPUIP.Copy
                inputs(%sv_out0 : memref<1x1x1024x512xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                outputs(%out_buf0 : memref<1x1x1024x512xf16, @DDR>)
                -> memref<1x1x1024x512xf16, @DDR>
            %cons0 = VPUIP.ConvertDMA
                inputs(%co0 : memref<1x1x1024x512xf16, @DDR>)
                outputs(%dst_cmx0 : memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                -> memref<1x1x1024x512xf16, [@CMX_NN, 0]>

            %sv_out1 = VPUIP.SubView %concat [0, 0, 0, 512] [1, 1, 1024, 512]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x1024x512xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %out_buf1 = memref.alloc() : memref<1x1x1024x512xf16, @DDR>
            %co1 = VPUIP.Copy
                inputs(%sv_out1 : memref<1x1x1024x512xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                outputs(%out_buf1 : memref<1x1x1024x512xf16, @DDR>)
                -> memref<1x1x1024x512xf16, @DDR>
            %cons1 = VPUIP.ConvertDMA
                inputs(%co1 : memref<1x1x1024x512xf16, @DDR>)
                outputs(%dst_cmx1 : memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                -> memref<1x1x1024x512xf16, [@CMX_NN, 0]>

            %sv_out2 = VPUIP.SubView %concat [0, 0, 0, 1024] [1, 1, 1024, 512]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x1024x512xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %out_buf2 = memref.alloc() : memref<1x1x1024x512xf16, @DDR>
            %co2 = VPUIP.Copy
                inputs(%sv_out2 : memref<1x1x1024x512xf16, {order = #strided_in, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                outputs(%out_buf2 : memref<1x1x1024x512xf16, @DDR>)
                -> memref<1x1x1024x512xf16, @DDR>
            %cons2 = VPUIP.ConvertDMA
                inputs(%co2 : memref<1x1x1024x512xf16, @DDR>)
                outputs(%dst_cmx2 : memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                -> memref<1x1x1024x512xf16, [@CMX_NN, 0]>

            return %cons0, %cons1, %cons2
                : memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                  memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                  memref<1x1x1024x512xf16, [@CMX_NN, 0]>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    // Single fully connected edge: every EdgeOp is reachable from every
    // other through the central ConcatView.
    ASSERT_EQ(edges.edges.size(), 1u);

    const auto& edge = edges.edges.front();

    // Topology: 3 producers, 3 consumers, plus the central buffer alloc in `sources`.
    EXPECT_EQ(edge.type, EdgeType::Many2Many);

    // Targets: exactly 3 ComputeOp consumers (the 3 ConvertDMA terminals).
    ASSERT_EQ(edge.targets.size(), 3u);
    EXPECT_EQ(countTargetsOfKind(edge, BoundaryKind::ComputeOp), 3u);

    // Sources: 3 ComputeOp producers + 1 AllocBuffer (central scatter buffer
    // pulled in via ConcatView output_buff).
    EXPECT_EQ(countSourcesOfKind(edge, BoundaryKind::ComputeOp), 3u);
    EXPECT_GE(countSourcesOfKind(edge, BoundaryKind::AllocBuffer), 1u);

    // Node count: 3 SubView_in + 3 Copy_in + 1 ConcatView + 3 SubView_out + 3 Copy_out.
    EXPECT_EQ(edge.nodes.size(), 13u);

    // Central ConcatView present and DAG invariants hold.
    EXPECT_TRUE(edge.hasConcatNode());
    EXPECT_TRUE(edge.validate(_log));
}

//===----------------------------------------------------------------------===//
// Many2Many: four ConcatView junctions with one transparent target branch.
//
// Producer side (N = 6 ComputeOp results = 3 pairs):
//   3 ConvertDMA pairs replace the 3 RMS SwKernel multi-result producers
//   in the original IR. Each "pair" stands for the two result lanes of one
//   SwKernel; the edge collector treats each (op, resultIdx) as a
//   distinct source boundary, so the semantics match.
//
// Internal junctions (3 small + 1 central):
//   pair0_a + pair0_b -> Concat_A (342) --Copy--+
//   pair1_a + pair1_b -> Concat_B (341) --Copy--+--> Concat_central (1024)
//   pair2_a + pair2_b -> Concat_C (341) --Copy--+
//
// Consumer side (M = 4 ComputeOp targets):
//   * GenericReshape -> PermuteCast -> MatMul_NCE (transparent walk)
//   * SubView [.., 0]    -> Copy -> Q-proj  (ConvertDMA terminal)
//   * SubView [.., 512]  -> Copy -> K-proj
//   * SubView [.., 1024] -> Copy -> V-proj
//
// EdgeCollector contract being exercised here:
//   * One edge: every EdgeOp is reachable through the central ConcatView.
//   * type == Many2Many.
//   * Exactly 4 distinct ComputeOp targets.
//   * 6 distinct ComputeOp sources (one per producer pair lane).
//   * AllocBuffer boundaries are observable on the source side.
//   * All four ConcatView nodes appear via `hasConcatNode()`.
//   * `validate` holds.
//
// Diagram:
//   source(AllocBuffer bufA) -> edgeNode(SubView A0) -> edgeNode(Copy cA0) <- source(ComputeOp p0a)
//                             -> edgeNode(SubView A1) -> edgeNode(Copy cA1) <- source(ComputeOp p0b)
//                                                        \       /
//                                                     edgeNode(ConcatView A)
//                                                               |
//                                                     edgeNode(SubView D0)
//                                                               |
//                                                     edgeNode(Copy cD0)
//
//   source(AllocBuffer bufB) -> edgeNode(SubView B0) -> edgeNode(Copy cB0) <- source(ComputeOp p1a)
//                             -> edgeNode(SubView B1) -> edgeNode(Copy cB1) <- source(ComputeOp p1b)
//                                                        \       /
//                                                     edgeNode(ConcatView B)
//                                                               |
//                                                     edgeNode(SubView D1)
//                                                               |
//                                                     edgeNode(Copy cD1)
//
//   source(AllocBuffer bufC) -> edgeNode(SubView C0) -> edgeNode(Copy cC0) <- source(ComputeOp p2a)
//                             -> edgeNode(SubView C1) -> edgeNode(Copy cC1) <- source(ComputeOp p2b)
//                                                        \       /
//                                                     edgeNode(ConcatView C)
//                                                               |
//                                                     edgeNode(SubView D2)
//                                                               |
//                                                     edgeNode(Copy cD2)
//
//   source(AllocBuffer bufD) also feeds D0/D1/D2 and ConcatView D output_buff.
//   Copy cD0/cD1/cD2 results feed ConcatView D.
//                             edgeNode(ConcatView D)
//                             |        |           |          |
//          edgeNode(GenericReshape) edgeNode(SubView Q) edgeNode(SubView K) edgeNode(SubView V)
//                    |                    |                 |                 |
//          target(ComputeOp mm)   edgeNode(Copy cQ)  edgeNode(Copy cK)  edgeNode(Copy cV)
//                                      |                 |                 |
//                         target(ComputeOp qOut) target(ComputeOp kOut) target(ComputeOp vOut)
//===----------------------------------------------------------------------===//
TEST_F(MLIR_EdgeCollector, Many2Many_FourJunctions) {
    constexpr llvm::StringLiteral IR = R"(
        #strided = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        func.func @main(
            %src0a: memref<1x1x171x1536xf16, [@CMX_NN, 0]>,
            %src0b: memref<1x1x171x1536xf16, [@CMX_NN, 0]>,
            %src1a: memref<1x1x171x1536xf16, [@CMX_NN, 0]>,
            %src1b: memref<1x1x170x1536xf16, [@CMX_NN, 0]>,
            %src2a: memref<1x1x171x1536xf16, [@CMX_NN, 0]>,
            %src2b: memref<1x1x170x1536xf16, [@CMX_NN, 0]>,
            %dst_mm: memref<1024x1536x1x1xf16, [@CMX_NN, 0]>,
            %dst_q:  memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
            %dst_k:  memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
            %dst_v:  memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                -> (memref<1024x1536x1x1xf16, [@CMX_NN, 0]>,
                    memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                    memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                    memref<1x1x1024x512xf16, [@CMX_NN, 0]>) {

            // ---------- 6 producer ComputeOps (3 RMS-pair lanes) ----------
            %pb0a = memref.alloc() : memref<1x1x171x1536xf16, @DDR>
            %p0a  = VPUIP.ConvertDMA inputs(%src0a : memref<1x1x171x1536xf16, [@CMX_NN, 0]>)
                                     outputs(%pb0a : memref<1x1x171x1536xf16, @DDR>)
                                     -> memref<1x1x171x1536xf16, @DDR>
            %pb0b = memref.alloc() : memref<1x1x171x1536xf16, @DDR>
            %p0b  = VPUIP.ConvertDMA inputs(%src0b : memref<1x1x171x1536xf16, [@CMX_NN, 0]>)
                                     outputs(%pb0b : memref<1x1x171x1536xf16, @DDR>)
                                     -> memref<1x1x171x1536xf16, @DDR>

            %pb1a = memref.alloc() : memref<1x1x171x1536xf16, @DDR>
            %p1a  = VPUIP.ConvertDMA inputs(%src1a : memref<1x1x171x1536xf16, [@CMX_NN, 0]>)
                                     outputs(%pb1a : memref<1x1x171x1536xf16, @DDR>)
                                     -> memref<1x1x171x1536xf16, @DDR>
            %pb1b = memref.alloc() : memref<1x1x170x1536xf16, @DDR>
            %p1b  = VPUIP.ConvertDMA inputs(%src1b : memref<1x1x170x1536xf16, [@CMX_NN, 0]>)
                                     outputs(%pb1b : memref<1x1x170x1536xf16, @DDR>)
                                     -> memref<1x1x170x1536xf16, @DDR>

            %pb2a = memref.alloc() : memref<1x1x171x1536xf16, @DDR>
            %p2a  = VPUIP.ConvertDMA inputs(%src2a : memref<1x1x171x1536xf16, [@CMX_NN, 0]>)
                                     outputs(%pb2a : memref<1x1x171x1536xf16, @DDR>)
                                     -> memref<1x1x171x1536xf16, @DDR>
            %pb2b = memref.alloc() : memref<1x1x170x1536xf16, @DDR>
            %p2b  = VPUIP.ConvertDMA inputs(%src2b : memref<1x1x170x1536xf16, [@CMX_NN, 0]>)
                                     outputs(%pb2b : memref<1x1x170x1536xf16, @DDR>)
                                     -> memref<1x1x170x1536xf16, @DDR>

            // ---------- Junction A: tile 0 (342 = 171+171) ----------
            %bufA = memref.alloc() : memref<1x1x342x1536xf16, @DDR>
            %svA0 = VPUIP.SubView %bufA [0, 0,   0, 0] [1, 1, 171, 1536]
                : memref<1x1x342x1536xf16, @DDR>
               to memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>
            %cA0  = VPUIP.Copy inputs(%p0a  : memref<1x1x171x1536xf16, @DDR>)
                               outputs(%svA0 : memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>)
                               -> memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>
            %svA1 = VPUIP.SubView %bufA [0, 0, 171, 0] [1, 1, 171, 1536]
                : memref<1x1x342x1536xf16, @DDR>
               to memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>
            %cA1  = VPUIP.Copy inputs(%p0b  : memref<1x1x171x1536xf16, @DDR>)
                               outputs(%svA1 : memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>)
                               -> memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>
            %concatA = VPUIP.ConcatView
                inputs(%cA0, %cA1 :
                    memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>,
                    memref<1x1x171x1536xf16, {order = #strided, strides = [525312, 525312, 1536, 1]}, @DDR>)
                outputs(%bufA : memref<1x1x342x1536xf16, @DDR>)
                -> memref<1x1x342x1536xf16, @DDR>

            // ---------- Junction B: tile 1 (341 = 171+170) ----------
            %bufB = memref.alloc() : memref<1x1x341x1536xf16, @DDR>
            %svB0 = VPUIP.SubView %bufB [0, 0,   0, 0] [1, 1, 171, 1536]
                : memref<1x1x341x1536xf16, @DDR>
               to memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %cB0  = VPUIP.Copy inputs(%p1a  : memref<1x1x171x1536xf16, @DDR>)
                               outputs(%svB0 : memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>)
                               -> memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %svB1 = VPUIP.SubView %bufB [0, 0, 171, 0] [1, 1, 170, 1536]
                : memref<1x1x341x1536xf16, @DDR>
               to memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %cB1  = VPUIP.Copy inputs(%p1b  : memref<1x1x170x1536xf16, @DDR>)
                               outputs(%svB1 : memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>)
                               -> memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %concatB = VPUIP.ConcatView
                inputs(%cB0, %cB1 :
                    memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>,
                    memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>)
                outputs(%bufB : memref<1x1x341x1536xf16, @DDR>)
                -> memref<1x1x341x1536xf16, @DDR>

            // ---------- Junction C: tile 2 (341 = 171+170) ----------
            %bufC = memref.alloc() : memref<1x1x341x1536xf16, @DDR>
            %svC0 = VPUIP.SubView %bufC [0, 0,   0, 0] [1, 1, 171, 1536]
                : memref<1x1x341x1536xf16, @DDR>
               to memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %cC0  = VPUIP.Copy inputs(%p2a  : memref<1x1x171x1536xf16, @DDR>)
                               outputs(%svC0 : memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>)
                               -> memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %svC1 = VPUIP.SubView %bufC [0, 0, 171, 0] [1, 1, 170, 1536]
                : memref<1x1x341x1536xf16, @DDR>
               to memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %cC1  = VPUIP.Copy inputs(%p2b  : memref<1x1x170x1536xf16, @DDR>)
                               outputs(%svC1 : memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>)
                               -> memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>
            %concatC = VPUIP.ConcatView
                inputs(%cC0, %cC1 :
                    memref<1x1x171x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>,
                    memref<1x1x170x1536xf16, {order = #strided, strides = [523776, 523776, 1536, 1]}, @DDR>)
                outputs(%bufC : memref<1x1x341x1536xf16, @DDR>)
                -> memref<1x1x341x1536xf16, @DDR>

            // ---------- Junction D: central scatter (1024 = 342+341+341) ----------
            %bufD = memref.alloc() : memref<1x1x1024x1536xf16, @DDR>

            %svD0 = VPUIP.SubView %bufD [0, 0,   0, 0] [1, 1, 342, 1536]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x342x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %cD0  = VPUIP.Copy inputs(%concatA : memref<1x1x342x1536xf16, @DDR>)
                               outputs(%svD0 : memref<1x1x342x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                               -> memref<1x1x342x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>

            %svD1 = VPUIP.SubView %bufD [0, 0, 342, 0] [1, 1, 341, 1536]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %cD1  = VPUIP.Copy inputs(%concatB : memref<1x1x341x1536xf16, @DDR>)
                               outputs(%svD1 : memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                               -> memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>

            %svD2 = VPUIP.SubView %bufD [0, 0, 683, 0] [1, 1, 341, 1536]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %cD2  = VPUIP.Copy inputs(%concatC : memref<1x1x341x1536xf16, @DDR>)
                               outputs(%svD2 : memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                               -> memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>

            %concatD = VPUIP.ConcatView
                inputs(%cD0, %cD1, %cD2 :
                    memref<1x1x342x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>,
                    memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>,
                    memref<1x1x341x1536xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                outputs(%bufD : memref<1x1x1024x1536xf16, @DDR>)
                -> memref<1x1x1024x1536xf16, @DDR>

            // ---------- Target 1: MatMul branch (transparent reshape chain) ----------
            %mm_in = VPUIP.GenericReshape inputs(%concatD : memref<1x1x1024x1536xf16, @DDR>)
                                          -> memref<1024x1536x1x1xf16, @DDR>
            %mm = VPUIP.ConvertDMA inputs(%mm_in : memref<1024x1536x1x1xf16, @DDR>)
                                   outputs(%dst_mm : memref<1024x1536x1x1xf16, [@CMX_NN, 0]>)
                                   -> memref<1024x1536x1x1xf16, [@CMX_NN, 0]>

            // ---------- Targets 2/3/4: SubView+Copy to Q/K/V ----------
            %svQ = VPUIP.SubView %concatD [0, 0, 0,    0] [1, 1, 1024, 512]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x1024x512xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %qbuf = memref.alloc() : memref<1x1x1024x512xf16, @DDR>
            %cQ  = VPUIP.Copy inputs(%svQ : memref<1x1x1024x512xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                              outputs(%qbuf : memref<1x1x1024x512xf16, @DDR>)
                              -> memref<1x1x1024x512xf16, @DDR>
            %qOut = VPUIP.ConvertDMA inputs(%cQ : memref<1x1x1024x512xf16, @DDR>)
                                     outputs(%dst_q : memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                                     -> memref<1x1x1024x512xf16, [@CMX_NN, 0]>

            %svK = VPUIP.SubView %concatD [0, 0, 0,  512] [1, 1, 1024, 512]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x1024x512xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %kbuf = memref.alloc() : memref<1x1x1024x512xf16, @DDR>
            %cK  = VPUIP.Copy inputs(%svK : memref<1x1x1024x512xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                              outputs(%kbuf : memref<1x1x1024x512xf16, @DDR>)
                              -> memref<1x1x1024x512xf16, @DDR>
            %kOut = VPUIP.ConvertDMA inputs(%cK : memref<1x1x1024x512xf16, @DDR>)
                                     outputs(%dst_k : memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                                     -> memref<1x1x1024x512xf16, [@CMX_NN, 0]>

            %svV = VPUIP.SubView %concatD [0, 0, 0, 1024] [1, 1, 1024, 512]
                : memref<1x1x1024x1536xf16, @DDR>
               to memref<1x1x1024x512xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>
            %vbuf = memref.alloc() : memref<1x1x1024x512xf16, @DDR>
            %cV  = VPUIP.Copy inputs(%svV : memref<1x1x1024x512xf16, {order = #strided, strides = [1572864, 1572864, 1536, 1]}, @DDR>)
                              outputs(%vbuf : memref<1x1x1024x512xf16, @DDR>)
                              -> memref<1x1x1024x512xf16, @DDR>
            %vOut = VPUIP.ConvertDMA inputs(%cV : memref<1x1x1024x512xf16, @DDR>)
                                     outputs(%dst_v : memref<1x1x1024x512xf16, [@CMX_NN, 0]>)
                                     -> memref<1x1x1024x512xf16, [@CMX_NN, 0]>

            return %mm, %qOut, %kOut, %vOut
                : memref<1024x1536x1x1xf16, [@CMX_NN, 0]>,
                  memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                  memref<1x1x1024x512xf16, [@CMX_NN, 0]>,
                  memref<1x1x1024x512xf16, [@CMX_NN, 0]>
        }
    )";

    auto module = parseModule(IR);
    ASSERT_NE(module.get(), nullptr);

    auto edges = collectMainEdges(module.get());
    // The whole graph collapses into one edge: every EdgeOp is
    // transitively connected through Junction D (the central ConcatView).
    ASSERT_EQ(edges.edges.size(), 1u);

    const auto& edge = edges.edges.front();

    EXPECT_EQ(edge.type, EdgeType::Many2Many);

    // Targets: exactly 4 ComputeOp boundaries (MatMul + Q + K + V).
    ASSERT_EQ(edge.targets.size(), 4u);
    EXPECT_EQ(countTargetsOfKind(edge, BoundaryKind::ComputeOp), 4u);

    // Sources: 6 ComputeOp producers (3 pairs x 2 lanes) plus AllocBuffer
    // boundaries from ConcatView output buffers.
    EXPECT_EQ(countSourcesOfKind(edge, BoundaryKind::ComputeOp), 6u);
    EXPECT_GE(countSourcesOfKind(edge, BoundaryKind::AllocBuffer), 4u);

    // Node count: every EdgeOp on the data path.
    //   6 SubView_in (svA0/A1, svB0/B1, svC0/C1)
    // + 6 Copy_in   (cA0/A1, cB0/B1, cC0/C1)
    // + 3 ConcatView (A, B, C)
    // + 3 SubView_mid (svD0..2)
    // + 3 Copy_mid    (cD0..2)
    // + 1 ConcatView D
    // + 1 GenericReshape (MatMul branch)
    // + 3 SubView_out (svQ, svK, svV)
    // + 3 Copy_out    (cQ, cK, cV)
    // = 29 nodes
    EXPECT_EQ(edge.nodes.size(), 29u);

    EXPECT_TRUE(edge.hasConcatNode());
    EXPECT_TRUE(edge.validate(_log));

    EXPECT_EQ(countConcatNodes(edge), 4u);
}
