//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

TEST(MLIR_SymbolLookup, LookupSymbolFromNeighborSymbolTable) {
    constexpr StringLiteral inputIR = R"(
    module @age_gender {
        func.func @main() {
            ELF.Main {
                ELF.CreateSection @Buffers aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                    VPUASM.DeclareBuffer @buffer !VPUASM.Buffer<"DDR"[0] <0> : memref<0x0x0x0xi32, @DDR> :  swizzling(0)>
                }
                ELF.CreateSection @DMAs aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                    VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) input(@Buffers::@buffer) outputs([@Buffers::@buffer]) waits([]) updates([]) start_after(0) clean_after(0) dma_descriptor(<numPlanes = 0 : i32, len = 5280 : i32, srcWidth = 5280 : i32, srcStride = 5280 : i32, srcPlaneStride = 0 : i32, dstWidth = 5280 : i32, dstStride = 5280 : i32, dstPlaneStride = 0 : i32>) acceleration_mode(<DISABLE>) is_out_of_order() is_critical() tile_indexes([1])
                }
            }
            return
        }
    })";

    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    VPUASM::NNDMAOp from = nullptr;
    module->walk([&](VPUASM::NNDMAOp op) {
        from = op;
    });
    ASSERT_NE(from, nullptr);

    auto resolvedOp = ELF::lookupNearestSymbolFrom(from, from.getInput());
    ASSERT_NE(resolvedOp, nullptr) << "Failed to resolve input symbol reference";

    auto declareBufferOp = mlir::dyn_cast<VPUASM::DeclareBufferOp>(resolvedOp);
    ASSERT_NE(declareBufferOp, nullptr) << "Resolved operation is not a DeclareBufferOp, got: "
                                        << resolvedOp->getName().getStringRef().str();

    EXPECT_EQ(declareBufferOp.getSymName(), "buffer");
}

TEST(MLIR_SymbolLookup, LookupNeighborSymbolTable) {
    constexpr StringLiteral inputIR = R"(
    module @age_gender {
        func.func @main() {
            ELF.Main {
                ELF.CreateSection @target_symbol_table aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                }
                ELF.CreateSymbolTableSection @symbol_table secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
                    ELF.Symbol @reference of(@target_symbol_table) type(<STT_SECTION>) size(46128)
                }
            }
            return
        }
    })";

    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    ELF::SymbolOp from = nullptr;
    module->walk([&](ELF::SymbolOp op) {
        from = op;
    });
    ASSERT_NE(from, nullptr);

    auto resolvedOp = ELF::lookupNearestSymbolFrom(from, from.getReference());
    ASSERT_NE(resolvedOp, nullptr) << "Failed to resolve input symbol reference";

    auto target = mlir::dyn_cast<ELF::DataSectionOp>(resolvedOp);
    ASSERT_NE(target, nullptr) << "Resolved operation is not a target, got: "
                               << resolvedOp->getName().getStringRef().str();

    EXPECT_EQ(target.getSymName(), "target_symbol_table");
}

TEST(MLIR_SymbolUses, SymbolInParent) {
    constexpr StringLiteral inputIR = R"(
    module @age_gender {
        func.func @main() {
            ELF.Main {
                VPUASM.DeclareBuffer @symbol_0 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1xf16, {order = affine_map<(d0) -> (d0)>}, [@CMX_NN, 0]> :  swizzling(0)>
                ELF.Symbol @reference_0 of(@symbol_0) type(<STT_SECTION>) size(0) value(0)
                ELF.CreateSymbolTableSection @symbol_table secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
                    ELF.Symbol @reference_1 of(@symbol_0) type(<STT_SECTION>) size(0) value(0)
                }
            }
            return
        }
    })";

    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    ELF::MainOp mainOp = nullptr;
    module->walk([&](ELF::MainOp op) {
        mainOp = op;
    });
    ASSERT_NE(mainOp, nullptr);

    VPUASM::DeclareBufferOp symbol = nullptr;
    module->walk([&](VPUASM::DeclareBufferOp op) {
        symbol = op;
    });
    ASSERT_NE(symbol, nullptr);

    ELF::SymbolOp reference0, reference1;
    module->walk([&](ELF::SymbolOp op) {
        if (op.getSymName() == "reference_0") {
            reference0 = op;
        } else if (op.getSymName() == "reference_1") {
            reference1 = op;
        }
    });
    ASSERT_NE(reference0, nullptr);
    ASSERT_NE(reference1, nullptr);

    ELF::CreateSymbolTableSectionOp nested_symbol_table = nullptr;
    module->walk([&](ELF::CreateSymbolTableSectionOp op) {
        nested_symbol_table = op;
    });
    ASSERT_NE(nested_symbol_table, nullptr) << "Nested symbol table not found in IR";

    const auto searchMain = ELF::getSymbolUses(symbol, mainOp);
    ASSERT_EQ(searchMain.size(), 2);
    ASSERT_EQ(searchMain[0].getUser(), reference0);
    ASSERT_EQ(searchMain[1].getUser(), reference1);

    const auto searchSymbolTable = mlir::SymbolTable::getSymbolUses(symbol, nested_symbol_table);
    ASSERT_TRUE(searchSymbolTable.has_value());
    ASSERT_TRUE(searchSymbolTable.value().empty());

    const auto searchSymbolTableRegion = mlir::SymbolTable::getSymbolUses(symbol, &nested_symbol_table->getRegion(0));
    ASSERT_TRUE(searchSymbolTableRegion.has_value());
    ASSERT_EQ(llvm::size(searchSymbolTableRegion.value()), 1);
    ASSERT_EQ(searchSymbolTableRegion.value().begin()->getUser(), reference1.getOperation());
}

TEST(MLIR_SymbolUses, SymbolInNestedSymbolTable) {
    constexpr StringLiteral inputIR = R"(
    module @age_gender {
        func.func @main() {
            ELF.Main {
                ELF.CreateSymbolTableSection @target_symbol_table secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
                    VPUASM.DeclareBuffer @symbol_0 !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1xf16, {order = affine_map<(d0) -> (d0)>}, [@CMX_NN, 0]> :  swizzling(0)>
                }
                ELF.Symbol @reference_0 of(@target_symbol_table::@symbol_0) type(<STT_SECTION>) size(0) value(0)
                ELF.CreateSymbolTableSection @symbol_table secFlags("VPU_SHF_JIT|VPU_SHF_USERINPUT") {
                    ELF.Symbol @reference_1 of(@target_symbol_table::@symbol_0) type(<STT_SECTION>) size(0) value(0)
                    ELF.Symbol @reference_2 of(@target_symbol_table::@symbol_0) type(<STT_SECTION>) size(0) value(0)
                }
            }
            return
        }
    })";

    auto registry = vpux::createDialectRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    ELF::MainOp mainOp = nullptr;
    module->walk([&](ELF::MainOp op) {
        mainOp = op;
    });
    ASSERT_NE(mainOp, nullptr);

    VPUASM::DeclareBufferOp symbol = nullptr;
    module->walk([&](VPUASM::DeclareBufferOp op) {
        symbol = op;
    });
    ASSERT_NE(symbol, nullptr);

    ELF::SymbolOp reference0, reference1, reference2;
    module->walk([&](ELF::SymbolOp op) {
        if (op.getSymName() == "reference_0") {
            reference0 = op;
        } else if (op.getSymName() == "reference_1") {
            reference1 = op;
        } else if (op.getSymName() == "reference_2") {
            reference2 = op;
        }
    });
    ASSERT_NE(reference0, nullptr);
    ASSERT_NE(reference1, nullptr);
    ASSERT_NE(reference2, nullptr);

    const auto searchMain = ELF::getSymbolUses(symbol, mainOp);
    ASSERT_EQ(searchMain.size(), 3);
    ASSERT_EQ(searchMain[0].getUser(), reference0);
    ASSERT_EQ(searchMain[1].getUser(), reference1);
    ASSERT_EQ(searchMain[2].getUser(), reference2);
}
