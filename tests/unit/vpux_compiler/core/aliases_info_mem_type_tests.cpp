//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/logger.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "common/utils.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using MLIR_AliasesInfoMemType = MLIR_UnitBase;

TEST_F(MLIR_AliasesInfoMemType, CmxAndDdrDma) {
    mlir::MLIRContext ctx(registry);

    constexpr llvm::StringLiteral inputIR = R"(
        #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

        !Type_DDR = memref<1x1x1x5120xui8, #NCHW, @DDR>
        !Type_CMX = memref<1x1x1x5120xui8, #NCHW, [@CMX_NN, 0]>

        module @test {
            func.func @main() -> !Type_DDR {
                %cst = const.Declare !Type_DDR = dense<1> : tensor<1x1x1x5120xui8>

                %buf_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Type_CMX
                %buf_ddr = VPURT.DeclareBuffer <DDR> <0> -> !Type_DDR

                %t0, %r0 = async.execute -> !async.value<!Type_CMX> attributes {VPUIP.executor = @DMA_NN} {
                    %1 = VPUIP.Copy inputs(%cst : !Type_DDR) outputs(%buf_cmx : !Type_CMX) -> !Type_CMX
                    async.yield %1 : !Type_CMX
                }

                %t1, %r1 = async.execute [%t0] (%r0 as %arg0: !async.value<!Type_CMX>)-> !async.value<!Type_DDR> attributes {VPUIP.executor = @DMA_NN} {
                    %1 = VPUIP.Copy inputs(%arg0 : !Type_CMX) outputs(%buf_ddr : !Type_DDR) -> !Type_DDR
                    async.yield %1 : !Type_DDR
                }

                %r = async.await %r1 : !async.value<!Type_DDR>
                return %r : !Type_DDR
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    auto func = module.get().lookupSymbol<mlir::func::FuncOp>("main");
    ASSERT_TRUE(func != nullptr);

    vpux::AliasesInfoMemType<vpux::VPU::MemoryKind::CMX_NN> aliasInfoCmx(func);

    for (auto declBuffOp : func.getOps<vpux::VPURT::DeclareBufferOp>()) {
        if (declBuffOp.getSection() == vpux::VPURT::BufferSection::DDR ||
            declBuffOp.getSection() == vpux::VPURT::BufferSection::FunctionInput ||
            declBuffOp.getSection() == vpux::VPURT::BufferSection::FunctionOutput) {
            // Since analysis was only for CMX buffers get aliases for a
            // DDR buffer should return an exception
            EXPECT_ANY_THROW(aliasInfoCmx.getAllAliases(declBuffOp.getBuffer()));
        } else {
            // In the given example IR there should be 4 aliases for CMX buffer
            EXPECT_EQ(aliasInfoCmx.getAllAliases(declBuffOp.getBuffer()).size(), 4);
        }
    }
}
