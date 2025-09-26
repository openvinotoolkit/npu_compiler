//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"

#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/ops.hpp"
#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/types.hpp"

#include <npu_40xx_nnrt.hpp>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace npu40xx;

template <typename T>
class MLIR_VPURegMapped_RegisterMapped : public MLIR_UnitBase {
    using registerType = typename T::Type;

    T testedRegisterDesc;
    std::unique_ptr<mlir::MLIRContext> ctx;

public:
    MLIR_VPURegMapped_RegisterMapped(): MLIR_UnitBase() {
        ctx = std::make_unique<mlir::MLIRContext>();
        ctx->appendDialectRegistry(registry);
        ctx->loadDialect<NPUReg40XX::NPUReg40XXDialect>();
        ctx->loadDialect<VPURegMapped::VPURegMappedDialect>();
    }

    void testFuncCreate() {
        mlir::OwningOpRef<mlir::ModuleOp> module =
                mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx.get()), StringRef("mainModule"));
        auto log = Logger{"mapped register test", LogLevel::Trace};
        auto builder = mlir::OpBuilder(module->getBodyRegion());

        const auto funcType = builder.getFunctionType({}, {});

        auto func = builder.create<mlir::func::FuncOp>(
                builder.getUnknownLoc(), printToString("MLIR_VPURegMapped_CreateDpuVariantRegister"), funcType,
                builder.getStringAttr("private"), /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);

        auto funcbuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), builder.getListener());

        [[maybe_unused]] auto reg = VPURegMapped::RegisterMappedAttr::get(
                ctx.get(), registerType::get(funcbuilder, registerType::getResetInitilizationValues()));

        funcbuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

        mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
        auto initCompilerOptions =
                VPU::InitCompilerOptions(vpux::config::ArchKind::NPU40XX, config::CompilationMode::DefaultHW);

        VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

        VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module.get())), "Compilation failed");
        VPUX_THROW_UNLESS(mlir::succeeded(mlir::verify(module.get())),
                          "Failed to create a valid MLIR module for the IR model");
    }
    void testFuncCheckSize() {
        mlir::OwningOpRef<mlir::ModuleOp> module =
                mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx.get()), StringRef("mainModule"));
        auto builder = mlir::OpBuilder(module->getBodyRegion());

        auto reg = registerType::get(builder, registerType::getResetInitilizationValues());

        EXPECT_EQ(reg.getWidth(), testedRegisterDesc.getSize());
    }
    void testVersion() {
        mlir::OwningOpRef<mlir::ModuleOp> module =
                mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx.get()), StringRef("mainModule"));
        auto builder = mlir::OpBuilder(module->getBodyRegion());

        auto reg = registerType::get(builder, registerType::getResetInitilizationValues());

        auto requiredVersion = reg.getRequiredMIVersion();
        EXPECT_TRUE(requiredVersion.checkValidity());
    }
};

template <typename T, uint32_t sizeInBytes>
class testRegisterSize {
public:
    using Type = T;
    static Byte getSize() {
        return Byte(sizeInBytes);
    }
};

// known register types and their sizes in bytes
using RegisterTypes = ::testing::Types<
        testRegisterSize<NPUReg40XX::RegMapped_DpuVariantRegisterType, sizeof(nn_public::VpuDPUVariant)>,
        testRegisterSize<NPUReg40XX::RegMapped_DpuInvariantRegisterType, sizeof(nn_public::VpuDPUInvariant)>,
        testRegisterSize<NPUReg40XX::RegMapped_DMARegisterType, sizeof(nn_public::VpuDMATask)>,
        testRegisterSize<NPUReg40XX::RegMapped_VpuBarrierCountConfigType, sizeof(nn_public::VpuBarrierCountConfig)>,
        testRegisterSize<NPUReg40XX::RegMapped_VpuActKernelRangeType, sizeof(nn_public::VpuActKernelRange)>,
        testRegisterSize<NPUReg40XX::RegMapped_VpuActKernelInvocationType, sizeof(nn_public::VpuActKernelInvocation)>>;
TYPED_TEST_SUITE(MLIR_VPURegMapped_RegisterMapped, RegisterTypes);

TYPED_TEST(MLIR_VPURegMapped_RegisterMapped, CreateRegisterAttr) {
    this->testFuncCreate();
}
TYPED_TEST(MLIR_VPURegMapped_RegisterMapped, CheckRegistersSize) {
    this->testFuncCheckSize();
}
TYPED_TEST(MLIR_VPURegMapped_RegisterMapped, CheckRegistersVersion) {
    this->testVersion();
}
