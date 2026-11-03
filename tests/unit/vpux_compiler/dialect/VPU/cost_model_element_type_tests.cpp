//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/types/quantile_types/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"

#include "common/utils.hpp"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/QuantStorageTypeInterface.h>

#include <array>
#include <optional>
#include <tuple>

using namespace vpux;

namespace {

using Signedness = mlir::IntegerType::SignednessSemantics;

class MLIR_VPUNNElementTypeTest : public MLIR_UnitBase {
public:
    MLIR_VPUNNElementTypeTest() {
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<mlir::quant::QuantDialect>();
    }

    mlir::Type intType(unsigned width, Signedness sign = Signedness::Signless) {
        return mlir::IntegerType::get(&ctx, width, sign);
    }

    // quant.uniform<storage:f16, 1.0:0> covering the storage type's natural value range.
    mlir::Type quantType(mlir::Type storage, bool isSigned) {
        const auto bounds = mlir::cast<mlir::QuantStorageTypeInterface>(storage);
        return mlir::quant::UniformQuantizedType::get(
                isSigned ? static_cast<unsigned>(mlir::quant::QuantizationFlags::Signed) : 0u, storage,
                mlir::Float16Type::get(&ctx), /*scale=*/1.0, /*zeroPoint=*/0, bounds.getDefaultMinimum(isSigned),
                bounds.getDefaultMaximum(isSigned));
    }

    mlir::Type nf4Type(Signedness sign) {
        return vpux::type::NF4Type::get(&ctx, intType(4, sign), mlir::Float16Type::get(&ctx));
    }

    // Quantile storage width VPUNN has no plain-integer counterpart for; the LUT size must match
    // the explicit bounds (2^3 = 8 representable values).
    mlir::Type unsupportedWidthQuantileType() {
        const std::array<double, 8> quantiles = {-1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0};
        return mlir::quant::QuantileType::get(&ctx, intType(3), mlir::Float16Type::get(&ctx), quantiles,
                                              /*storageMin=*/0, /*storageMax=*/7);
    }

    static std::optional<VPUNN::DataType> legacy(mlir::Type type) {
        return VPU::getVPUNNElementType(type, VPUNN::VPUDevice::VPU_4_0);
    }

    mlir::MLIRContext ctx;
};

}  // namespace

TEST_F(MLIR_VPUNNElementTypeTest, FloatTypes) {
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::BFloat16Type::get(&ctx)), VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float16Type::get(&ctx)), VPUNN::DataType::FLOAT16);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float32Type::get(&ctx)), VPUNN::DataType::FLOAT32);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float64Type::get(&ctx)), VPUNN::DataType::FLOAT64);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float8E5M2Type::get(&ctx)), VPUNN::DataType::BF8);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float8E4M3Type::get(&ctx)), VPUNN::DataType::HF8);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float8E4M3FNType::get(&ctx)), VPUNN::DataType::HF8);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float4E2M1FNType::get(&ctx)), VPUNN::DataType::FLOAT4);
}

// Signless integers follow the MLIR convention of being interpreted as signed.
TEST_F(MLIR_VPUNNElementTypeTest, IntegerSignednessIsHonored) {
    const SmallVector<std::tuple<unsigned, VPUNN::DataType, VPUNN::DataType>> cases = {
            {1, VPUNN::DataType::INT1, VPUNN::DataType::UINT1},
            {2, VPUNN::DataType::INT2, VPUNN::DataType::UINT2},
            {4, VPUNN::DataType::INT4, VPUNN::DataType::UINT4},
            {8, VPUNN::DataType::INT8, VPUNN::DataType::UINT8},
            {16, VPUNN::DataType::INT16, VPUNN::DataType::UINT16},
            {32, VPUNN::DataType::INT32, VPUNN::DataType::UINT32},
            {64, VPUNN::DataType::INT64, VPUNN::DataType::UINT64},
    };

    for (const auto& [width, signedDataType, unsignedDataType] : cases) {
        SCOPED_TRACE(width);
        EXPECT_EQ(VPU::getVPUNNElementType(intType(width, Signedness::Signless)), signedDataType);
        EXPECT_EQ(VPU::getVPUNNElementType(intType(width, Signedness::Signed)), signedDataType);
        EXPECT_EQ(VPU::getVPUNNElementType(intType(width, Signedness::Unsigned)), unsignedDataType);
    }
}

TEST_F(MLIR_VPUNNElementTypeTest, QuantizedIntegerStorage) {
    const SmallVector<std::tuple<unsigned, VPUNN::DataType, VPUNN::DataType>> cases = {
            {1, VPUNN::DataType::INT1, VPUNN::DataType::UINT1},    {2, VPUNN::DataType::INT2, VPUNN::DataType::UINT2},
            {4, VPUNN::DataType::INT4, VPUNN::DataType::UINT4},    {8, VPUNN::DataType::INT8, VPUNN::DataType::UINT8},
            {16, VPUNN::DataType::INT16, VPUNN::DataType::UINT16},
    };

    for (const auto& [width, signedDataType, unsignedDataType] : cases) {
        SCOPED_TRACE(width);
        EXPECT_EQ(VPU::getVPUNNElementType(quantType(intType(width), /*isSigned=*/true)), signedDataType);
        EXPECT_EQ(VPU::getVPUNNElementType(quantType(intType(width, Signedness::Unsigned), /*isSigned=*/false)),
                  unsignedDataType);
    }
}

// Low-precision float storage must not be reported as an integer of the same width.
TEST_F(MLIR_VPUNNElementTypeTest, QuantizedFloatStorage) {
    EXPECT_EQ(VPU::getVPUNNElementType(quantType(mlir::Float8E5M2Type::get(&ctx), /*isSigned=*/true)),
              VPUNN::DataType::BF8);
    EXPECT_EQ(VPU::getVPUNNElementType(quantType(mlir::Float8E4M3FNType::get(&ctx), /*isSigned=*/true)),
              VPUNN::DataType::HF8);
    EXPECT_EQ(VPU::getVPUNNElementType(quantType(mlir::Float4E2M1FNType::get(&ctx), /*isSigned=*/true)),
              VPUNN::DataType::FLOAT4);
}

// NF4 is a QuantileType used either directly or as the storage of a quant.uniform wrapper.
// VPUNN has no palletized weight data type (FLOAT4 is rejected as a weight type on VPU4/VPU5, and
// isn't the hardware FP4 format NF4 represents anyway), so it cost-models as the plain integer of
// the LUT-index storage width instead.
TEST_F(MLIR_VPUNNElementTypeTest, QuantileStorage) {
    EXPECT_EQ(VPU::getVPUNNElementType(nf4Type(Signedness::Signless)), VPUNN::DataType::UINT4);
    EXPECT_EQ(VPU::getVPUNNElementType(nf4Type(Signedness::Unsigned)), VPUNN::DataType::UINT4);
    EXPECT_EQ(VPU::getVPUNNElementType(quantType(nf4Type(Signedness::Signless), /*isSigned=*/true)),
              VPUNN::DataType::INT4);
    EXPECT_EQ(VPU::getVPUNNElementType(quantType(nf4Type(Signedness::Unsigned), /*isSigned=*/false)),
              VPUNN::DataType::UINT4);
}

TEST_F(MLIR_VPUNNElementTypeTest, UnsupportedTypes) {
    EXPECT_EQ(VPU::getVPUNNElementType(intType(3)), std::nullopt);
    EXPECT_EQ(VPU::getVPUNNElementType(quantType(intType(3), /*isSigned=*/true)), std::nullopt);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::Float8E4M3FNUZType::get(&ctx)), std::nullopt);
    EXPECT_EQ(VPU::getVPUNNElementType(unsupportedWidthQuantileType()), std::nullopt);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::IndexType::get(&ctx)), std::nullopt);
    EXPECT_EQ(VPU::getVPUNNElementType(mlir::NoneType::get(&ctx)), std::nullopt);
}

TEST_F(MLIR_VPUNNElementTypeTest, LegacyMapping) {
    EXPECT_EQ(legacy(mlir::Float32Type::get(&ctx)), VPUNN::DataType::FLOAT32);
    EXPECT_EQ(legacy(intType(8, Signedness::Unsigned)), VPUNN::DataType::INT8);
    EXPECT_EQ(legacy(intType(32, Signedness::Unsigned)), VPUNN::DataType::INT32);
    EXPECT_EQ(legacy(intType(16)), std::nullopt);
    EXPECT_EQ(legacy(mlir::Float64Type::get(&ctx)), std::nullopt);
    EXPECT_EQ(legacy(nf4Type(Signedness::Unsigned)), VPUNN::DataType::INT4);
    EXPECT_EQ(legacy(quantType(nf4Type(Signedness::Signless), /*isSigned=*/true)), VPUNN::DataType::INT4);
    EXPECT_EQ(legacy(quantType(nf4Type(Signedness::Unsigned), /*isSigned=*/false)), VPUNN::DataType::UINT4);

    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(mlir::Float32Type::get(&ctx), VPUNN::VPUDevice::VPU_4_0),
              VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(quantType(intType(4), /*isSigned=*/true), VPUNN::VPUDevice::VPU_4_0),
              VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(mlir::Float32Type::get(&ctx), VPUNN::VPUDevice::NPU_5_0),
              VPUNN::DataType::FLOAT32);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(mlir::Float8E4M3Type::get(&ctx), VPUNN::VPUDevice::NPU_5_0),
              VPUNN::DataType::HF8);

    EXPECT_FALSE(VPU::isVPUNNSupportedElementType(mlir::Float32Type::get(&ctx), config::ArchKind::NPU40XX));
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(mlir::Float32Type::get(&ctx), config::ArchKind::NPU50XX));
    EXPECT_FALSE(VPU::isVPUNNSupportedElementType(mlir::Float8E4M3Type::get(&ctx), config::ArchKind::NPU40XX));
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(mlir::Float8E4M3Type::get(&ctx), config::ArchKind::NPU50XX));
}

TEST_F(MLIR_VPUNNElementTypeTest, LegacyWorkloadFallback) {
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(intType(4), VPUNN::VPUDevice::VPU_4_0), VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(intType(32), VPUNN::VPUDevice::VPU_4_0), VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(intType(32), VPUNN::VPUDevice::NPU_5_0), VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(intType(1), VPUNN::VPUDevice::NPU_5_0_W), VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(nf4Type(Signedness::Signless), VPUNN::VPUDevice::VPU_4_0),
              VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(quantType(intType(2), /*isSigned=*/true), VPUNN::VPUDevice::VPU_4_0),
              VPUNN::DataType::BFLOAT16);
    EXPECT_EQ(VPU::getVPUNNWorkloadElementType(intType(8, Signedness::Unsigned), VPUNN::VPUDevice::VPU_4_0),
              VPUNN::DataType::INT8);
}

TEST_F(MLIR_VPUNNElementTypeTest, ShaveMappingUsesLegacyDeviceThroughNPU50) {
    EXPECT_EQ(VPU::getVPUNNShaveMappingDevice(VPUNN::VPUDevice::VPU_4_0), VPUNN::VPUDevice::VPU_4_0);
    EXPECT_EQ(VPU::getVPUNNShaveMappingDevice(VPUNN::VPUDevice::NPU_5_0), VPUNN::VPUDevice::VPU_4_0);
    EXPECT_EQ(VPU::getVPUNNShaveMappingDevice(VPUNN::VPUDevice::NPU_5_0_W), VPUNN::VPUDevice::VPU_4_0);
}

TEST_F(MLIR_VPUNNElementTypeTest, SupportedElementTypePerArch) {
    const auto legacyArch = config::ArchKind::NPU40XX;
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(intType(8, Signedness::Unsigned), legacyArch));
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(quantType(mlir::Float8E5M2Type::get(&ctx), /*isSigned=*/true),
                                                 legacyArch));
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(quantType(mlir::Float8E4M3FNType::get(&ctx), /*isSigned=*/true),
                                                 legacyArch));
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(quantType(intType(4), /*isSigned=*/true), legacyArch));
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(quantType(mlir::Float4E2M1FNType::get(&ctx), /*isSigned=*/true),
                                                 legacyArch));
    EXPECT_FALSE(VPU::isVPUNNSupportedElementType(quantType(intType(2), /*isSigned=*/true), legacyArch));
    EXPECT_FALSE(VPU::isVPUNNSupportedElementType(intType(4), legacyArch));
    EXPECT_FALSE(VPU::isVPUNNSupportedElementType(intType(32), legacyArch));
    EXPECT_FALSE(VPU::isVPUNNSupportedElementType(nf4Type(Signedness::Unsigned), legacyArch));

    const auto npu5Arch = config::ArchKind::NPU50XX;
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(intType(2, Signedness::Unsigned), npu5Arch));
    EXPECT_TRUE(VPU::isVPUNNSupportedElementType(mlir::Float64Type::get(&ctx), npu5Arch));
    EXPECT_FALSE(VPU::isVPUNNSupportedElementType(intType(3), npu5Arch));
}
