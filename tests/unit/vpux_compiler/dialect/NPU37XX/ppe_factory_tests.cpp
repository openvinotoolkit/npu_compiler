//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/ppe_utils.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/ppe_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_utils.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

#define EXPECT_INT_ATTR_EQ(act, ref)                   \
    {                                                  \
        ASSERT_NE(act, nullptr);                       \
        EXPECT_EQ(act.getValue().getSExtValue(), ref); \
    }

#define EXPECT_FP_ATTR_NEAR(act, ref)                     \
    {                                                     \
        ASSERT_NE(act, nullptr);                          \
        EXPECT_NEAR(act.getValueAsDouble(), ref, 1.0e-8); \
    }

#define EXPECT_PACKED_CLAMP_EQ(act, refLow, refHigh)                                               \
    {                                                                                              \
        ASSERT_NE(act, nullptr);                                                                   \
        const auto unpackedClamp = VPU::unpackClamp<type::float16>(act.getValue().getSExtValue()); \
        EXPECT_NEAR(unpackedClamp.first, refLow, 1.0e-8);                                          \
        EXPECT_NEAR(unpackedClamp.second, refHigh, 1.0e-8);                                        \
    }

#define EXPECT_INT_ATTR_ARRAY_EQ(act, ref)                                        \
    {                                                                             \
        ASSERT_NE(act, nullptr);                                                  \
        std::vector<int64_t> values(act.size());                                  \
        llvm::transform(act, values.begin(), [](const auto attr) {                \
            return mlir::cast<mlir::IntegerAttr>(attr).getValue().getSExtValue(); \
        });                                                                       \
        EXPECT_THAT(values, ::testing::Pointwise(::testing::Eq(), ref));          \
    }

#define EXPECT_FP_ATTR_ARRAY_NEAR(act, ref)                                        \
    {                                                                              \
        ASSERT_NE(act, nullptr);                                                   \
        std::vector<double> values(act.size());                                    \
        llvm::transform(act, values.begin(), [](const auto attr) {                 \
            return mlir::cast<mlir::FloatAttr>(attr).getValueAsDouble();           \
        });                                                                        \
        EXPECT_THAT(values, testing::Pointwise(testing::DoubleNear(1.0e-8), ref)); \
    }

class NPU37xxPpeIfcUnitTest : public VPU_PpeUnitBase {
public:
    NPU37xxPpeIfcUnitTest(): VPU_PpeUnitBase(std::make_unique<vpux::VPU::arch37xx::PpeFactory>()) {
    }
};

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Adapters) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    const auto clampAdapter = dynamic_cast<const vpux::VPU::IPpeAdapterClamp*>(_ppeIfc.get());
    ASSERT_NE(clampAdapter, nullptr);

    const auto newClampsAttr = vpux::VPU::PPEIntAttr::get(
            &_ctx, intPpeAttr.getMode(), vpux::getIntAttr(&_ctx, std::numeric_limits<int32_t>::min()),
            vpux::getIntAttr(&_ctx, 14), intPpeAttr.getLreluMult(), intPpeAttr.getLreluShift(),
            intPpeAttr.getQuantScale(), intPpeAttr.getQuantMult(), intPpeAttr.getQuantShift(),
            intPpeAttr.getQuantPostShift(), intPpeAttr.getIn1QuantMult(), intPpeAttr.getIn2QuantMult(),
            intPpeAttr.getFpPreluAlpha());

    auto updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(clampAdapter->updateClamps(intPpeAttr, newClampsAttr));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(updatedPpe.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(updatedPpe.getClampHigh().getValue().getSExtValue(), 14);

    updatedPpe =
            mlir::dyn_cast<vpux::VPU::PPEIntAttr>(clampAdapter->intersectClamps(updatedPpe, -16.0, 16.0, getU8Type()));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(updatedPpe.getClampLow().getValue().getSExtValue(), -7872);
    EXPECT_EQ(updatedPpe.getClampHigh().getValue().getSExtValue(), 14);

    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(clampAdapter->discardClamp(updatedPpe, getF16Type()));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(updatedPpe.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    // packed fp16 min & fp16 max
    EXPECT_EQ(updatedPpe.getClampHigh().getValue().getSExtValue(), 2147483647);

    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(
            clampAdapter->discardClamp(updatedPpe, mlir::BFloat16Type::get(&_ctx)));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(updatedPpe.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    // packed bf16 min & bf16 max
    EXPECT_EQ(updatedPpe.getClampHigh().getValue().getSExtValue(), 2147483647);

    constexpr int64_t bitWidth = 8;
    const auto storageMin = mlir::quant::QuantizedType::getDefaultMinimumForInteger(true, bitWidth);
    const auto storageMax = mlir::quant::QuantizedType::getDefaultMaximumForInteger(true, bitWidth);
    const auto storageType = mlir::IntegerType::get(&_ctx, bitWidth, mlir::IntegerType::Signed);
    auto quantType = mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, storageType,
                                                            getF16Type(), 1.2, 0, storageMin, storageMax);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(clampAdapter->discardClamp(updatedPpe, quantType));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(updatedPpe.getClampLow().getValue().getSExtValue(), storageMin);
    EXPECT_EQ(updatedPpe.getClampHigh().getValue().getSExtValue(), storageMax);

    const auto adapterScale = dynamic_cast<const vpux::VPU::IPpeAdapterScaleBias*>(_ppeIfc.get());
    ASSERT_NE(adapterScale, nullptr);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(adapterScale->updateScale(intPpeAttr, {0.1}));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(updatedPpe.getQuantScale(), {0.1});

    const auto adapterPreluAlpha = dynamic_cast<const vpux::VPU::IPpeAdapterFpPreluAlpha*>(_ppeIfc.get());
    ASSERT_NE(adapterPreluAlpha, nullptr);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(adapterPreluAlpha->updateFpPreluAlpha(intPpeAttr, {0.1}));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getFpPreluAlpha(), 0.1);

    const auto adapterMode = dynamic_cast<const vpux::VPU::IPpeAdapterMode*>(_ppeIfc.get());
    ASSERT_NE(adapterMode, nullptr);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(adapterMode->updateMode(intPpeAttr, vpux::VPU::PPEMode::LRELUX));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(updatedPpe.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);

    const auto adapterQuantParams = dynamic_cast<const vpux::VPU::IPpeAdapterQuantParams*>(_ppeIfc.get());
    ASSERT_NE(adapterQuantParams, nullptr);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(
            adapterQuantParams->recomputeQuantParams(intPpeAttr, getF16Type(), getU8Type(), {2, 2}));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(updatedPpe.getQuantMult(), {32000});
    EXPECT_INT_ATTR_ARRAY_EQ(updatedPpe.getQuantShift(), {8});
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_INT_ATTR_EQ(updatedPpe.getQuantPostShift(), 0.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_F16_NOOP) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {1.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_U8_NOOP) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {500.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 499.99996948242188);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_F16_NOOP) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {16384});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {37});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_U8_NOOP) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {32000});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {29});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_F16_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {1.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_U8_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {500.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 499.99996948242188);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_F16_RELU) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {16384});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {37});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_U8_RELU) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {32000});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {29});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_F16_LEAKY_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {1.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 0.1);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_U8_LEAKY_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {500.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 50.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_F16_LEAKY_RELU) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {16384});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {37});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 0.1);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_U8_LEAKY_RELU) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {32000});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {29});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 0.1);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_F16_CLAMP) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::ClampAttr>(20.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_PACKED_CLAMP_EQ(intPpeAttr.getClampHigh(), 20.0, 300.0f);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {1.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_F16_F16_U8_CLAMP) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), create<IE::ClampAttr>(20.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 10128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {500.0});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 499.99996948242188);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_F16_CLAMP) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), create<IE::ClampAttr>(20.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_PACKED_CLAMP_EQ(intPpeAttr.getClampHigh(), 20.0, 300.0);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {16384});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {37});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Add_U8_U8_U8_CLAMP) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::ClampAttr>(20.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 10128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantMult(), {32000});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getQuantShift(), {29});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn1QuantMult(), {16777});
    EXPECT_INT_ATTR_ARRAY_EQ(intPpeAttr.getIn2QuantMult(), {16777});
    EXPECT_INT_ATTR_EQ(intPpeAttr.getQuantPostShift(), 0.0);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_F16_NOOP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, nullptr, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_U8_NOOP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, nullptr, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 499.99996948242188);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_F16_NOOP) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, nullptr, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_U8_NOOP) {
    auto op = createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, nullptr, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_F16_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_U8_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 499.99996948242188);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_F16_RELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_U8_RELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_F16_LEAKY_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 0.1);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_U8_LEAKY_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 50.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_F16_LEAKY_RELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 0.1);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_U8_LEAKY_RELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 0.1);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_F16_CLAMP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::ClampAttr>(20.0, 300.0),
                                nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_PACKED_CLAMP_EQ(intPpeAttr.getClampHigh(), 20.0, 300.0f);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_F16_F16_U8_CLAMP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::ClampAttr>(20.0, 300.0),
                                nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 10128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 499.99996948242188);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_F16_CLAMP) {
    auto op =
            createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::ClampAttr>(20.0, 300.0), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), std::numeric_limits<int32_t>::min());
    EXPECT_PACKED_CLAMP_EQ(intPpeAttr.getClampHigh(), 20.0, 300.0);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_Conv_U8_U8_U8_CLAMP) {
    auto op =
            createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, create<IE::ClampAttr>(20.0, 300.0), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 10128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {0.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_MatMul_U8_U8_U8_NOOP) {
    auto op = createMatMul(getU8Type(), getU8Type(), getU8Type());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_AvgPool_F16_U8_RELU) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {2, 2}, 0.5, create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1638);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 14);
    EXPECT_FP_ATTR_ARRAY_NEAR(intPpeAttr.getQuantScale(), {62.5});
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 50.0);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_MaxPool_F16_U8_CLAMP) {
    auto op = createMaxPool(getF16Type(), getU8Type(), create<IE::ClampAttr>(20.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 10128);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 499.99996948242188);
}

TEST_F(NPU37xxPpeIfcUnitTest, IntPPE_ReduceMean_U8_U8_NOOP) {
    auto op = createReduceMean(getU8Type(), getU8Type(), {Dims4D::Act::C.ind()}, {});
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto intPpeAttr = mlir::dyn_cast<vpux::VPU::PPEIntAttr>(ppeAttr);
    ASSERT_NE(intPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(intPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_EQ(intPpeAttr.getClampLow().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getClampHigh().getValue().getSExtValue(), 255);
    EXPECT_EQ(intPpeAttr.getLreluMult().getValue().getSExtValue(), 1);
    EXPECT_EQ(intPpeAttr.getLreluShift().getValue().getSExtValue(), 0);
    EXPECT_EQ(intPpeAttr.getQuantScale(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantShift(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn1QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getIn2QuantMult(), nullptr);
    EXPECT_EQ(intPpeAttr.getQuantPostShift(), nullptr);
    EXPECT_FP_ATTR_NEAR(intPpeAttr.getFpPreluAlpha(), 1.0);
}
