//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/ppe_utils.hpp"

#include "vpux/compiler/NPU50XX/dialect/VPU/impl/ppe_factory.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

#define EXPECT_FP_ATTR_ARRAY_NEAR(act, ref)                                        \
    {                                                                              \
        ASSERT_NE(act, nullptr);                                                   \
        std::vector<double> values(act.size());                                    \
        llvm::transform(act, values.begin(), [](const auto attr) {                 \
            return mlir::cast<mlir::FloatAttr>(attr).getValueAsDouble();           \
        });                                                                        \
        EXPECT_THAT(values, testing::Pointwise(testing::DoubleNear(1.0e-8), ref)); \
    }

class NPU50xxPpeIfcUnitTest : public VPU_PpeUnitBase {
public:
    NPU50xxPpeIfcUnitTest(): VPU_PpeUnitBase(std::make_unique<vpux::VPU::arch50xx::PpeFactory>()) {
    }
};

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Adapters) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    const auto clampAdapter = dynamic_cast<const vpux::VPU::IPpeAdapterClamp*>(_ppeIfc.get());
    ASSERT_NE(clampAdapter, nullptr);

    auto newClampsAttr = VPU::PPEFpAttr::get(
            &_ctx, fpPpeAttr.getMode(), vpux::getFPAttr(&_ctx, std::numeric_limits<float>::lowest()),
            vpux::getFPAttr(&_ctx, 14.0), fpPpeAttr.getScale(), fpPpeAttr.getPreluAlpha(), fpPpeAttr.getBias(),
            fpPpeAttr.getAdder(), fpPpeAttr.getIn1Mult(), fpPpeAttr.getIn2Mult(), fpPpeAttr.getSprlut());

    auto updatedPpe = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(clampAdapter->updateClamps(fpPpeAttr, newClampsAttr));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(updatedPpe.getClampHigh(), 14.0);

    updatedPpe =
            mlir::dyn_cast<vpux::VPU::PPEFpAttr>(clampAdapter->intersectClamps(updatedPpe, -16.0, 16.0, getU8Type()));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getClampLow(), -8000.0);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getClampHigh(), 14.0);

    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(clampAdapter->discardClamp(updatedPpe, nullptr));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(updatedPpe.getClampHigh(), std::numeric_limits<float>::max());

    const auto adapterScale = dynamic_cast<const vpux::VPU::IPpeAdapterScaleBias*>(_ppeIfc.get());
    ASSERT_NE(adapterScale, nullptr);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(adapterScale->updateScale(fpPpeAttr, {0.1}));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getScale(), 0.1);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(adapterScale->updateBias(updatedPpe, 1.0));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getBias(), 1.0);

    const auto adapterPreluAlpha = dynamic_cast<const vpux::VPU::IPpeAdapterFpPreluAlpha*>(_ppeIfc.get());
    ASSERT_NE(adapterPreluAlpha, nullptr);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(adapterPreluAlpha->updateFpPreluAlpha(fpPpeAttr, {0.1}));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(updatedPpe.getPreluAlpha(), {0.1});

    const auto adapterMode = dynamic_cast<const vpux::VPU::IPpeAdapterMode*>(_ppeIfc.get());
    ASSERT_NE(adapterMode, nullptr);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(adapterMode->updateMode(fpPpeAttr, vpux::VPU::PPEMode::LRELUX));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(updatedPpe.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);

    const auto adapterWTInfo = dynamic_cast<const vpux::VPU::IPpeAdapterWeightsTableInfo*>(_ppeIfc.get());
    ASSERT_NE(adapterWTInfo, nullptr);
    EXPECT_EQ(adapterWTInfo->hasWeightsTable(fpPpeAttr), false);
    updatedPpe = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(adapterWTInfo->useWeightsTable(fpPpeAttr));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(adapterWTInfo->hasWeightsTable(updatedPpe), true);
    updatedPpe =
            mlir::dyn_cast<vpux::VPU::PPEFpAttr>(adapterWTInfo->discardWeightsTableIfPresent(updatedPpe, 0.1, 0.5));
    ASSERT_NE(updatedPpe, nullptr);
    EXPECT_EQ(adapterWTInfo->hasWeightsTable(updatedPpe), false);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getScale(), 0.1);
    EXPECT_FP_ATTR_NEAR(updatedPpe.getBias(), 0.5);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_F16_NOOP) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_U8_NOOP) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_F16_NOOP) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.19209e-07);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_NOOP) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_F8_NOOP) {
    auto op = createAdd(getF16Type(), getF16Type(), getF8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F8_F8_F16_NOOP) {
    auto op = createAdd(getF8Type(), getF8Type(), getF16Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {0.002});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {0.002});
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F8_F8_F8_NOOP) {
    auto op = createAdd(getU8Type(), getU8Type(), getF8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 5.96046e-05);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_F16_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_U8_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_F16_RELU) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.19209e-07);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_F16_LEAKY_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_U8_LEAKY_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_F16_LEAKY_RELU) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.19209e-07);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_LEAKY_RELU) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_F16_CLAMP) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::ClampAttr>(-25.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -25.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 300.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_F16_F16_U8_CLAMP) {
    auto op = createAdd(getF16Type(), getF16Type(), getU8Type(), create<IE::ClampAttr>(-25.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_F16_CLAMP) {
    auto op = createAdd(getU8Type(), getU8Type(), getF16Type(), create<IE::ClampAttr>(-25.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -25.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 300.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.19209e-07);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_CLAMP) {
    auto op = createAdd(getF16Type(), getF16Type(), getF16Type(), create<IE::ClampAttr>(-25.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -25.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 300.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_TANH) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::TanhAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 5.96e-05);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_SIGMOID) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::SigmoidAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 5.96e-05);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_SWISH_THROWS_ON_NON_ONE_BETA) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::SwishAttr>(0.1));
    ASSERT_NE(op, nullptr);
    EXPECT_ANY_THROW([[maybe_unused]] auto ppeAttr = _ppeIfc->retrievePPEAttribute(op));
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_SWISH) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::SwishAttr>(1.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 5.96e-05);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_GELU) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::GeluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 5.96e-05);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Add_U8_U8_U8_EXP) {
    auto op = createAdd(getU8Type(), getU8Type(), getU8Type(), create<IE::ExpAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::EXP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 5.96e-05);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn1Mult(), {16777.0});
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getIn2Mult(), {16777.0});
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_F16_NOOP_BiasPT) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.25);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_NOOP_BiasPT) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 250.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.25);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_NOOP_BiasPT) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 62500);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_NOOP_BiasPT) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.25);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_F8_NOOP_BiasPT) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF8Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 250.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.25);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F8_F8_F16_NOOP_BiasPT) {
    auto op = createConvolution(getF8Type(), getF8Type(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 62500);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F8_F8_F8_BiasPT) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF8Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 448.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.001);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 62500);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8PC_NOOP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8PerAxisType(), 0.5, nullptr, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_U8PC_FP16_NOOP) {
    auto op = createConvolution(getF16Type(), getU8PerAxisType(), getF16Type(), 0.5, nullptr, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8PC_F16_F16_NOOP) {
    auto op = createConvolution(getU8PerAxisType(), getF16Type(), getF16Type(), 0.5, nullptr, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_U8PC_F16_NOOP_BiasPT) {
    auto op = createConvolution(getF16Type(), getU8PerAxisType(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8PC_F16_F16_NOOP_BiasPT) {
    auto op = createConvolution(getU8PerAxisType(), getF16Type(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_F16_NOOP_BiasSplatPC) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(16, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.25);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_F16_NOOP_BiasPC) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                                      0.25, 0.1, 0.1, 0.1, 0.1, 0.1}));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8PC_NOOP_BiasPT) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8PerAxisType(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(1, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8PC_NOOP_BiasPC) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8PerAxisType(), 0.5, nullptr,
                                createBias(SmallVector<type::float16>(16, 0.25)));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_F16_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 250.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_RELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::ReluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_F16_LEAKY_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_LEAKY_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 250.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_LEAKY_RELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_LEAKY_RELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::LeakyReluAttr>(0.1), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_F16_CLAMP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::ClampAttr>(-25.0, 300.0),
                                nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -25.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 300.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_CLAMP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::ClampAttr>(-25.0, 300.0),
                                nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 250.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_CLAMP) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::ClampAttr>(-25.0, 300.0),
                                nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -25.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 300.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_CLAMP) {
    auto op = createConvolution(getF16Type(), getF16Type(), getF16Type(), 0.5, create<IE::ClampAttr>(-25.0, 300.0),
                                nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -25.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 300.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_TANH) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::TanhAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_SIGMOID) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::SigmoidAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_SWISH) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::SwishAttr>(1.0), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_F16_F16_U8_GELU) {
    auto op = createConvolution(getF16Type(), getF16Type(), getU8Type(), 0.5, create<IE::GeluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_TANH) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::TanhAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_SIGMOID) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::SigmoidAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_SWISH) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::SwishAttr>(1.0), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_F16_GELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getF16Type(), 0.5, create<IE::GeluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_TANH) {
    auto op = createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, create<IE::TanhAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2.0e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_SIGMOID) {
    auto op = createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, create<IE::SigmoidAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2.0e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_SWISH) {
    auto op = createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, create<IE::SwishAttr>(1.0), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2.0e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8_U8_GELU) {
    auto op = createConvolution(getU8Type(), getU8Type(), getU8Type(), 0.5, create<IE::GeluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 2.0e-06);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8PC_U8_TANH) {
    auto op = createConvolution(getU8Type(), getU8PerAxisType(), getU8Type(), 0.5, create<IE::TanhAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8PC_U8_SIGMOID) {
    auto op = createConvolution(getU8Type(), getU8PerAxisType(), getU8Type(), 0.5, create<IE::SigmoidAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8PC_U8_SWISH) {
    auto op = createConvolution(getU8Type(), getU8PerAxisType(), getU8Type(), 0.5, create<IE::SwishAttr>(1.0), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Conv_U8_U8PC_U8_GELU) {
    auto op = createConvolution(getU8Type(), getU8PerAxisType(), getU8Type(), 0.5, create<IE::GeluAttr>(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_EQ(fpPpeAttr.getScale(), nullptr);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_EQ(fpPpeAttr.getBias(), nullptr);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_NOOP) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {2, 2}, 0.5, nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 62.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_RELU) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {2, 2}, 0.5, create<IE::ReluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 62.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {-0.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_LEAKY_RELU) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {2, 2}, 0.5, create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 62.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_CLAMP) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {2, 2}, 0.5, create<IE::ClampAttr>(-25.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 62.5);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_F16_TANH) {
    auto op = createAvgPool(getF16Type(), getF16Type(), {1, 1}, 1.0, create<IE::TanhAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_F16_SIGMOID) {
    auto op = createAvgPool(getF16Type(), getF16Type(), {1, 1}, 1.0, create<IE::SigmoidAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_F16_SWISH) {
    auto op = createAvgPool(getF16Type(), getF16Type(), {1, 1}, 1.0, create<IE::SwishAttr>(1.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_F16_GELU) {
    auto op = createAvgPool(getF16Type(), getF16Type(), {1, 1}, 1.0, create<IE::GeluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_TANH) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {1, 1}, 1.0, create<IE::TanhAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_SIGMOID) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {1, 1}, 1.0, create<IE::SigmoidAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_SWISH) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {1, 1}, 1.0, create<IE::SwishAttr>(1.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_F16_U8_GELU) {
    auto op = createAvgPool(getF16Type(), getU8Type(), {1, 1}, 1.0, create<IE::GeluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_U8_U8_TANH) {
    auto op = createAvgPool(getU8Type(), getU8Type(), {1, 1}, 1.0, create<IE::TanhAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.002);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_U8_U8_SIGMOID) {
    auto op = createAvgPool(getU8Type(), getU8Type(), {1, 1}, 1.0, create<IE::SigmoidAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.002);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_U8_U8_SWISH) {
    auto op = createAvgPool(getU8Type(), getU8Type(), {1, 1}, 1.0, create<IE::SwishAttr>(1.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.002);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_AvgPool_U8_U8_GELU) {
    auto op = createAvgPool(getU8Type(), getU8Type(), {1, 1}, 1.0, create<IE::GeluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), -128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 127.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.002);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {500.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_F16_F16_NOOP) {
    auto op = createMaxPool(getF16Type(), getF16Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_U8_U8_NOOP) {
    auto op = createMaxPool(getU8Type(), getU8Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_F16_F32_NOOP) {
    auto op = createMaxPool(getF16Type(), getF32Type(), nullptr);
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_U8_U8_TANH) {
    auto op = createMaxPool(getU8Type(), getU8Type(), create<IE::TanhAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::TANH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_U8_U8_SIGMOID) {
    auto op = createMaxPool(getU8Type(), getU8Type(), create<IE::SigmoidAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SIGMOID);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_U8_U8_SWISH) {
    auto op = createMaxPool(getU8Type(), getU8Type(), create<IE::SwishAttr>(1.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::SWISH);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_U8_U8_GELU) {
    auto op = createMaxPool(getU8Type(), getU8Type(), create<IE::GeluAttr>());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::GELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    ASSERT_NE(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_F16_F16_CLAMP) {
    auto op = createMaxPool(getF16Type(), getF16Type(), create<IE::ClampAttr>(0.0, 6.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LRELUX);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 6.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MaxPool_U8_U8_CLAMP) {
    auto op = createMaxPool(getU8Type(), getU8Type(), create<IE::ClampAttr>(0.0, 6.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), 128.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), 383.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Subtract_F16_F16_U8_LEAKY_RELU) {
    auto op = createSubtract(getF16Type(), getF16Type(), getU8Type(), create<IE::LeakyReluAttr>(0.1));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::LPRELU);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {0.1});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_Multiply_F16_F16_U8_CLAMP) {
    auto op = createMultiply(getF16Type(), getF16Type(), getU8Type(), create<IE::ClampAttr>(-25.0, 300.0));
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MatMul_F16_F16_U8_NOOP) {
    auto op = createMatMul(getF16Type(), getF16Type(), getU8Type());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 500.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_MatMul_F8_F8_F16_NOOP) {
    auto op = createMatMul(getF8Type(), getF8Type(), getF16Type());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<float>::lowest());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<float>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.000'004);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 0.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_ReduceMean_U8_U8_NOOP) {
    auto op = createReduceMean(getU8Type(), getU8Type(), {Dims4D::Act::C.ind()}, {});
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.0625);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_ReduceMean_PAD_U8_U8_NOOP) {
    auto op = createReduceMean(getU8Type(), getU8Type(), {Dims4D::Act::C.ind()}, {0, 8, 0, 0});
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.125);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_NCEReduce_MEAN_PAD_U8_U8_NOOP) {
    auto op = createNCEReduce(getU8Type(), getU8Type(), {Dims4D::Act::C.ind()}, VPU::ReduceType::MEAN, {0, 8, 0, 0});
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.125);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_NCEReduce_SUM_PAD_U8_U8_NOOP) {
    auto op = createNCEReduce(getU8Type(), getU8Type(), {Dims4D::Act::C.ind()}, VPU::ReduceType::SUM, {0, 8, 0, 0});
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 1.0);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_NCEInterpolate_U8_U8_U8_NOOP) {
    auto op = createNCEInterpolate(getU8Type(), getU8Type(), getU8Type(), getStubPPEAttr());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.002);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}

TEST_F(NPU50xxPpeIfcUnitTest, FpPPE_NCEDWConv_U8_U8_U8_NOOP) {
    auto op = createNCEDWConv(getU8Type(), getU8Type(), getU8Type(), getStubPPEAttr());
    ASSERT_NE(op, nullptr);
    auto ppeAttr = _ppeIfc->retrievePPEAttribute(op);
    ASSERT_NE(ppeAttr, nullptr);
    auto fpPpeAttr = mlir::dyn_cast<vpux::VPU::PPEFpAttr>(ppeAttr);
    ASSERT_NE(fpPpeAttr, nullptr) << "Failed to specialize PPE attribute";

    EXPECT_EQ(fpPpeAttr.getMode().getValue(), vpux::VPU::PPEMode::NOOP);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampLow(), std::numeric_limits<int8_t>::min());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getClampHigh(), std::numeric_limits<int8_t>::max());
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getScale(), 0.002);
    EXPECT_FP_ATTR_ARRAY_NEAR(fpPpeAttr.getPreluAlpha(), {1.0});
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getBias(), 0.0);
    EXPECT_FP_ATTR_NEAR(fpPpeAttr.getAdder(), 128.0);
    EXPECT_EQ(fpPpeAttr.getIn1Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getIn2Mult(), nullptr);
    EXPECT_EQ(fpPpeAttr.getSprlut(), nullptr);
}
