//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "test_model/kmb_test_base.hpp"

struct PowerTwoOutputsTestParams final {
    SizeVector _inDims;
    Precision _outPrecision;

    PowerTwoOutputsTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    PowerTwoOutputsTestParams& outPrecision(const Precision& outPrecision) {
        this->_outPrecision = outPrecision;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const PowerTwoOutputsTestParams& p) {
    vpux::printTo(os, "[inDims:{0}, outPrecision:{1}]", p._inDims, p._outPrecision);
    return os;
}

class KmbPowerTwoOutputsTest : public KmbLayerTestBase, public testing::WithParamInterface<PowerTwoOutputsTestParams> {};

// [Track number: D#3226]
TEST_P(KmbPowerTwoOutputsTest, powerWithSigmoid) {
    GTEST_SKIP() << "Skip due to error: output precision conversion from U8 to FP16 is not supported";
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(p._outPrecision, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;
    const auto netPrecision = Precision::FP32;

    registerBlobGenerator(
            "input", userInDesc,
            [&](const TensorDesc& desc) {
                return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
            }
    );

    const auto powerTensorDesc = TensorDesc(Precision::FP32, {1, 1, 1, 1}, Layout::NHWC);
    registerBlobGenerator(
            "scale", powerTensorDesc,
            [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, 1.f);
            }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPrecision)
            .addLayer<PowerLayerDef>("power")
                .input1("input")
                .input2(getBlobByName("scale"))
                .build()
            .setUserOutput(PortInfo("power"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .addNetOutput(PortInfo("power"))
            .addLayer<SigmoidLayerDef>("sigmoid")
                .input("power")
                .build()
            .addNetOutput(PortInfo("sigmoid"))
            .setUserOutput(PortInfo("sigmoid"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

TEST_P(KmbPowerTwoOutputsTest, twoPowerWithSigmoid) {
    SKIP_INFER("Error: output precision conversion from U8 to FP16 is not supported");
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(p._outPrecision, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;
    const auto netPrecision = Precision::FP32;

    registerBlobGenerator(
            "input", userInDesc,
            [&](const TensorDesc& desc) {
                return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
            }
    );

    const auto powerTensorDesc = TensorDesc(Precision::FP32, {1, 1, 1, 1}, Layout::NHWC);
    registerBlobGenerator(
            "scale", powerTensorDesc,
            [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, 1.f);
            }
    );

    registerBlobGenerator(
            "scale1", powerTensorDesc,
            [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, 1.f);
            }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
                .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
                .addNetInput("input", userInDesc.getDims(), netPrecision)
                .addLayer<PowerLayerDef>("power")
                    .input1("input")
                    .input2(getBlobByName("scale"))
                    .build()
                .addLayer<SigmoidLayerDef>("sigmoid")
                    .input("power")
                    .build()
                .addNetOutput(PortInfo("sigmoid"))
                .setUserOutput(PortInfo("sigmoid"), userOutDesc.getPrecision(), userOutDesc.getLayout())
                .addLayer<PowerLayerDef>("power1")
                        .input1("input")
                        .input2(getBlobByName("scale1"))
                        .build()
                .setUserOutput(PortInfo("power1"), userOutDesc.getPrecision(), userOutDesc.getLayout())
                .addNetOutput(PortInfo("power1"))
                .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<PowerTwoOutputsTestParams> powerParams {
        PowerTwoOutputsTestParams()
                .inDims({1, 3, 32, 32})
                .outPrecision(Precision::FP16),
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbPowerTwoOutputsTest, testing::ValuesIn(powerParams));
