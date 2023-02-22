//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "test_model/kmb_test_base.hpp"

class KmbNetworkNameTest : public KmbLayerTestBase {
public:
    TestNetwork buildPowerLayer(const std::string& netName);
    std::string runTest(TestNetwork& netDesc, const std::string& netName);
};

TestNetwork KmbNetworkNameTest::buildPowerLayer(const std::string& netName) {
    const Precision precision = Precision::FP16;
    const std::vector<size_t> dims = {1, 3, 224, 224};
    const Layout layout = Layout::NCHW;
    const auto userInDesc = TensorDesc(precision, dims, layout);
    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.0f);
    });

    const auto powerTensorDesc = TensorDesc(Precision::FP32, {1, 1, 1, 1}, Layout::NCHW);
    registerBlobGenerator("scale", powerTensorDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.0f);
    });

    TestNetwork testNet;
    testNet.setUserInput("input", precision, layout)
            .addNetInput("input", dims, Precision::FP32)
            .addLayer<PowerLayerDef>("power")
            .input1("input")
            .input2(getBlobByName("scale"))
            .build()
            .setUserOutput(PortInfo("power"), precision, layout)
            .addNetOutput(PortInfo("power"))
            .finalize(netName);

    return testNet;
}

std::string KmbNetworkNameTest::runTest(TestNetwork& netDesc, const std::string& netFileName) {
    const auto blobFileName = vpux::printToString("{0}/{1}.net", DUMP_PATH, netFileName);
    if (RUN_COMPILER) {
        ExecutableNetwork exeNet = KmbLayerTestBase::getExecNetwork(netDesc);
        exeNet.Export(blobFileName);
    }
    ExecutableNetwork importedNet = core->ImportNetwork(blobFileName, DEVICE_NAME, {});
    const std::string netName = importedNet.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));

    return netName;
}
// [Track number: E#9687]
TEST_F(KmbNetworkNameTest, DISABLED_fetchNetworkName) {
    const std::string expectedNetName = "singleLayerNet";
    TestNetwork testNet = buildPowerLayer(expectedNetName);
    std::string netName = runTest(testNet, expectedNetName);

    ASSERT_EQ(netName, expectedNetName);
}

// [Track number: E#9687]
TEST_F(KmbNetworkNameTest, DISABLED_checkUniqueId) {
    const std::string firstExpectedNetName = "firstNet";
    const std::string secondExpectedNetName = "secondNet";
    TestNetwork firstNetDesc = buildPowerLayer(firstExpectedNetName);
    TestNetwork secondNetDesc = buildPowerLayer(secondExpectedNetName);
    std::string squeezeNetName = runTest(firstNetDesc, firstExpectedNetName);
    std::string googleNetName = runTest(secondNetDesc, secondExpectedNetName);

    ASSERT_NE(squeezeNetName, googleNetName);
}
