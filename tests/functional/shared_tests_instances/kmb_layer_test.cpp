// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"
#include "kmb_test_report.hpp"
#include "vpux_private_config.hpp"

#include <ie_utils.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>

#include "kmb_test_tool.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include <vpu/utils/error.hpp>

namespace LayerTestsUtils {

// might need to use CommonTestUtils::DEVICE_CPU for ref calc
const TargetDevice testPlatformTargetDevice("VPUX");

const KmbTestEnvConfig KmbLayerTestsCommon::envConfig;

KmbLayerTestsCommon::KmbLayerTestsCommon(): kmbTestTool(envConfig) {
    IE_ASSERT(core != nullptr);

    if (!envConfig.IE_KMB_TESTS_LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), envConfig.IE_KMB_TESTS_LOG_LEVEL}}, testPlatformTargetDevice);
    }
}

void KmbLayerTestsCommon::BuildNetworkWithoutCompile() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();
}

void KmbLayerTestsCommon::ImportNetwork() {
    IE_ASSERT(core != nullptr);
    executableNetwork = kmbTestTool.importNetwork(core,
        filesysName(testing::UnitTest::GetInstance()->current_test_info(), ".net", !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
}

void KmbLayerTestsCommon::ExportNetwork() {
    kmbTestTool.exportNetwork(executableNetwork,
        filesysName(testing::UnitTest::GetInstance()->current_test_info(), ".net", !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
}

void KmbLayerTestsCommon::ExportInput() {
    int i = 0;
    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        const auto ext = vpu::formatString(".%v.%v", info->name(), "in");
        kmbTestTool.exportBlob(inputs[i++],
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ImportInput() {
    // infer request should be adapted afterwards
    int i = 0;
    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        const auto ext = vpu::formatString(".%v.%v", info->name(), "in");
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info->getTensorDesc());
        blob->allocate();
        kmbTestTool.importBlob(blob,
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
        inputs[i++] = blob;
    }
}

void KmbLayerTestsCommon::ExportReference(const std::vector<std::vector<std::uint8_t>>& refs) {
    size_t i = 0;
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
        const auto &name = output.first;

        auto& ref = refs[i++];
        auto referenceBlob = InferenceEngine::make_shared_blob<uint8_t>(
            InferenceEngine::TensorDesc{
                InferenceEngine::Precision::U8,
                InferenceEngine::SizeVector{ref.size()},
                InferenceEngine::Layout::C
            }, const_cast<std::uint8_t*>(&ref[0]), ref.size());
        const auto ext = vpu::formatString(".%v.%v", name, "ref");
        kmbTestTool.exportBlob(referenceBlob,
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ImportReference(std::vector<std::vector<std::uint8_t>>& refs) {
    size_t i = 0;
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
        const auto &name = output.first;

        auto& ref = refs[i++];
        auto referenceBlob = InferenceEngine::make_shared_blob<uint8_t>(
            InferenceEngine::TensorDesc{
                InferenceEngine::Precision::U8,
                InferenceEngine::SizeVector{ref.size()},
                InferenceEngine::Layout::C
            }, &ref[0], ref.size());
        const auto ext = vpu::formatString(".%v.%v", name, "ref");
        kmbTestTool.importBlob(referenceBlob,
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::GenerateInputs() {
    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        auto blob = GenerateInput(*info);
        inputs.push_back(blob);
    }
}

void KmbLayerTestsCommon::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();

    int i = 0;
    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        auto blob = inputs[i++];
        inferRequest.SetBlob(info->name(), blob);
    }

    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
        inferRequest.SetBatch(batchSize);
    }
    inferRequest.Infer();
}

void KmbLayerTestsCommon::Validate() {
    std::cout << "LayerTestsCommon::Validate()" << std::endl;

    auto expectedOutputs = CalculateRefs();
    const auto &actualOutputs = GetOutputs();

    if (envConfig.IE_KMB_TESTS_IMPORT_REF) {
        std::cout << "KmbLayerTestsCommon::ImportReference()" << std::endl;
        ImportReference(expectedOutputs);
    }

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
        << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    Compare(expectedOutputs, actualOutputs);
}

void KmbLayerTestsCommon::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::cout << "KmbLayerTestsCommon::BuildNetworkWithoutCompile" << std::endl;
    BuildNetworkWithoutCompile();
    KmbTestReport& report = KmbTestReport::getInstance();
    const auto& testInfo = testing::UnitTest::GetInstance()->current_test_info();
    report.run(testInfo);

    try {
        if (envConfig.IE_KMB_TESTS_RUN_COMPILER) {
            std::cout << "KmbLayerTestsCommon::Compile" << std::endl;
            SkipBeforeLoad();
            ASSERT_NO_THROW(executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration));
            report.compiled(testInfo);

            if (envConfig.IE_KMB_TESTS_RUN_EXPORT) {
                std::cout << "KmbLayerTestsCommon::ExportNetwork()" << std::endl;
                ASSERT_NO_THROW(ExportNetwork());
            }
        } else {
            std::cout << "KmbLayerTestsCommon::ImportNetwork()" << std::endl;
            SkipBeforeLoad();
            SkipBeforeImport();
            ImportNetwork();
            report.imported(testInfo);
        }
        GenerateInputs();
        if (envConfig.IE_KMB_TESTS_EXPORT_INPUT) {
            std::cout << "KmbLayerTestsCommon::ExportInput()" << std::endl;
            ExportInput();
        }
        if (envConfig.IE_KMB_TESTS_IMPORT_INPUT) {
            std::cout << "KmbLayerTestsCommon::ImportInput()" << std::endl;
            ImportInput();
        }
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            std::cout << "KmbLayerTestsCommon::Infer()" << std::endl;
            SkipBeforeInfer();
            Infer();
            report.inferred(testInfo);
        }
        if (envConfig.IE_KMB_TESTS_EXPORT_REF) {
            std::cout << "KmbLayerTestsCommon::ExportReference()" << std::endl;
            ExportReference(CalculateRefs());
        }
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            std::cout << "KmbLayerTestsCommon::Validate()" << std::endl;
            SkipBeforeValidate();
            Validate();
            report.validated(testInfo);
        } else {
            std::cout << "Skip KmbLayerTestsCommon::Infer()" << std::endl;
        }
    } catch (const KmbSkipTestException &e) {
        std::cout << "Skipping the test due to: " << e.what() << std::endl;
        report.skipped(testInfo);
        SKIP() << "Skipping the test due to: " << e.what();
    }
}

void KmbLayerTestsCommon::useCompilerMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
}

bool KmbLayerTestsCommon::isCompilerMCM() const {
    const auto it = configuration.find(VPUX_CONFIG_KEY(COMPILER_TYPE));
    if (it == configuration.end()) {
        // Default value for COMPILER_TYPE is MCM
        return true;
    }

    return it->second == VPUX_CONFIG_VALUE(MCM);
}

bool KmbLayerTestsCommon::isCompilerMLIR() const {
    const auto it = configuration.find(VPUX_CONFIG_KEY(COMPILER_TYPE));
    if (it == configuration.end()) {
        // Default value for COMPILER_TYPE is MCM
        return false;
    }

    return it->second == VPUX_CONFIG_VALUE(MLIR);
}

}  // namespace LayerTestsUtils
