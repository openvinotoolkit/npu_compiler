//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <thread>
#include <chrono>
#include <iostream>

#include <vpu/kmb_plugin_config.hpp>

#include "kmb_layers_tests.hpp"
#include "plugin_cache.hpp"

using namespace InferenceEngine;

void kmbLayersTests_nightly::NetworkInit(const std::string& layer_type,
                                         std::map<std::string, std::string>* params,
                                         int weights_size,
                                         int biases_size,
                                         InferenceEngine::TBlob<uint8_t>::Ptr weights,
                                         InferenceEngine::Precision outputPrecision,
                                         InferenceEngine::Precision inputPrecision,
                                         bool useHWOpt)
{
    ASSERT_NO_FATAL_FAILURE(
            doNetworkInit(layer_type,
                          params,
                          weights_size,
                          biases_size,
                          weights,
                          outputPrecision,
                          inputPrecision);
    );
}

void kmbLayersTests_nightly::setup(InferenceEngine::Precision outputPrecision,
                                   InferenceEngine::Precision inputPrecision,
                                   bool useHWOpt) {
    CNNNetwork network = _net_reader.getNetwork();
    _inputsInfo = network.getInputsInfo();
    for (const auto &in : _inputsInfo) {
        in.second->setPrecision(inputPrecision);
    }
    _outputsInfo = network.getOutputsInfo();
    for (const auto &outputInfo : _outputsInfo) {
        outputInfo.second->setPrecision(outputPrecision);
    }

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    InferenceEngine::StatusCode st = InferenceEngine::StatusCode::GENERAL_ERROR;
    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));

}

void kmbLayersTests_nightly::setCommonConfig(std::map<std::string, std::string>& config)
{
    config = _config;
#if 0
    config[CONFIG_KEY(LOG_LEVEL)] = CONFIG_VALUE(LOG_INFO);
#endif

#if 0 // TODO: mcmCompiler generate BLOB issue

#endif
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)]  = CONFIG_VALUE(YES);

    const ::testing::TestInfo* const test_info =
            ::testing::UnitTest::GetInstance()->current_test_info();

    config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH)] = test_info->test_case_name();
    config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS)] = test_info->name();
}


void kmbLayersTests_nightly::doNetworkInit(const std::string& layer_type,
                                           std::map<std::string, std::string>* params,
                                           int weights_size,
                                           int biases_size,
                                           InferenceEngine::TBlob<uint8_t>::Ptr weights,
                                           InferenceEngine::Precision outputPrecision,
                                           InferenceEngine::Precision inputPrecision)
{
    std::string xml;
    genXML(layer_type, params, weights_size, biases_size, xml);
    ASSERT_NO_THROW(_net_reader.ReadNetwork(xml.data(), xml.length()));
    ASSERT_EQ(_net_reader.isParseSuccess(), true);
    if (weights != nullptr)
        ASSERT_NO_THROW(_net_reader.SetWeights(weights));
    setup(outputPrecision, inputPrecision, true);
}
