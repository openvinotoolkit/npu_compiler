//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "kmb_layers_tests.hpp"

#include "vpux/utils/core/helper_macros.hpp"

#include <chrono>
#include <iostream>
#include <thread>

#include <vpux/vpux_compiler_config.hpp>
#include <vpux/vpux_plugin_config.hpp>

#include "functional_test_utils/plugin_cache.hpp"

using namespace InferenceEngine;

void kmbLayersTests_nightly::NetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params,
                                         int weights_size, int biases_size,
                                         InferenceEngine::TBlob<uint8_t>::Ptr weights,
                                         InferenceEngine::Precision outputPrecision,
                                         InferenceEngine::Precision inputPrecision) {
    ASSERT_NO_FATAL_FAILURE(
            doNetworkInit(layer_type, params, weights_size, biases_size, weights, outputPrecision, inputPrecision););
}

void kmbLayersTests_nightly::setup(const CNNNetwork& network, InferenceEngine::Precision outputPrecision,
                                   InferenceEngine::Precision inputPrecision, bool) {
    _inputsInfo = network.getInputsInfo();
    for (const auto& in : _inputsInfo) {
        in.second->setPrecision(inputPrecision);
        in.second->setLayout(InferenceEngine::Layout::NHWC);
    }
    _outputsInfo = network.getOutputsInfo();
    for (const auto& outputInfo : _outputsInfo) {
        outputInfo.second->setPrecision(outputPrecision);
        outputInfo.second->setLayout(InferenceEngine::Layout::NHWC);
    }

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    ASSERT_NO_THROW(core->LoadNetwork(network, deviceName, config));
}

void kmbLayersTests_nightly::doNetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params,
                                           int weights_size, int biases_size,
                                           InferenceEngine::TBlob<uint8_t>::Ptr weights,
                                           InferenceEngine::Precision outputPrecision,
                                           InferenceEngine::Precision inputPrecision) {
    std::string xml;
    genXML(layer_type, params, weights_size, biases_size, xml);
    CNNNetwork network;
    ASSERT_NO_THROW(network = core->ReadNetwork(xml, weights));
    setup(network, outputPrecision, inputPrecision);
}

std::map<std::string, std::string> KmbPerLayerTest::getCommonConfig() const {
    std::map<std::string, std::string> config;

    return config;
}

std::string KmbPerLayerTest::getTestResultFilename() const {
    std::string testResultFilename = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    for (auto& letter : testResultFilename) {
        letter = (letter == '/') ? '_' : letter;
    }

    return testResultFilename;
}

void kmbLayersTests_nightly::setCommonConfig(std::map<std::string, std::string>& config) {
    config = this->config;
}
