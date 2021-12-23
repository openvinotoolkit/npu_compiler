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

#include "vpux_compiler.hpp"

#include "vpux.hpp"
#include "vpux/al/config/compiler.hpp"

#include <vpu/utils/io.hpp>

#include <file_reader.h>
#include <file_utils.h>

#include <fstream>

// FIXME: use OPENVINO_STATIC_LIBRARY instead of
// BUILD_COMPILER_FOR_DRIVER once the compiler can be used in
// purely static build
// #ifdef OPENVINO_STATIC_LIBRARY
#ifdef BUILD_COMPILER_FOR_DRIVER
#include "vpux/compiler/compiler.hpp"
#endif

vpux::NetworkDescription::NetworkDescription(INetworkDescription::Ptr actual, const vpux::CompilerPluginPtr& plg)
        : _actual(actual), _plg(plg) {
    if (_actual == nullptr) {
        IE_THROW() << "ExecutableNetwork wrapper was not initialized.";
    }
}

static std::string extractFileName(const std::string& fullPath) {
    const size_t lastSlashIndex = fullPath.find_last_of("/\\");
    return fullPath.substr(lastSlashIndex + 1);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(const std::string& filename, const Config& config) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream.is_open()) {
        IE_THROW() << "Could not open file: " << filename;
    }
    const std::string graphName = extractFileName(filename);
    return parse(stream, config, graphName);
}

std::shared_ptr<vpux::INetworkDescription> vpux::ICompiler::parse(std::istream& stream, const Config& config,
                                                                  const std::string& graphName) {
    const size_t graphSize = vpu::KmbPlugin::utils::getFileSize(stream);
    if (graphSize == 0) {
        IE_THROW() << "Blob is empty";
    }
    std::vector<char> blob(graphSize);
    stream.read(blob.data(), graphSize);
    return parse(blob, config, graphName);
}

vpux::Compiler::Ptr vpux::Compiler::create(const Config& config) {
// FIXME: use OPENVINO_STATIC_LIBRARY instead of
// BUILD_COMPILER_FOR_DRIVER once the compiler can be used in
// purely static build
// #ifdef OPENVINO_STATIC_LIBRARY
#ifdef BUILD_COMPILER_FOR_DRIVER
    // Always use vpux compiler
    (void)(config);
    vpux::ICompiler::Ptr mlir = std::make_shared<vpux::CompilerImpl>();
    return std::make_shared<Compiler>(mlir);
#else
    const auto compilerType = config.get<COMPILER_TYPE>();

    switch (compilerType) {
    case InferenceEngine::VPUXConfigParams::CompilerType::MCM: {
        return std::make_shared<Compiler>(getLibFilePath("frontend_mcm"));
    }
    case InferenceEngine::VPUXConfigParams::CompilerType::MLIR: {
        return std::make_shared<Compiler>(getLibFilePath("vpux_compiler"));
    }
    default:
        IE_THROW() << "Compiler type not found";
    }
    IE_ASSERT(false);
#endif
}

InferenceEngine::InputsDataMap vpux::helpers::dataMapIntoInputsDataMap(const vpux::DataMap& dataMap) {
    InferenceEngine::InputsDataMap inputsDataMap = {};

    for (const auto& input : dataMap) {
        InferenceEngine::InputInfo info;
        info.setInputData(std::make_shared<InferenceEngine::Data>(*input.second));
        inputsDataMap.insert({input.first, std::make_shared<InferenceEngine::InputInfo>(info)});
    }

    return inputsDataMap;
}

InferenceEngine::OutputsDataMap vpux::helpers::dataMapIntoOutputsDataMap(const vpux::DataMap& dataMap) {
    InferenceEngine::OutputsDataMap outputsDataMap = {};

    for (const auto& output : dataMap) {
        outputsDataMap.insert({output.first, std::make_shared<InferenceEngine::Data>(*output.second)});
    }

    return outputsDataMap;
}
