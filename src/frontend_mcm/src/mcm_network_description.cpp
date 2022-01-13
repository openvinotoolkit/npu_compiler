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

#include "mcm_network_description.hpp"

#include "vpux/al/config/common.hpp"

#include "blob_parser.hpp"
#include "mcm_adapter.hpp"

using namespace vpux;
using namespace vpu::MCMAdapter;

namespace ie = InferenceEngine;

namespace {

// TODO: remove once parser works with DataMap instead of inputs and outputs info
DataMap inputsDataMapToDataMap(const InferenceEngine::InputsDataMap& inputs) {
    DataMap dataMap = {};
    for (auto&& in : inputs) {
        dataMap.insert({in.first, in.second->getInputData()});
    }

    return dataMap;
}
DataMap outputsDataMapToDataMap(const InferenceEngine::OutputsDataMap& outputs) {
    DataMap dataMap = {};
    for (auto&& out : outputs) {
        dataMap.insert({out.first, out.second});
    }

    return dataMap;
}

template <typename T>
std::vector<std::string> extractKeys(const std::map<std::string, T>& map) {
    std::vector<std::string> keys;

    std::transform(map.begin(), map.end(), std::back_inserter(keys),
                   [](const typename std::map<std::string, T>::value_type& pair) {
                       return pair.first;
                   });

    return keys;
}

}  // namespace

MCMNetworkDescription::MCMNetworkDescription(const std::vector<char>& compiledNetwork, const Config& config,
                                             const std::string& name)
        : _name(name), _compiledNetwork(compiledNetwork), _logger("MCMNetworkDescription", config.get<LOG_LEVEL>()) {
    IE_ASSERT(compiledNetwork.data() != nullptr);

    const auto* graphFilePtr = MVCNN::GetGraphFile(compiledNetwork.data());
    IE_ASSERT(graphFilePtr != nullptr);
    const auto graphHeader = graphFilePtr->header();
    IE_ASSERT(graphHeader != nullptr);
    const auto metaInfo = MCMAdapter::deserializeMetaData(*graphHeader, config);
    const ie::InputsDataMap& deserializedInputs = metaInfo._inputs;
    const ie::OutputsDataMap& deserializedOutputs = metaInfo._outputs;
    const std::string& networkName = metaInfo._networkName;
    if (deserializedInputs.empty()) {
        IE_THROW() << "MCMNetworkDescription: meta-data does not contain inputs.";
    }

    if (deserializedOutputs.empty()) {
        IE_THROW() << "MCMNetworkDescription: meta-data does not contain outputs.";
    }

    // FIXME: the code below does matching of actual device in/outs with meta data to give
    // the device in/outs proper names and to be able identify them.
    // It can be avoided if compiler does not change in/outs names, passed by a user
    // S#34832
    const auto graphInputs = graphHeader->net_input();
    IE_ASSERT(graphInputs != nullptr);
    const auto deviceInputs = MCMAdapter::getNetworkInputs(*graphInputs);
    _deviceInputs = inputsDataMapToDataMap(deviceInputs);
    const auto inputsNames = extractKeys(deserializedInputs);
    _deviceInputs = createDeviceMapWithCorrectNames(_deviceInputs, inputsNames);

    const auto graphOutputs = graphHeader->net_output();
    IE_ASSERT(graphOutputs != nullptr);
    const auto deviceOutputs = MCMAdapter::getNetworkOutputs(*graphOutputs);
    _deviceOutputs = outputsDataMapToDataMap(deviceOutputs);
    const auto outputsNames = extractKeys(deserializedOutputs);
    _deviceOutputs = createDeviceMapWithCorrectNames(_deviceOutputs, outputsNames);

    const auto graphProfilingOutputs = graphHeader->profiling_output();
    if (graphProfilingOutputs != nullptr) {
        const auto deviceProfilingOutputs = MCMAdapter::getNetworkOutputs(*graphProfilingOutputs);
        _deviceProfilingOutputs = outputsDataMapToDataMap(deviceProfilingOutputs);
    }

    _networkInputs = inputsDataMapToDataMap(deserializedInputs);
    _networkOutputs = outputsDataMapToDataMap(deserializedOutputs);

    _quantParams = metaInfo._quantParams;

    // network name is preferable
    // override default name 'net#' if flatbuffer contains the name
    if (!networkName.empty()) {
        _name = networkName;
    }

    // TODO: it makes sense to print maps here under log level
}

const DataMap& MCMNetworkDescription::getInputsInfo() const {
    return _networkInputs;
}

const DataMap& MCMNetworkDescription::getOutputsInfo() const {
    return _networkOutputs;
}

const DataMap& MCMNetworkDescription::getDeviceInputsInfo() const {
    return _deviceInputs;
}

const DataMap& MCMNetworkDescription::getDeviceOutputsInfo() const {
    return _deviceOutputs;
}

const DataMap& MCMNetworkDescription::getDeviceProfilingOutputsInfo() const {
    return _deviceProfilingOutputs;
}

const std::vector<OVRawNode>& MCMNetworkDescription::getOVParameters() const {
    return _ovParameters;
}

const std::vector<OVRawNode>& MCMNetworkDescription::getOVResults() const {
    return _ovResults;
}

const QuantizationParamMap& MCMNetworkDescription::getQuantParamsInfo() const {
    return _quantParams;
}

const std::vector<char>& MCMNetworkDescription::getCompiledNetwork() const {
    return _compiledNetwork;
}

const void* MCMNetworkDescription::getNetworkModel() const {
    return _compiledNetwork.data();
}

std::size_t MCMNetworkDescription::getNetworkModelSize() const {
    return _compiledNetwork.size();
}

const std::string& MCMNetworkDescription::getName() const {
    return _name;
}

DataMap MCMNetworkDescription::matchElementsByName(const DataMap& actualDeviceData,
                                                   const std::vector<std::string>& names) {
    _logger.debug("MCMNetworkDescription::matchElementsByName started.");
    DataMap updatedMap;

    // Copy original device outputs. Once the output name is matched it will be removed from the list //
    // The rest will be copied and return with original name //
    DataMap actualDeviceDataLocal = actualDeviceData;

    for (const auto& name : names) {
        bool isNameFound = false;
        for (const auto& data : actualDeviceDataLocal) {
            if (data.first.find(name) != std::string::npos) {
                const auto dataCorrectedName = data.second;
                dataCorrectedName->setName(name);
                updatedMap.insert({name, dataCorrectedName});
                isNameFound = true;
                _logger.debug("Matched '{0}' with '{1}'", name, data.first);
                actualDeviceDataLocal.erase(data.first);
                break;
            }
        }
        if (!isNameFound) {
            _logger.warning("Cannot match actual output names with device names.");
            updatedMap.clear();
            break;
        }
    }

    if (updatedMap.size() != 0) {
        for (const auto& data : actualDeviceDataLocal) {
            std::string name = data.first;
            const auto dataCorrectedName = data.second;
            dataCorrectedName->setName(name);
            updatedMap.insert({name, dataCorrectedName});
            _logger.debug("Added '{0}'", name);
        }
    }

    _logger.debug("MCMNetworkDescription::matchElementsByName finished.");
    return updatedMap;
}

DataMap MCMNetworkDescription::matchElementsByLexicographicalOrder(const DataMap& actualDeviceData,
                                                                   const std::vector<std::string>& names) {
    _logger.debug("MCMNetworkDescription::matchElementsByLexicographicalOrder started.");
    DataMap updatedMap;

    if (names.empty()) {
        // FIXME fail more gracefully
        IE_THROW() << "matchElementsByLexicographicalOrder meta-data does not contain names.";
    }

    std::size_t curMatchPos = 0;
    for (const auto& data : actualDeviceData) {
        std::string name;
        if (names.size() > curMatchPos)
            name = names[curMatchPos];
        else {
            name = data.first;
            _logger.info("Additional output: {0}", name);
        }
        const auto dataCorrectedName = data.second;
        dataCorrectedName->setName(name);
        updatedMap.insert({name, dataCorrectedName});
        _logger.debug("Matched '{0}' with '{1}'", name, data.first);
        curMatchPos++;
    }

    _logger.debug("MCMNetworkDescription::matchElementsByLexicographicalOrder finished.");
    return updatedMap;
}

DataMap MCMNetworkDescription::createDeviceMapWithCorrectNames(const DataMap& actualDeviceData,
                                                               const std::vector<std::string>& names) {
    DataMap updatedMap;

    updatedMap = matchElementsByName(actualDeviceData, names);
    if (updatedMap.empty()) {
        updatedMap = matchElementsByLexicographicalOrder(actualDeviceData, names);
    }

    return updatedMap;
}
