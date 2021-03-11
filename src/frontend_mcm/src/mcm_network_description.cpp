#include "mcm_network_description.hpp"

#include "blob_parser.hpp"
#include "mcm_adapter.hpp"

using namespace vpu::MCMAdapter;
namespace ie = InferenceEngine;

namespace {
// TODO: remove once parser works with DataMap instead of inputs and outputs info
vpux::DataMap inputsDataMapToDataMap(const InferenceEngine::InputsDataMap& inputs) {
    vpux::DataMap dataMap = {};
    for (auto&& in : inputs) {
        dataMap.insert({in.first, in.second->getInputData()});
    }

    return dataMap;
}
vpux::DataMap outputsDataMapToDataMap(const InferenceEngine::OutputsDataMap& outputs) {
    vpux::DataMap dataMap = {};
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

MCMNetworkDescription::MCMNetworkDescription(const std::vector<char>& compiledNetwork, const vpu::MCMConfig& config,
                                             const std::string& name)
        : _name(name),
          _compiledNetwork(compiledNetwork),
          _logger(std::make_shared<vpu::Logger>("MCMNetworkDescription", config.logLevel(), consoleOutput())) {
    MetaInfo metaInfo = MCMAdapter::deserializeMetaData(compiledNetwork, config);
    const ie::InputsDataMap& deserializedInputs = metaInfo._inputs;
    const ie::OutputsDataMap& deserializedOutputs = metaInfo._outputs;
    const std::string& networkName = metaInfo._networkName;
    if (deserializedInputs.empty()) {
        THROW_IE_EXCEPTION << "MCMNetworkDescription: meta-data does not contain inputs.";
    }

    if (deserializedOutputs.empty()) {
        THROW_IE_EXCEPTION << "MCMNetworkDescription: meta-data does not contain outputs.";
    }

    // FIXME: the code below does matching of actual device in/outs with meta data to give
    // the device in/outs proper names and to be able identify them.
    // It can be avoided if compiler does not change in/outs names, passed by a user
    // S#34832
    ie::InputsDataMap deviceInputs;
    MCMAdapter::getNetworkInputs(compiledNetwork.data(), deviceInputs);
    _deviceInputs = inputsDataMapToDataMap(deviceInputs);
    auto inputsNames = extractKeys(deserializedInputs);
    _deviceInputs = createDeviceMapWithCorrectNames(_deviceInputs, inputsNames);

    ie::OutputsDataMap deviceOutputs;
    MCMAdapter::getNetworkOutputs(compiledNetwork.data(), deviceOutputs);
    _deviceOutputs = outputsDataMapToDataMap(deviceOutputs);
    auto outputsNames = extractKeys(deserializedOutputs);
    _deviceOutputs = createDeviceMapWithCorrectNames(_deviceOutputs, outputsNames);

    _networkInputs = inputsDataMapToDataMap(deserializedInputs);
    _networkOutputs = outputsDataMapToDataMap(deserializedOutputs);

    // network name is preferable
    // override default name 'net#' if flatbuffer contains the name
    if (!networkName.empty()) {
        _name = networkName;
    }

    // TODO: it makes sense to print maps here under log level
}

const vpux::DataMap& MCMNetworkDescription::getInputsInfo() const {
    return _networkInputs;
}

const vpux::DataMap& MCMNetworkDescription::getOutputsInfo() const {
    return _networkOutputs;
}

const vpux::DataMap& MCMNetworkDescription::getDeviceInputsInfo() const {
    return _deviceInputs;
}

const vpux::DataMap& MCMNetworkDescription::getDeviceOutputsInfo() const {
    return _deviceOutputs;
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

vpux::DataMap MCMNetworkDescription::matchElementsByName(const vpux::DataMap& actualDeviceData,
                                                         const std::vector<std::string>& names) {
    _logger->debug("MCMNetworkDescription::matchElementsByName started.");
    vpux::DataMap updatedMap;

    for (const auto& name : names) {
        bool isNameFound = false;
        for (const auto& data : actualDeviceData) {
            if (data.first.find(name) != std::string::npos) {
                const auto dataCorrectedName = data.second;
                dataCorrectedName->setName(name);
                updatedMap.insert({name, dataCorrectedName});
                isNameFound = true;
                _logger->debug("Matched \'%s\' with \'%s'\\n", name, data.first);
            }
        }
        if (!isNameFound) {
            _logger->warning("Cannot match actual output names with device names.\n");
            updatedMap.clear();
            break;
        }
    }

    _logger->debug("MCMNetworkDescription::matchElementsByName finished.");
    return updatedMap;
}

vpux::DataMap MCMNetworkDescription::matchElementsByLexicographicalOrder(const vpux::DataMap& actualDeviceData,
                                                                         const std::vector<std::string>& names) {
    _logger->debug("MCMNetworkDescription::matchElementsByLexicographicalOrder started.");
    vpux::DataMap updatedMap;

    if (names.empty()) {
        // FIXME fail more gracefully
        THROW_IE_EXCEPTION << "matchElementsByLexicographicalOrder meta-data does not contain names.";
    }

    std::size_t curMatchPos = 0;
    for (const auto& data : actualDeviceData) {
        std::string name;
        if (names.size() > curMatchPos)
            name = names[curMatchPos];
        else {
            name = data.first;
            _logger->info("Additional output: %s\n", name);
        }
        const auto dataCorrectedName = data.second;
        dataCorrectedName->setName(name);
        updatedMap.insert({name, dataCorrectedName});
        _logger->debug("Matched \'%s\' with \'%s'\\n", name, data.first);
        curMatchPos++;
    }

    _logger->debug("MCMNetworkDescription::matchElementsByLexicographicalOrder finished.");
    return updatedMap;
}

vpux::DataMap MCMNetworkDescription::createDeviceMapWithCorrectNames(const vpux::DataMap& actualDeviceData,
                                                                     const std::vector<std::string>& names) {
    vpux::DataMap updatedMap;

    updatedMap = matchElementsByName(actualDeviceData, names);
    if (updatedMap.empty()) {
        updatedMap = matchElementsByLexicographicalOrder(actualDeviceData, names);
    }

    return updatedMap;
}
