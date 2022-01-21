//
// Copyright Intel Corporation.
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

#pragma once

#include "vpux_compiler.hpp"

namespace vpux {
namespace zeroCompilerAdapter {

/**
 * @brief Network blob + meta information
 */
class NetworkDescription final : public INetworkDescription {
public:
    explicit NetworkDescription(const std::vector<char>& compiledNetwork, const std::string& name,
                                const DataMap& networkInputs, const DataMap& networkOutputs,
                                const DataMap& deviceInputs, const DataMap& deviceOutputs)
            : _compiledNetwork(compiledNetwork),
              _name(name),
              _networkInputs(networkInputs),
              _networkOutputs(networkOutputs),
              _deviceInputs(deviceInputs),
              _deviceOutputs(deviceOutputs) {
    }

    explicit NetworkDescription(
            const std::vector<char>& compliedNetwork, const std::string graphName,
            const std::tuple<const DataMap, const DataMap, const DataMap, const DataMap>& networkMeta)
            : NetworkDescription(compliedNetwork, graphName, std::get<0>(networkMeta), std::get<1>(networkMeta),
                                 std::get<2>(networkMeta), std::get<3>(networkMeta)) {
    }

    explicit NetworkDescription(void* handle, const std::string& name, const DataMap& networkInputs,
                                const DataMap& networkOutputs, const DataMap& deviceInputs,
                                const DataMap& deviceOutputs)
            : _handle(handle),
              _name(name),
              _networkInputs(networkInputs),
              _networkOutputs(networkOutputs),
              _deviceInputs(deviceInputs),
              _deviceOutputs(deviceOutputs) {
    }

    explicit NetworkDescription(void* handle, const std::tuple<const std::string, const DataMap, const DataMap,
                                                               const DataMap, const DataMap>& networkMeta)
            : NetworkDescription(handle, std::get<0>(networkMeta), std::get<1>(networkMeta), std::get<2>(networkMeta),
                                 std::get<3>(networkMeta), std::get<4>(networkMeta)) {
    }

public:
    const vpux::QuantizationParamMap& getQuantParamsInfo() const final {
        return _quantParams;
    }

    const std::vector<char>& getCompiledNetwork() const final {
        return _compiledNetwork;
    }

    const void* getNetworkModel() const final {
        return _handle;
    }

    std::size_t getNetworkModelSize() const final {
        return _compiledNetwork.size();
    }

    const std::string& getName() const final {
        return _name;
    }

    const DataMap& getInputsInfo() const final {
        return _networkInputs;
    }
    const DataMap& getOutputsInfo() const final {
        return _networkOutputs;
    }

    const DataMap& getDeviceInputsInfo() const final {
        return _deviceInputs;
    }
    const DataMap& getDeviceOutputsInfo() const final {
        return _deviceOutputs;
    }
    const DataMap& getDeviceProfilingOutputsInfo() const override {
        return _profilingOutputs;
    }

    const std::vector<vpux::OVRawNode>& getOVParameters() const override {
        return _ovParameters;
    }
    const std::vector<vpux::OVRawNode>& getOVResults() const override {
        return _ovResults;
    }

private:
    void* _handle;
    std::vector<char> _compiledNetwork;

    std::string _name;

    DataMap _networkInputs;
    DataMap _networkOutputs;

    DataMap _deviceInputs;
    DataMap _deviceOutputs;

    // TODO Not implemented
    DataMap _profilingOutputs;

    // TODO Not implemented
    vpux::QuantizationParamMap _quantParams{};

    // TODO Not implemented
    const std::vector<vpux::OVRawNode> _ovParameters;
    const std::vector<vpux::OVRawNode> _ovResults;
};

}  // namespace zeroCompilerAdapter
}  // namespace vpux
