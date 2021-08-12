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

#pragma once
#include <details/ie_so_loader.h>

#include <cstddef>
#include <set>

#include <details/ie_so_pointer.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_remote_context.hpp>
#include <vpux_config.hpp>
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

namespace vpux {

// Input name and surface FP quantization flag
using QuantizationParamMap = std::unordered_map<std::string, bool>;

///////////////////////////////////// INetworkDescription /////////////////////////////////////////
using DataMap = std::map<std::string, InferenceEngine::DataPtr>;
class INetworkDescription : public std::enable_shared_from_this<INetworkDescription> {
public:
    using Ptr = std::shared_ptr<INetworkDescription>;
    using CPtr = std::shared_ptr<const INetworkDescription>;

    virtual const std::string& getName() const = 0;
    virtual const DataMap& getInputsInfo() const = 0;
    virtual const DataMap& getOutputsInfo() const = 0;
    virtual const DataMap& getDeviceInputsInfo() const = 0;
    virtual const DataMap& getDeviceOutputsInfo() const = 0;
    virtual const vpux::QuantizationParamMap getQuantParamsInfo() const {
        return {};
    }

    // TODO Remove interface returning std::vector<char>.
    /**
     * @deprecated Return type should follow the function below.
     * The name itself can be reused once the old return type is dropped.
     */
    virtual const std::vector<char>& getCompiledNetwork() const = 0;
    virtual const void* getNetworkModel() const = 0;
    virtual std::size_t getNetworkModelSize() const = 0;
    virtual ~INetworkDescription() = default;
};

///////////////////////////////////// NetworkDescription //////////////////////////////////////////
class NetworkDescription final {
public:
    using Ptr = std::shared_ptr<NetworkDescription>;
    using CPtr = std::shared_ptr<const NetworkDescription>;

    NetworkDescription(INetworkDescription::Ptr actual, const InferenceEngine::details::SharedObjectLoader& plg = {});

    const std::string& getName() const {
        return _actual->getName();
    }
    const DataMap& getInputsInfo() const {
        return _actual->getInputsInfo();
    }
    const DataMap& getOutputsInfo() const {
        return _actual->getOutputsInfo();
    }
    const DataMap& getDeviceInputsInfo() const {
        return _actual->getDeviceInputsInfo();
    }
    const DataMap& getDeviceOutputsInfo() const {
        return _actual->getDeviceOutputsInfo();
    }
    const vpux::QuantizationParamMap getQuantParamsInfo() const {
        return _actual->getQuantParamsInfo();
    }
    const std::vector<char>& getCompiledNetwork() const {
        return _actual->getCompiledNetwork();
    }
    const void* getNetworkModel() const {
        return _actual->getNetworkModel();
    }
    std::size_t getNetworkModelSize() const {
        return _actual->getNetworkModelSize();
    }

    ~NetworkDescription() {
        _actual = nullptr;
    }

private:
    INetworkDescription::Ptr _actual = nullptr;
    // NB: NetworkDescription is created by Compiler which is object from shared library,
    // so it has to keep pointer to this lib in case Compiler lifecycle less than objects which created by it
    InferenceEngine::details::SharedObjectLoader _plg;
};

//////////////////////////////////////////ICompiler ///////////////////////////////////////////////
class ICompiler : public std::enable_shared_from_this<ICompiler> {
public:
    using Ptr = std::shared_ptr<ICompiler>;
    using CPtr = std::shared_ptr<const ICompiler>;

    virtual std::shared_ptr<INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                         const std::string& netName,
                                                         const InferenceEngine::InputsDataMap& inputsInfo,
                                                         const InferenceEngine::OutputsDataMap& outputsInfo,
                                                         const VPUXConfig& config = {}) = 0;

    virtual InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                                      const VPUXConfig& config = {}) = 0;

    virtual std::shared_ptr<vpux::INetworkDescription> parse(const std::vector<char>& network,
                                                             const VPUXConfig& config = {},
                                                             const std::string& graphName = "") = 0;

    virtual std::shared_ptr<vpux::INetworkDescription> parse(const std::string& filename,
                                                             const VPUXConfig& config = {});
    virtual std::shared_ptr<vpux::INetworkDescription> parse(std::istream& stream, const VPUXConfig& config = {},
                                                             const std::string& graphName = "");

    virtual std::unordered_set<std::string> getSupportedOptions() {
        return {};
    };

protected:
    ~ICompiler() = default;
};

//////////////////////////////////////////Compiler ////////////////////////////////////////////////
class Compiler final {
public:
    using Ptr = std::shared_ptr<Compiler>;
    using CPtr = std::shared_ptr<const Compiler>;

    static Ptr create(const VPUXConfig& config = {});

    Compiler(std::string libpath): _actual(std::move(libpath)){};

    std::shared_ptr<vpux::NetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                      const std::string& netName,
                                                      const InferenceEngine::InputsDataMap& inputsInfo,
                                                      const InferenceEngine::OutputsDataMap& outputsInfo,
                                                      const vpux::VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(_actual->compile(func, netName, inputsInfo, outputsInfo, config),
                                                    _actual);
    }

    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                              const vpux::VPUXConfig& config = {}) {
        return _actual->query(network, config);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::vector<char>& network,
                                                    const vpux::VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(_actual->parse(network, config), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(const std::string& filename, const VPUXConfig& config = {}) {
        return std::make_shared<NetworkDescription>(_actual->parse(filename, config), _actual);
    }

    std::shared_ptr<vpux::NetworkDescription> parse(std::istream& stream, const VPUXConfig& config = {},
                                                    const std::string& graphName = "") {
        return std::make_shared<NetworkDescription>(_actual->parse(stream, config, graphName), _actual);
    }

    std::unordered_set<std::string> getSupportedOptions() {
        return _actual->getSupportedOptions();
    };

private:
    using CompilerPluginPtr = InferenceEngine::details::SOPointer<vpux::ICompiler>;
    CompilerPluginPtr _actual;
};

namespace helpers {
InferenceEngine::InputsDataMap dataMapIntoInputsDataMap(const vpux::DataMap& dataMap);
InferenceEngine::OutputsDataMap dataMapIntoOutputsDataMap(const vpux::DataMap& dataMap);
}  // namespace helpers

}  // namespace vpux

namespace InferenceEngine {
namespace details {
template <>
class SOCreatorTrait<vpux::ICompiler> {
public:
    static constexpr auto name = "CreateVPUXCompiler";
};
}  // namespace details
}  // namespace InferenceEngine
