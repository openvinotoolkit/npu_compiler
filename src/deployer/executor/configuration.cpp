#include "include/mcm/deployer/executor/configuration.hpp"

namespace mv
{
    Configuration::Configuration(std::shared_ptr<mv::RuntimeBinary> binaryPointer):
        target_(mv::Target::Unknown), //unknown == any device is good.
        protocol_(Protocol::USB_VSC),
        inputMode_(InputMode::ALL_ZERO),
        binaryPointer_(binaryPointer)
    {
    }
    Configuration::Configuration(std::shared_ptr<mv::RuntimeBinary> binaryPointer,
        Target target, Protocol protocol,
        InputMode inputMode, const std::string& inputFilePath): binaryPointer_(binaryPointer)
    {
        setTarget(target);
        setProtocol(protocol);
        setInputMode(inputMode);
        setInputFilePath(inputFilePath);
    }
    Configuration::Configuration(const Configuration &c):
        target_(c.target_),
        protocol_(c.protocol_),
        inputMode_(c.inputMode_),
        inputFilePath_(c.inputFilePath_),
        binaryPointer_(c.binaryPointer_)
    {
    }
    void Configuration::setTarget(Target target)
    {
        target_ = target;
    }
    void Configuration::setProtocol(Protocol protocol)
    {
        if (protocol == Protocol::Unknown)
            throw ArgumentError(*this, "protocol", "unknown", "Defining protocol as unknown is illegal");
        protocol_ = protocol;
    }
    void Configuration::setInputMode(InputMode inputMode)
    {
        if (inputMode == InputMode::Unknown)
            throw ArgumentError(*this, "inputMode", "unknown", "Defining inputMode as unknown is illegal");
        inputMode_ = inputMode;
    }

    void Configuration::setInputFilePath(const std::string& inputFilePath)
    {
        if (inputFilePath.empty())
            throw ArgumentError(*this, "inputFilePath", "Empty", "Defining inputFilePath as empty path is illegal");
        inputFilePath_ = inputFilePath;
    }

    Target Configuration::getTarget() const
    {
        return target_;
    }
    Protocol Configuration::getProtocol() const
    {
        return protocol_;
    }
    InputMode Configuration::getInputMode() const
    {
        return inputMode_;
    }
    std::string Configuration::getInputFilePath() const
    {
        return inputFilePath_;
    }
    std::string Configuration::getLogID() const
    {
        return "Configuration";
    }
    std::shared_ptr<mv::RuntimeBinary> Configuration::getRuntimePointer() const
    {
        return binaryPointer_;
    }

}
