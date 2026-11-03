//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/runtime/properties.hpp>

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/ov/config.hpp"

namespace vpux {

namespace OV {

// Splits the `str` string onto separate elements using `delim` as delimiter and
// call `callback` for each element.
void splitAndApply(const std::string& str, char delim, std::function<void(std::string_view)> callback) {
    const auto begin = str.begin();
    const auto end = str.end();

    auto curBegin = begin;
    auto curEnd = begin;
    while (curEnd != end) {
        while (curEnd != end && *curEnd != delim) {
            ++curEnd;
        }

        callback(std::string_view(&(*curBegin), static_cast<size_t>(curEnd - curBegin)));

        if (curEnd != end) {
            ++curEnd;
            curBegin = curEnd;
        }
    }
}

//
// OptionParser
//

bool OptionParser<bool>::parse(std::string_view val) {
    std::string strVal(val);
    std::transform(strVal.begin(), strVal.end(), strVal.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    if (strVal == "YES" || strVal == "TRUE" || strVal == "1") {
        return true;
    } else if (strVal == "NO" || strVal == "FALSE" || strVal == "0") {
        return false;
    }

    VPUX_THROW("Value '{0}' is not a valid BOOL option", val.data());
}

int32_t OptionParser<int32_t>::parse(std::string_view val) {
    int32_t value;
    const auto result = std::from_chars(val.data(), val.data() + val.size(), value);
    if (result.ec == std::errc()) {
        return value;
    }
    VPUX_THROW("Value '{0}' is not a valid INT32 option", val.data());
}

uint32_t OptionParser<uint32_t>::parse(std::string_view val) {
    uint32_t value;
    const auto result = std::from_chars(val.data(), val.data() + val.size(), value);
    if (result.ec == std::errc()) {
        return value;
    }
    VPUX_THROW("Value '{0}' is not a valid UINT32 option", val.data());
}

int64_t OptionParser<int64_t>::parse(std::string_view val) {
    int64_t value;
    const auto result = std::from_chars(val.data(), val.data() + val.size(), value);
    if (result.ec == std::errc()) {
        return value;
    }
    VPUX_THROW("Value '{0}' is not a valid INT64 option", val.data());
}

uint64_t OptionParser<uint64_t>::parse(std::string_view val) {
    uint64_t value;
    const auto result = std::from_chars(val.data(), val.data() + val.size(), value);
    if (result.ec == std::errc()) {
        return value;
    }
    VPUX_THROW("Value '{0}' is not a valid UINT64 option", val.data());
}

double OptionParser<double>::parse(std::string_view val) {
    double value;
    const auto result = std::from_chars(val.data(), val.data() + val.size(), value);
    if (result.ec == std::errc()) {
        return value;
    }
    VPUX_THROW("Value '{0}' is not a valid FP64 option", val.data());
}

LogLevel OptionParser<LogLevel>::parse(std::string_view val) {
    std::string strVal(val);
    std::transform(strVal.begin(), strVal.end(), strVal.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    if (strVal == "NONE") {
        return LogLevel::None;
    } else if (strVal == "FATAL") {
        return LogLevel::Fatal;
    } else if (strVal == "ERROR") {
        return LogLevel::Error;
    } else if (strVal == "WARNING") {
        return LogLevel::Warning;
    } else if (strVal == "INFO") {
        return LogLevel::Info;
    } else if (strVal == "DEBUG") {
        return LogLevel::Debug;
    } else if (strVal == "TRACE") {
        return LogLevel::Trace;
    }

    VPUX_THROW("Value '{0}' is not a valid LOG_LEVEL option", val.data());
}

//
// OptionPrinter
//

std::string OptionPrinter<bool>::toString(bool val) {
    return val ? "YES" : "NO";
}

//
// OptionMode
//

std::string_view stringifyEnum(OptionMode val) {
    switch (val) {
    case OptionMode::Both:
        return "Both";
    case OptionMode::CompileTime:
        return "CompileTime";
    case OptionMode::RunTime:
        return "RunTime";
    default:
        return "<UNKNOWN>";
    }
}

//
// OptionValue
//

details::OptionValue::~OptionValue() = default;

//
// OptionsDesc
//

details::OptionConcept OptionsDesc::get(std::string_view key) const {
    std::string searchKey{key};
    const auto itDeprecated = _deprecated.find(std::string(key));
    if (itDeprecated != _deprecated.end()) {
        searchKey = itDeprecated->second;
        _log.warning("Deprecated option '{0}' was used, '{1}' should be used instead", key.data(), searchKey.c_str());
    }

    const auto itMain = _impl.find(searchKey);
    VPUX_THROW_WHEN(itMain == _impl.end(), "[ NOT_FOUND ] Option '{0}' is not supported for current configuration",
                    key.data());

    return itMain->second;
}

void OptionsDesc::reset() {
    _impl.clear();
    _deprecated.clear();
}

bool OptionsDesc::has(std::string_view key) const {
    std::string searchKey{key};
    const auto itDeprecated = _deprecated.find(searchKey);
    if (itDeprecated != _deprecated.end()) {
        return true;
    }
    const auto itMain = _impl.find(searchKey);
    if (itMain != _impl.end()) {
        return true;
    }
    return false;
}

std::vector<std::string> OptionsDesc::getSupported(bool includePrivate) const {
    std::vector<std::string> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res.push_back(p.first.data());
        }
    }

    return res;
}

std::vector<ov::PropertyName> OptionsDesc::getSupportedOptions(bool includePrivate) const {
    std::vector<ov::PropertyName> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res.push_back({p.first.data(), p.second.mutability()});
        }
    }

    return res;
}

std::string OptionsDesc::getSupportedAsString(bool includePrivate) const {
    std::string res;
    bool first = true;
    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            if (!first) {
                res += " ";
            }
            res += p.first;
            first = false;
        }
    }

    return res;
}

void OptionsDesc::walk(std::function<void(const details::OptionConcept&)> cb) const {
    for (const auto& itr : _impl) {
        cb(itr.second);
    }
}

//
// Config
//

Config::Config(const std::shared_ptr<const OptionsDesc>& desc): _desc(desc) {
    VPUX_THROW_WHEN(_desc == nullptr, "Got NULL OptionsDesc");
}

void Config::parseEnvVars() {
    _desc->walk([&](const details::OptionConcept& opt) {
        if (!opt.envVar().empty()) {
            if (const auto envVar = std::getenv(opt.envVar().data())) {
                _log.trace("Update option '{0}' to value '{1}' parsed from environment variable '{2}'",
                           opt.key().data(), envVar, opt.envVar().data());

                try {
                    _impl[opt.key().data()] = opt.validateAndParseFromString(envVar);
                } catch (const std::exception& e) {
                    _log.warning("Environment variable '{0}' with value '{1}' was ignored for option '{2}' due to "
                                 "error:\n{3}",
                                 opt.envVar().data(), envVar, opt.key().data(), e.what());
                }
            }
        }
    });
}

bool Config::has(const std::string& key) const {
    return _impl.count(key) != 0;
}

void Config::update(const ConfigMap& options) {
    for (const auto& p : options) {
        _log.trace("Update option '{0}' to value '{1}'", p.first.c_str(), p.second.c_str());

        const auto opt = _desc->get(p.first);
        _impl[opt.key().data()] = opt.validateAndParseFromString(p.second);
    }
}

void Config::updateAny(const ov::AnyMap& options) {
    for (const auto& p : options) {
        _log.trace("Update option '{0}' to given 'ov::Any' value", p.first.c_str());

        const auto opt = _desc->get(p.first);
        _impl[opt.key().data()] = opt.validateAndParseFromAny(p.second);
    }
}

void Config::erase(const std::string& key) {
    _log.trace("Erase option '{0}'", key.c_str());

    const auto opt = _desc->get(key);
    _impl.erase(opt.key());
}

std::string Config::toString() const {
    std::stringstream resultStream;
    for (auto it = _impl.cbegin(); it != _impl.cend(); ++it) {
        const auto& key = it->first;

        // include only enabled configs
        resultStream << key << "=\"" << it->second->toString() << "\"";
        if (std::next(it) != _impl.end()) {
            resultStream << " ";
        }
    }

    return resultStream.str();
}

Config::ConfigMap parseConfigString(std::string_view str) {
    std::vector<std::string> options;
    std::stringstream input{std::string(str)};

    /// A singleOption is consist of one or more words, like value of NPU_COMPILATION_MODE_PARAMS
    std::string word;
    std::string singleOption = "";
    while (input >> word) {
        if (singleOption.compare("") == 0) {
            /// Save the first word
            singleOption = word;
        } else {
            /// Save the word that belongs to this option
            singleOption = singleOption + " " + word;
        }

        /// If this is not the last word of this option, contine for this option
        if (word[word.size() - 1] != '"') {
            continue;
        }

        /// Save current option
        options.push_back(singleOption);
        /// Clean for the next option
        singleOption = "";
    }
    /// Save the last option
    if (singleOption.compare("") != 0) {
        options.push_back(std::move(singleOption));
    }

    Config::ConfigMap config;
    for (auto& option : options) {
        if (option.find_first_not_of(' ') == std::string::npos) {
            continue;
        }
        size_t length = option.size();
        /// Skip the terminator of string
        if (option[length - 1] == '\0') {
            length--;
        }

        std::size_t lastDelimPos = option.find_first_of('=');
        /// Use 2 to skip =" , the format shall follow key="value"
        if (lastDelimPos == std::string::npos || lastDelimPos + 2 > length) {
            throw std::logic_error(option + " is in bad format!");
        }
        std::string key = option.substr(0, lastDelimPos);
        /// For key="value", the val shall be value
        /// Skip =" in the front and " at the end
        config[key] = option.substr(lastDelimPos + 2, length - 1 - (lastDelimPos + 2));
    }

    return config;
}

void Config::fromString(const std::string& str) {
    update(parseConfigString(str));
}

//
// envVarStrToBool
//

bool envVarStrToBool(const char* varName, const char* varValue) {
    try {
        const auto intVal = std::stoi(varValue);
        if (intVal != 0 && intVal != 1) {
            throw std::invalid_argument("Only 0 and 1 values are supported");
        }
        return (intVal != 0);
    } catch (const std::exception& e) {
        VPUX_THROW("Environment variable '{0}' has wrong value : {1}", varName, e.what());
    }
}

}  // namespace OV

LogLevel getLogLevel(ov::log::Level level) {
    switch (level) {
    case ov::log::Level::NO:
        return LogLevel::None;
    case ov::log::Level::ERR:
        return LogLevel::Error;
    case ov::log::Level::WARNING:
        return LogLevel::Warning;
    case ov::log::Level::INFO:
        return LogLevel::Info;
    case ov::log::Level::DEBUG:
        return LogLevel::Debug;
    case ov::log::Level::TRACE:
        return LogLevel::Trace;
    }
    // Should not happen unless the enum is extended
    OPENVINO_THROW("Invalid log level.");
}

}  // namespace vpux
