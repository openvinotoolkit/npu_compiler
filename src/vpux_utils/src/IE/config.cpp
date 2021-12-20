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

#include "vpux/utils/IE/config.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <ie_plugin_config.hpp>

using namespace vpux;
using namespace InferenceEngine;

//
// OptionParser
//

bool vpux::OptionParser<bool>::parse(StringRef val) {
    if (val == CONFIG_VALUE(YES)) {
        return true;
    } else if (val == CONFIG_VALUE(NO)) {
        return false;
    }

    VPUX_THROW("Value '{0}' is not a valid BOOL option");
}

int64_t OptionParser<int64_t>::parse(StringRef val) {
    try {
        return std::stoll(val.str());
    } catch (...) {
        VPUX_THROW("Value '{0}' is not a valid INT64 option", val);
    }
}

double OptionParser<double>::parse(StringRef val) {
    try {
        return std::stod(val.str());
    } catch (...) {
        VPUX_THROW("Value '{0}' is not a valid FP64 option", val);
    }
}

LogLevel vpux::OptionParser<LogLevel>::parse(StringRef val) {
    if (val == CONFIG_VALUE(LOG_NONE)) {
        return LogLevel::None;
    } else if (val == CONFIG_VALUE(LOG_ERROR)) {
        return LogLevel::Error;
    } else if (val == CONFIG_VALUE(LOG_WARNING)) {
        return LogLevel::Warning;
    } else if (val == CONFIG_VALUE(LOG_INFO)) {
        return LogLevel::Info;
    } else if (val == CONFIG_VALUE(LOG_DEBUG)) {
        return LogLevel::Debug;
    } else if (val == CONFIG_VALUE(LOG_TRACE)) {
        return LogLevel::Trace;
    }

    VPUX_THROW("Value '{0}' is not a valid LOG_LEVEL option");
}

//
// OptionMode
//

StringLiteral vpux::stringifyEnum(OptionMode val) {
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

vpux::details::OptionValue::~OptionValue() = default;

//
// OptionsDesc
//

vpux::details::OptionConcept vpux::OptionsDesc::get(StringRef key, OptionMode mode) const {
    auto log = Logger::global().nest("OptionsDesc", 0);

    auto searchKey = key;

    const auto itDeprecated = _deprecated.find(key);
    if (itDeprecated != _deprecated.end()) {
        searchKey = itDeprecated->second;
        log.warning("Deprecated option '{0}' was used, '{1}' should be used instead", key, searchKey);
    }

    const auto itMain = _impl.find(searchKey);
    VPUX_THROW_WHEN(itMain == _impl.end(), "{0} Option '{1}' is not supported for current configuration",
                    InferenceEngine::details::ExceptionTraits<InferenceEngine::NotFound>::string(), key);

    const auto& desc = itMain->second;

    if (mode == OptionMode::RunTime) {
        if (desc.mode() == OptionMode::CompileTime) {
            log.warning("{0} option '{1}' was used in {2} mode", desc.mode(), key, mode);
        }
    }

    return desc;
}

std::vector<std::string> vpux::OptionsDesc::getSupported(bool includePrivate) const {
    std::vector<std::string> res;
    res.reserve(_impl.size());

    for (const auto& p : _impl) {
        if (p.second.isPublic() || includePrivate) {
            res.push_back(p.first.str());
        }
    }

    return res;
}

void vpux::OptionsDesc::walk(FuncRef<void(const details::OptionConcept&)> cb) const {
    for (const auto& opt : _impl | map_values) {
        cb(opt);
    }
}

//
// Config
//

vpux::Config::Config(const std::shared_ptr<const OptionsDesc>& desc): _desc(desc) {
    VPUX_THROW_WHEN(_desc == nullptr, "Got NULL OptionsDesc");
}

#ifdef VPUX_DEVELOPER_BUILD
void vpux::Config::parseEnvVars() {
    auto log = Logger::global().nest("Config", 0);

    _desc->walk([&](const details::OptionConcept& opt) {
        if (!opt.envVar().empty()) {
            if (const auto envVar = std::getenv(opt.envVar().data())) {
                log.trace("Update option '{0}' to value '{1}' parsed from environment variable '{2}'", opt.key(),
                          envVar, opt.envVar());

                _impl[opt.key()] = opt.validateAndParse(envVar);
            }
        }
    });
}
#endif

void vpux::Config::update(const ConfigMap& options, OptionMode mode) {
    auto log = Logger::global().nest("Config", 0);

    for (const auto& p : options) {
        log.trace("Update option '{0}' to value '{1}'", p.first, p.second);

        const auto opt = _desc->get(p.first, mode);
        _impl[opt.key()] = opt.validateAndParse(p.second);
    }
}

bool vpux::envVarStrToBool(const char* varName, const char* varValue) {
    try {
        const auto intVal = std::stoi(varValue);
        if (intVal != 0 && intVal != 1) {
            throw std::invalid_argument("Only 0 and 1 values are supported");
        }
        return (intVal != 0);
    } catch (const std::exception& e) {
        IE_THROW() << "Environment variable " << varName << " has wrong value : " << e.what();
    }
}
