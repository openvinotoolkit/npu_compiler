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

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vpu/kmb_plugin_config.hpp>
#include <cpp_interfaces/exception2status.hpp>

#include "kmb_config.h"

using namespace vpu::KmbPlugin;

KmbConfig::KmbConfig(const std::map<std::string, std::string> &config, ConfigMode mode)
    : ParsedConfigBase(mode)  {
    _parsedConfig = parse(config);
    configure(_parsedConfig);
}

std::map<std::string, std::string> KmbConfig::getDefaultConfig() const {
    auto defaultVpuConfig = ParsedConfigBase::getDefaultConfig();
    std::map<std::string, std::string> kmbSpecific = {
#ifdef NDEBUG
            {CONFIG_KEY(LOG_LEVEL),                                 CONFIG_VALUE(LOG_NONE)},
#else
            {CONFIG_KEY(LOG_LEVEL),                                 CONFIG_VALUE(LOG_DEBUG)},
#endif
#ifdef ENABLE_VPUAL
            {VPU_KMB_CONFIG_KEY(KMB_EXECUTOR),                      CONFIG_VALUE(YES)},
#else
            {VPU_KMB_CONFIG_KEY(KMB_EXECUTOR),                      CONFIG_VALUE(NO)},
#endif
            {VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH),        "config/target"},
            {VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR),             "ma2490"},
            {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH),   "config/compilation"},
            {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR),             "debug_ma2490"},
            {VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB),                 CONFIG_VALUE(YES)},
            {VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON),                 CONFIG_VALUE(YES)},
            {VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT),                  CONFIG_VALUE(NO)},
            {VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY),                  CONFIG_VALUE(NO)},
            {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH),      "."},
            {VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS),           ""},
            {VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS),                "1"},
    };
    defaultVpuConfig.insert(kmbSpecific.begin(), kmbSpecific.end());
    return defaultVpuConfig;
}

void KmbConfig::checkInvalidValues(const std::map<std::string, std::string> &config) const {
    ParsedConfigBase::checkInvalidValues(config);
}

std::unordered_set<std::string> KmbConfig::getCompileOptions() const {
    std::unordered_set<std::string> compileOptions = {
            VPU_KMB_CONFIG_KEY(KMB_EXECUTOR),
            VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR_PATH),
            VPU_KMB_CONFIG_KEY(MCM_TARGET_DESCRIPTOR),
            VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR_PATH),
            VPU_KMB_CONFIG_KEY(MCM_COMPILATION_DESCRIPTOR),
            VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB),
            VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON),
            VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT),
            VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY),
            VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH),
            VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS),
            VPU_KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
        };

    auto parentCompileOptions = ParsedConfigBase::getCompileOptions();
    compileOptions.insert(parentCompileOptions.begin(), parentCompileOptions.end());

    return compileOptions;
}
