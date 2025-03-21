//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/reports/api.hpp"

#include "vpux/utils/core/env.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/reports/tasklist.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

using namespace vpux;

namespace vpux::profiling {

namespace {

enum class ProfilingFormat { NONE, JSON, TEXT, RAW };

std::string capitalize(const std::string& str) {
    std::string capStr(str);
    std::transform(capStr.begin(), capStr.end(), capStr.begin(), ::toupper);
    return capStr;
}

std::string getProfilingFileNameEnv(ProfilingFormat format) {
    const auto filename = env::getEnvVar("NPU_PROFILING_OUTPUT_FILE");
    if (filename.has_value()) {
        return filename.value();
    }
    switch (format) {
    case ProfilingFormat::JSON:
        return "profiling.json";
    case ProfilingFormat::TEXT:
        return "profiling.txt";
    default:
        return "profiling.out";
    }
}

VerbosityLevel getProfilingVerbosityEnv() {
    const auto verbosity = capitalize(env::getEnvVar("NPU_PROFILING_VERBOSITY", "LOW"));
    if (verbosity == "LOW") {
        return VerbosityLevel::LOW;
    } else if (verbosity == "MEDIUM") {
        return VerbosityLevel::MEDIUM;
    } else if (verbosity == "HIGH") {
        return VerbosityLevel::HIGH;
    }
    VPUX_THROW("Invalid NPU_PROFILING_VERBOSITY value");
}

ProfilingFormat getProfilingFormatEnv() {
    const auto profilingMode = env::getEnvVar("NPU_PRINT_PROFILING");
    if (!profilingMode.has_value()) {
        return ProfilingFormat::NONE;
    }
    const auto format = capitalize(profilingMode.value());
    if (format == "JSON") {
        return ProfilingFormat::JSON;
    } else if (format == "TEXT") {
        return ProfilingFormat::TEXT;
    } else if (format == "RAW") {
        return ProfilingFormat::RAW;
    }
    VPUX_THROW("Invalid NPU_PRINT_PROFILING value");
}

std::ofstream openProfilingStream(ProfilingFormat format) {
    std::ofstream output;
    if (format == ProfilingFormat::NONE) {
        return output;
    }
    const auto outFileName = getProfilingFileNameEnv(format);
    auto flags = std::ios::out | std::ios::trunc;
    if (format == ProfilingFormat::RAW) {
        flags |= std::ios::binary;
    }
    output.open(outFileName, flags);
    if (!output) {
        VPUX_THROW("Can't write into file '{0}'", outFileName);
    }
    output.exceptions(std::ios::badbit | std::ios::failbit);
    return output;
}

void saveProfilingDataToFile(std::ostream& output, ProfilingFormat format, const ProfInfo& profInfo) {
    switch (format) {
    case ProfilingFormat::JSON:
        printProfilingAsTraceEvent(profInfo.tasks, profInfo.layers, profInfo.dpuFreq, output);
        break;
    case ProfilingFormat::TEXT:
        printProfilingAsText(profInfo.tasks, profInfo.layers, output);
        break;
    case ProfilingFormat::RAW:
    case ProfilingFormat::NONE:
        VPUX_THROW("Unsupported profiling format");
    }
}

void saveRawDataToFile(const uint8_t* rawBuffer, size_t size, std::ostream& output) {
    output.write(reinterpret_cast<const char*>(rawBuffer), size);
    output.flush();
}

}  // namespace

std::vector<LayerInfo> getLayerProfilingInfoHook(const uint8_t* profData, size_t profSize, const uint8_t* blobData,
                                                 size_t blobSize) {
    const auto format = getProfilingFormatEnv();
    auto output = openProfilingStream(format);
    if (format == ProfilingFormat::RAW) {
        // Save raw data first in case post-processing fails
        saveRawDataToFile(profData, profSize, output);
    }
    auto verbosity = getProfilingVerbosityEnv();
    auto profInfo = getProfInfo(blobData, blobSize, profData, profSize, verbosity);
    if (format != ProfilingFormat::NONE && format != ProfilingFormat::RAW) {
        saveProfilingDataToFile(output, format, profInfo);
    }
    return profInfo.layers;
}

std::vector<LayerInfo> getLayerProfilingInfoHook(const std::vector<uint8_t>& data, const std::vector<uint8_t>& blob) {
    return getLayerProfilingInfoHook(data.data(), data.size(), blob.data(), blob.size());
}

std::vector<LayerInfo> getLayerProfilingInfoHook(const uint8_t* profData, size_t profSize,
                                                 const std::vector<uint8_t>& blob) {
    return getLayerProfilingInfoHook(profData, profSize, blob.data(), blob.size());
}

}  // namespace vpux::profiling
