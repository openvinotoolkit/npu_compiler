//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/utils/logger/logger.hpp"
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/utils/core/developer_build_utils.hpp"
#endif

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <mutex>

using namespace vpux;

//
// LogCb
//

void vpux::emptyLogCb(const formatv_object_base&) {
}

void vpux::globalLogCb(const formatv_object_base& msg) {
    Logger::global().trace("{0}", msg.str());
}

//
// Logger
//
static const char* logLevelPrintout[] = {"NONE", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"};

Logger& vpux::Logger::global() {
    static Logger globalLogger = []() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
        LogLevel logLvl = LogLevel::Warning;
        if (const auto env = std::getenv("OV_NPU_LOG_LEVEL")) {
            auto logStr = std::string(env);
            if (logStr == "LOG_NONE") {
                logLvl = LogLevel::None;
            } else if (logStr == "LOG_ERROR") {
                logLvl = LogLevel::Error;
            } else if (logStr == "LOG_WARNING") {
                logLvl = LogLevel::Warning;
            } else if (logStr == "LOG_INFO") {
                logLvl = LogLevel::Info;
            } else if (logStr == "LOG_DEBUG") {
                logLvl = LogLevel::Debug;
            } else if (logStr == "LOG_TRACE") {
                logLvl = LogLevel::Trace;
            }
        }
        return Logger("global", logLvl);
#else
        return Logger("global", LogLevel::None);
#endif
    }();

    return globalLogger;
}

vpux::Logger::Logger(StringLiteral name, LogLevel lvl): _name(name), _logLevel(lvl) {
}

Logger vpux::Logger::nest(size_t inc) const {
    return nest(name(), inc);
}

Logger vpux::Logger::nest(StringLiteral name, size_t inc) const {
    Logger nested(name, level());
    nested._indentLevel = _indentLevel + inc;
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    nested._alternateName = _alternateName;
#endif
    return nested;
}

Logger vpux::Logger::unnest(size_t inc) const {
    assert(_indentLevel >= inc);
    Logger unnested(name(), level());
    unnested._indentLevel = _indentLevel - inc;
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    unnested._alternateName = _alternateName;
#endif
    return unnested;
}

bool vpux::Logger::isActive(LogLevel msgLevel) const {
#if !defined(NDEBUG)
    if (llvm::DebugFlag && llvm::isCurrentDebugType(name().data())) {
        return true;
    }
#endif

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    // Global filter - read once from environment, shared by all Logger instances
    static const auto logFilter = []() -> llvm::Regex {
        const auto logFilterEnv = std::getenv("IE_NPU_LOG_FILTER");
        const auto logFilterEnabled = logFilterEnv != nullptr && logFilterEnv[0] != '\0';

        if (isPerfDebugMode()) {
            if (logFilterEnabled) {
                std::string combinedFilter = std::string(logFilterEnv) + "|" + perfDebugDefaultLogFilter.str();
                return llvm::Regex(combinedFilter, llvm::Regex::IgnoreCase);
            }
            return llvm::Regex(perfDebugDefaultLogFilter.str(), llvm::Regex::IgnoreCase);
        }
        if (logFilterEnabled) {
            return llvm::Regex(logFilterEnv, llvm::Regex::IgnoreCase);
        }
        return {};
    }();

    if (static_cast<int32_t>(msgLevel) > static_cast<int32_t>(_logLevel)) {
        return false;
    }

    // If filter is set, check if it matches _name or _alternateName
    // Supports both CamelCase (e.g., "OptimizeCopies") and dash-separated (e.g., "optimize-copies")
    if (logFilter.isValid()) {
        if (logFilter.match(_name)) {
            return true;
        }
        // Check alternate name (e.g., pass name in CamelCase vs dash-separated argument)
        if (!_alternateName.empty() && logFilter.match(_alternateName)) {
            return true;
        }
        // Filter is set but doesn't match either name - return false
        return false;
    }

    return true;
#else
    return static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(_logLevel);
#endif
}

llvm::raw_ostream& vpux::Logger::getBaseStream() {
    if (_baseStreamOverride) {
        return *_baseStreamOverride;
    }
#ifdef NDEBUG
    return llvm::outs();
#else
    return llvm::DebugFlag ? llvm::dbgs() : llvm::outs();
#endif
}

void vpux::Logger::setBaseStream(llvm::raw_ostream& stream) {
    _baseStreamOverride = &stream;
}

namespace {

llvm::raw_ostream::Colors getColor(LogLevel msgLevel) {
    switch (msgLevel) {
    case LogLevel::Fatal:
    case LogLevel::Error:
        return llvm::raw_ostream::RED;
    case LogLevel::Warning:
        return llvm::raw_ostream::YELLOW;
    case LogLevel::Info:
        return llvm::raw_ostream::CYAN;
    case LogLevel::Debug:
    case LogLevel::Trace:
        return llvm::raw_ostream::GREEN;
    default:
        return llvm::raw_ostream::SAVEDCOLOR;
    }
}

}  // namespace

llvm::WithColor vpux::Logger::getLevelStream(LogLevel msgLevel) {
    const auto color = getColor(msgLevel);
    return llvm::WithColor(getBaseStream(), color, true, false, llvm::ColorMode::Auto);
}

void vpux::Logger::addEntryPackedActive(LogLevel msgLevel, const formatv_object_base& msg) const {
    llvm::SmallString<512> tempBuf;
    llvm::raw_svector_ostream tempStream(tempBuf);

    char timeStr[] = "undefined_time";
    time_t now = time(nullptr);
    {
        // localtime returns a pointer to a shared static struct tm and may allocate/free the TZ
        // string buffer internally, so both the call and the read of its result must be serialized.
        static std::mutex localtimeMtx;
        std::lock_guard<std::mutex> localtimeMtxLock(localtimeMtx);
        struct tm* loctime = localtime(&now);
        if (loctime != nullptr) {
            strftime(timeStr, sizeof(timeStr), "%H:%M:%S", loctime);
        }
    }

    using namespace std::chrono;
    uint32_t ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() % 1000;

    printTo(tempStream, "[{0}] {1}.{2,0+3} [{3}] ", logLevelPrintout[static_cast<uint8_t>(msgLevel)], timeStr, ms,
            _name);

    for (size_t i = 0; i < _indentLevel; ++i) {
        tempStream << "  ";
    }

    msg.format(tempStream);
    tempStream << "\n";

    static std::mutex logMtx;
    std::lock_guard<std::mutex> logMtxLock(logMtx);

    auto colorStream = getLevelStream(msgLevel);
    auto& stream = colorStream.get();
    stream << tempStream.str();
    stream.flush();
}
