//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/logger.hpp"

#include "vpux/utils/core/optional.hpp"

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
// LogLevel
//

StringLiteral vpux::stringifyEnum(LogLevel val) {
    switch (val) {
    case LogLevel::None:
        return "None";
    case LogLevel::Fatal:
        return "Fatal";
    case LogLevel::Error:
        return "Error";
    case LogLevel::Warning:
        return "Warning";
    case LogLevel::Info:
        return "Info";
    case LogLevel::Debug:
        return "Debug";
    case LogLevel::Trace:
        return "Trace";
    default:
        return "<UNKNOWN>";
    }
}

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

Logger& vpux::Logger::global() {
#ifdef VPUX_DEVELOPER_BUILD
    static Logger log("global", LogLevel::Warning);
#else
    static Logger log("global", LogLevel::None);
#endif

    return log;
}

vpux::Logger::Logger(StringLiteral name, LogLevel lvl): _name(name), _logLevel(lvl) {
}

Logger vpux::Logger::nest(size_t inc) const {
    return nest(name(), inc);
}

Logger vpux::Logger::nest(StringLiteral name, size_t inc) const {
    Logger nested(name, level());
    nested._indentLevel = _indentLevel + inc;
    return nested;
}

Logger vpux::Logger::unnest(size_t inc) const {
    assert(_indentLevel >= inc);
    Logger unnested(name(), level());
    unnested._indentLevel = _indentLevel - inc;
    return unnested;
}

bool vpux::Logger::isActive(LogLevel msgLevel) const {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    static const auto logFilter = []() -> llvm::Regex {
        if (const auto env = std::getenv("IE_VPUX_LOG_FILTER")) {
            const StringRef filter(env);

            if (!filter.empty()) {
                return llvm::Regex(filter, llvm::Regex::IgnoreCase);
            }
        }

        return {};
    }();

    if (logFilter.isValid() && logFilter.match(_name)) {
        return true;
    }
#endif

#ifdef NDEBUG
    return static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(_logLevel);
#else
    return (static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(_logLevel)) ||
           (llvm::DebugFlag && llvm::isCurrentDebugType(name().data()));
#endif
}

llvm::raw_ostream& vpux::Logger::getBaseStream() {
#ifdef NDEBUG
    return llvm::outs();
#else
    return llvm::DebugFlag ? llvm::dbgs() : llvm::outs();
#endif
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

void vpux::Logger::addEntryPacked(LogLevel msgLevel, const formatv_object_base& msg) const {
    if (!isActive(msgLevel)) {
        return;
    }

    llvm::SmallString<512> tempBuf;
    llvm::raw_svector_ostream tempStream(tempBuf);

    time_t now = time(nullptr);
    struct tm tstruct;
    char timeStr[10];
    struct tm* tstruct_ptr = localtime(&now);
    if (tstruct_ptr != NULL) {
        tstruct = *tstruct_ptr;
    }
    strftime(timeStr, sizeof(timeStr), "%H:%M:%S", &tstruct);

    using namespace std::chrono;
    uint32_t ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() % 1000;

    printTo(tempStream, "{0}.{1,0+3} [{2}]", timeStr, ms, _name);

    for (size_t i = 0; i < _indentLevel; ++i)
        tempStream << "  ";

    msg.format(tempStream);
    tempStream << "\n";

    static std::mutex logMtx;
    std::lock_guard<std::mutex> logMtxLock(logMtx);

    auto colorStream = getLevelStream(msgLevel);
    auto& stream = colorStream.get();
    stream << tempStream.str();
    stream.flush();
}
