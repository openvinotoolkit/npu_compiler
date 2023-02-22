//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Class for pretty-logging.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/Support/WithColor.h>

#include <utility>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace vpux {

//
// LogLevel
//

// Logger verbosity levels

enum class LogLevel {
    None = 0,     // Logging is disabled
    Fatal = 1,    // Used for very severe error events that will most probably
                  // cause the application to terminate
    Error = 2,    // Reporting events which are not expected during normal
                  // execution, containing probable reason
    Warning = 3,  // Indicating events which are not usual and might lead to
                  // errors later
    Info = 4,     // Short enough messages about ongoing activity in the process
    Debug = 5,    // More fine-grained messages with references to particular data
                  // and explanations
    Trace = 6,    // Involved and detailed information about execution, helps to
                  // trace the execution flow, produces huge output
};

StringLiteral stringifyEnum(LogLevel val);

//
// Logging callback
//

using LogCb = FuncRef<void(const formatv_object_base&)>;
void emptyLogCb(const formatv_object_base&);
void globalLogCb(const formatv_object_base&);

//
// Logger
//

class Logger final {
public:
    static Logger& global();

public:
    explicit Logger(StringLiteral name, LogLevel lvl);

public:
    Logger nest(size_t inc = 1) const;
    Logger nest(StringLiteral name, size_t inc = 1) const;
    Logger unnest(size_t inc = 1) const;

public:
    auto name() const {
        return _name;
    }

    void setName(StringLiteral name) {
        _name = name;
    }

public:
    auto level() const {
        return _logLevel;
    }

    void setLevel(LogLevel lvl) {
        _logLevel = lvl;
    }

    bool isActive(LogLevel msgLevel) const;

public:
    static llvm::raw_ostream& getBaseStream();
    static llvm::WithColor getLevelStream(LogLevel msgLevel);

public:
    template <typename... Args>
    void fatal(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Fatal, formatv(format.data(), std::forward<Args>(args)...));
    }

    template <typename... Args>
    void error(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Error, formatv(format.data(), std::forward<Args>(args)...));
    }

    template <typename... Args>
    void warning(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Warning, formatv(format.data(), std::forward<Args>(args)...));
    }

    template <typename... Args>
    void info(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Info, formatv(format.data(), std::forward<Args>(args)...));
    }

    template <typename... Args>
    void debug(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Debug, formatv(format.data(), std::forward<Args>(args)...));
    }

    template <typename... Args>
    void trace(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Trace, formatv(format.data(), std::forward<Args>(args)...));
    }

public:
    template <typename... Args>
    void addEntry(LogLevel msgLevel, StringLiteral format, Args&&... args) const {
        addEntryPacked(msgLevel, formatv(format.data(), std::forward<Args>(args)...));
    }

private:
    void addEntryPacked(LogLevel msgLevel, const formatv_object_base& msg) const;

private:
    StringLiteral _name;
    LogLevel _logLevel = LogLevel::None;
    size_t _indentLevel = 0;
};

}  // namespace vpux
