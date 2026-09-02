//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/openvino.hpp>

#include <vpux/utils/logger/logger.hpp>
#include <vpux/utils/ov/config.hpp>
#include <vpux/utils/ov/options.hpp>

#include <chrono>
#include <type_traits>
#include <utility>

namespace vpux {

static inline bool isInfoOrDefaultLogLevel(const vpux::OV::Config& config) {
    return !config.has<vpux::OV::LOG_LEVEL>() || config.get<vpux::OV::LOG_LEVEL>() == LogLevel::Info;
}

template <class Callable>
class ScopedTimer {
public:
    using fp_milliseconds = std::chrono::duration<double, std::chrono::milliseconds::period>;

    explicit ScopedTimer(const vpux::OV::Config& config, Callable&& callable)
            : _enabled(isInfoOrDefaultLogLevel(config)), _callable(std::forward<Callable>(callable)) {
        if (_enabled) {
            start();
        }
    }

    ~ScopedTimer() noexcept {
        try {
            if (_enabled) {
                stop();
                _callable(deltaMs());
            }
        } catch (const std::exception& e) {
            vpux::Logger::global().error("Exception in ScopedTimer destructor: {0}", e.what());
        } catch (...) {
            vpux::Logger::global().error("Unknown exception in ScopedTimer destructor");
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

    void start() {
        _startTime = std::chrono::steady_clock::now();
    }

    void stop() {
        _stopTime = std::chrono::steady_clock::now();
    }

    double deltaMs() const {
        return std::chrono::duration_cast<fp_milliseconds>(_stopTime - _startTime).count();
    }

private:
    bool _enabled;
    Callable _callable;
    std::chrono::steady_clock::time_point _startTime;
    std::chrono::steady_clock::time_point _stopTime;
};

template <class Callable>
auto startScopedTimer(const vpux::OV::Config& config, Callable&& callable) {
    return ScopedTimer<std::decay_t<Callable>>(config, std::forward<Callable>(callable));
}

}  // namespace vpux
