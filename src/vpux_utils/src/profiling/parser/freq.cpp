//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/compiler/NPU37XX/dialect/config/constraints.hpp"
#include "vpux/compiler/NPU40XX/dialect/config/constraints.hpp"
#include "vpux/utils/profiling/parser/parser.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <utility>

using namespace vpux::profiling;

namespace {

FrequenciesSetup getFreqSetupForDevice(TargetDevice device, uint16_t pllMult, FreqStatus freqStatus,
                                       bool highFreqPerfClk, vpux::Logger& log) {
    uint16_t pllMultMin = 1;
    uint16_t pllMultMax = 0;
    uint16_t pllMultDefault = 0;
    double perfClk = vpux::arch40xx::PERF_CLK_DEFAULT_VALUE_MHZ;
    double perfClkFast = vpux::arch40xx::PERF_CLK_HIGHFREQ_VALUE_MHZ;
    bool hasSharedDmaDpuCounter = true;

    switch (device) {
    case TargetDevice::TargetDevice_VPUX37XX:
        pllMultMin = 10;
        pllMultMax = 48;
        pllMultDefault = 39;  // 975 / 1300 MHz
        perfClk = vpux::arch37xx::PERF_CLK_DEFAULT_VALUE_MHZ;
        perfClkFast = 0.0;  // not supported
        hasSharedDmaDpuCounter = false;
        break;
    case TargetDevice::TargetDevice_VPUX40XX:
    case TargetDevice::TargetDevice_VPUX50XX:
        pllMultMin = 8;
        pllMultMax = 78;
        pllMultDefault = 74;  // 1057 / 1850 MHz
        break;
    case TargetDevice::TargetDevice_VPUX60XX:
        pllMultMin = 43;
        pllMultMax = 101;
        pllMultDefault = 84;  // 1200 / 2100 MHz
        break;
    default:
        VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(device));
    }

    if (pllMult < pllMultMin || pllMult > pllMultMax) {
        log.warning("PLL multiplier '{0}' is out of range. Default frequency will be used.", pllMult);
        freqStatus = FreqStatus::INVALID;
        pllMult = pllMultDefault;
    }
    VPUX_THROW_WHEN(highFreqPerfClk && perfClkFast == 0.0,
                    "High frequency perf clock is not supported on this device.");

    double base = 50.0 * pllMult;
    double vpuFreq = 0.0;
    double dpuFreq = 0.0;
    if (device == TargetDevice::TargetDevice_VPUX37XX) {
        vpuFreq = base / 2.0;
        dpuFreq = base / 1.5;
    } else {
        vpuFreq = base / 3.5;
        dpuFreq = base / 2.0;
    }

    return FrequenciesSetup{vpuFreq, dpuFreq, highFreqPerfClk ? perfClkFast : perfClk, hasSharedDmaDpuCounter,
                            freqStatus};
}

FrequenciesSetup getFpgaFreqSetup(TargetDevice device) {
    VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(device));
}

FrequenciesSetup getFreqSetupFromPll(TargetDevice device, const WorkpointRecords& workpoints, bool highFreqPerfClk,
                                     vpux::Logger& log) {
    uint16_t pllMult = 0;
    FreqStatus freqStatus = FreqStatus::VALID;

    if (workpoints.empty()) {
        log.warning("No frequency data");
        freqStatus = FreqStatus::UNKNOWN;
    } else {
        // PLL value from the beginning of inference
        const auto pllMultFirst = workpoints.front().first.pllMultiplier;
        // PLL value from the end of inference
        const auto pllMultLast = workpoints.back().first.pllMultiplier;
        pllMult = pllMultFirst;
        log.trace("Got PLL value '{0}' [{1}]", pllMult, to_string(freqStatus));
        if (pllMultFirst != pllMultLast) {
            freqStatus = FreqStatus::INVALID;
            log.warning("Frequency changed during the inference!");
        }
    }

    return getFreqSetupForDevice(device, pllMult, freqStatus, highFreqPerfClk, log);
}

}  // namespace

namespace vpux::profiling {

FrequenciesSetup getFrequencySetup(const TargetDevice device, const WorkpointRecords& workpoints, bool highFreqPerfClk,
                                   bool fpga, vpux::Logger& log) {
    FrequenciesSetup frequenciesSetup;

    if (fpga) {
        frequenciesSetup = getFpgaFreqSetup(device);
    } else {
        frequenciesSetup = getFreqSetupFromPll(device, workpoints, highFreqPerfClk, log);
    }
    log.trace("Frequency setup is profClk={0}MHz, vpuClk={1}MHz, dpuClk={2}MHz [{3}]", frequenciesSetup.profClk,
              frequenciesSetup.vpuClk, frequenciesSetup.dpuClk, to_string(frequenciesSetup.clockStatus));

    return frequenciesSetup;
}

const char* to_string(FreqStatus freqStatus) {
    switch (freqStatus) {
    case FreqStatus::UNKNOWN:
        return "UNKNOWN";
    case FreqStatus::VALID:
        return "OK";
    case FreqStatus::INVALID:
        return "INVALID";
    case FreqStatus::SIM:
        return "SIM";
    }
    return NULL;
}

}  // namespace vpux::profiling
