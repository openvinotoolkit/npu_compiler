//
// Copyright (C) 2022-2025 Intel Corporation.
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

template <bool SHARED_DMA_SW_CNT, bool SHARED_DMA_DPU_CNT>
constexpr FrequenciesSetup getFreqSetupHelper(const double vpuFreq, const double dpuFreq, double profClk,
                                              FreqStatus freqStatus) {
    return FrequenciesSetup{vpuFreq, dpuFreq, profClk, SHARED_DMA_SW_CNT, SHARED_DMA_DPU_CNT, freqStatus};
}

constexpr auto getFreqSetup37XXHelper = getFreqSetupHelper<true, false>;
constexpr auto getFreqSetup40XXHelper = getFreqSetupHelper<true, true>;

FrequenciesSetup get37XXSetup(uint16_t pllMult, FreqStatus freqStatus, vpux::Logger& log) {
    if (pllMult < 10 || pllMult > 48) {
        log.warning("PLL multiplier '{0}' is out of [10; 42] range. MAX freq. setup will be used.", pllMult);
        freqStatus = FreqStatus::INVALID;
        pllMult = 39;  // 975 / 1300 MHz
    }
    const double base = 50.0 * pllMult;
    const double vpuFreq = base / 2.0;
    const double dpuFreq = base / 1.5;
    return getFreqSetup37XXHelper(vpuFreq, dpuFreq, vpux::arch37xx::PERF_CLK_DEFAULT_VALUE_MHZ, freqStatus);
}

FrequenciesSetup get40XXSetup(uint16_t pllMult, FreqStatus freqStatus, bool highFreqPerfClk, vpux::Logger& log) {
    if (pllMult < 8 || pllMult > 78) {
        log.warning("PLL multiplier '{0}' is out of [8; 78] range. MAX freq. setup will be used.", pllMult);
        freqStatus = FreqStatus::INVALID;
        pllMult = 74;  // 1057 / 1850 MHz
    }
    const double base = 50.0 * pllMult;
    const double vpuFreq = base / 3.5;
    const double dpuFreq = base / 2.0;
    return getFreqSetup40XXHelper(
            vpuFreq, dpuFreq,
            highFreqPerfClk ? vpux::arch40xx::PERF_CLK_HIGHFREQ_VALUE_MHZ : vpux::arch40xx::PERF_CLK_DEFAULT_VALUE_MHZ,
            freqStatus);
}

FrequenciesSetup getFpgaFreqSetup(TargetDevice device) {
    if (device >= TargetDevice::TargetDevice_VPUX40XX) {
        return getFreqSetup40XXHelper(2.86, 5.0, 1.176, FreqStatus::UNKNOWN);
    }
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

    if (device == TargetDevice::TargetDevice_VPUX37XX) {
        VPUX_THROW_WHEN(highFreqPerfClk, "Requested perf_clk high frequency value is not supported on this device.");
        return get37XXSetup(pllMult, freqStatus, log);
    }
    if (device >= TargetDevice::TargetDevice_VPUX40XX) {
        return get40XXSetup(pllMult, freqStatus, highFreqPerfClk, log);
    }
    VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(device));
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
