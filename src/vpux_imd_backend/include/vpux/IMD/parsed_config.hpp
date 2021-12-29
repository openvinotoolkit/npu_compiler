//
// Copyright Intel Corporation.
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

#pragma once

#include "vpux/IMD/private_config.hpp"

#include "vpux/utils/IE/config.hpp"

namespace vpux {
namespace IMD {

//
// MV_TOOLS_PATH
//

struct MV_TOOLS_PATH final : OptionBase<MV_TOOLS_PATH, std::string> {
    static StringRef key() {
        return VPUX_IMD_CONFIG_KEY(MV_TOOLS_PATH);
    }

    static StringRef envVar() {
        return "IE_VPUX_MV_TOOLS_PATH";
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// LAUNCH_MODE
//

enum class LaunchMode {
    MoviSim,
};

StringLiteral stringifyEnum(LaunchMode val);

struct LAUNCH_MODE final : OptionBase<LAUNCH_MODE, LaunchMode> {
    static StringRef key() {
        return VPUX_IMD_CONFIG_KEY(LAUNCH_MODE);
    }

    static StringRef envVar() {
        return "IE_VPUX_IMD_LAUNCH_MODE";
    }

    static LaunchMode parse(StringRef val);

    static LaunchMode defaultValue() {
        return LaunchMode::MoviSim;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

}  // namespace IMD
}  // namespace vpux