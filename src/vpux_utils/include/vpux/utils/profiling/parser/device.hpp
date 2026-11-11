//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux::profiling {

enum TargetDevice {
    TargetDevice_NONE = 0,
    TargetDevice_MIN = TargetDevice_NONE,
    TargetDevice_VPUX30XX = 1,
    TargetDevice_VPUX37XX = 2,
    TargetDevice_VPUX311X = 3,
    TargetDevice_VPUX40XX = 4,
    TargetDevice_VPUX50XX = 5,
    TargetDevice_VPUX60XX = 6,
    TargetDevice_MAX
};

template <typename T>
inline bool IsOutRange(const T& v, const T& low, const T& high) {
    return (v < low) || (high < v);
}

inline const char* const* EnumNamesTargetDevice() {
    static const char* const names[5] = {"NONE", "VPUX30XX", "VPUX37XX", "VPUX311X", nullptr};
    return names;
}

inline const char* EnumNameTargetDevice(TargetDevice e) {
    if (IsOutRange(e, TargetDevice_NONE, TargetDevice_VPUX311X)) {
        return "";
    }
    const size_t index = static_cast<size_t>(e);
    return EnumNamesTargetDevice()[index];
}

}  // namespace vpux::profiling
