//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
    TargetDevice_MAX
};

enum TargetDeviceRevision {
    TargetDeviceRevision_NONE = 0,
    TargetDeviceRevision_A0 = 1,
    TargetDeviceRevision_B0 = 2,
    TargetDeviceRevision_MIN = TargetDeviceRevision_NONE,
    TargetDeviceRevision_MAX = TargetDeviceRevision_B0
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
    if (IsOutRange(e, TargetDevice_NONE, TargetDevice_VPUX311X))
        return "";
    const size_t index = static_cast<size_t>(e);
    return EnumNamesTargetDevice()[index];
}

inline const char* const* EnumNamesTargetDeviceRevision() {
    static const char* const names[4] = {"NONE", "A0", "B0", nullptr};
    return names;
}

inline const char* EnumNameTargetDeviceRevision(TargetDeviceRevision e) {
    if (IsOutRange(e, TargetDeviceRevision_NONE, TargetDeviceRevision_B0))
        return "";
    const size_t index = static_cast<size_t>(e);
    return EnumNamesTargetDeviceRevision()[index];
}

enum PhysicalProcessor {
    PhysicalProcessor_NULL = 0,
    PhysicalProcessor_LEON_RT = 1,
    PhysicalProcessor_LEON_NN = 2,
    PhysicalProcessor_NN_SHV = 3,
    PhysicalProcessor_ARM = 4,
    PhysicalProcessor_NCE_Cluster = 5,
    PhysicalProcessor_NCE_PerClusterDPU = 6,
    PhysicalProcessor_MIN = PhysicalProcessor_NULL,
    PhysicalProcessor_MAX = PhysicalProcessor_NCE_PerClusterDPU
};

}  // namespace vpux::profiling
