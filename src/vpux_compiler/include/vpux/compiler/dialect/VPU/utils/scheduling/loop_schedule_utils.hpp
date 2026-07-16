//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::VPU {

/// Generate predefined loop schedules for all compute regions.
/// Runs temporal tiling strategy on each region that has a recognized loop type
/// and collects the results into a ComputeRegionSchedule.
/// This function must be called before constructing FeasibleMemoryScheduler.
///
/// @param loopRegions                Compute regions extracted from async operations
/// @param memorySize                 Total available CMX memory size
/// @param enableVfUndefinedScheduler TODO: E#216928 - remove flag when VF region scheduler feature fully enabled.
///                                   When false, VF regions are skipped here and handled by the legacy scheduler path.
/// @param log                        Logger instance
/// @return ComputeRegionSchedule containing predefined schedules and operation index sets
ComputeRegionsSchedule generateLoopSchedules(const ComputeRegionVec& loopRegions, vpux::AddressType memorySize,
                                             bool enableVfUndefinedScheduler, Logger log);

}  // namespace vpux::VPU
