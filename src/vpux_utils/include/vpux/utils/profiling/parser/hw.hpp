//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Profiling HW interface

#pragma once

#include <cstdint>

struct ActShaveData_t {
    uint64_t begin;
    uint32_t duration;
    uint32_t stallCycles;
    uint32_t executedInstructions;
    uint32_t clockCycles;
    uint32_t branchTaken;
    uint32_t reserved32;
};
static_assert(sizeof(ActShaveData_t) == 32);

struct ActShaveDataEx_t {
    uint64_t begin;
    uint32_t duration;
    uint32_t executedInstructions;
    uint32_t clockCycles;
    uint32_t lsu0Stalls;
    uint32_t lsu1Stalls;
    uint32_t instStalls;
};
static_assert(sizeof(ActShaveData_t) == sizeof(ActShaveDataEx_t));

// HWP DPU profiling data payload
struct HwpDpu27Mode0Data_t {
    uint64_t idu_wl_duration : 28;
    uint64_t idu_tstamp : 28;
    uint64_t sve_id : 5;
    uint64_t reserved3 : 3;
    uint64_t odu_wl_duration : 28;
    uint64_t odu_tstamp : 28;
    uint64_t reserved8 : 8;
};
static_assert(sizeof(HwpDpu27Mode0Data_t) == 16);

struct HwpDma40Data_t {
    uint64_t desc_addr;
    uint64_t fetch_time;
    uint64_t ready_time;
    uint64_t start_time;
    uint64_t wdone_time;
    uint64_t finish_time;
    uint8_t la_id;
    uint8_t ch_id;
    uint16_t rsvd;
    uint16_t rstall_cnt;
    uint16_t wstall_cnt;
    uint32_t twbytes_cnt;
    uint32_t chcycle_cnt;
};
static_assert(sizeof(HwpDma40Data_t) == 64);

struct HwpDpuIduOduData_t {
    uint64_t idu_wl_duration : 32;
    uint64_t idu_wl_id : 16;
    uint64_t idu_dpu_id : 16;
    uint64_t idu_tstamp;
    uint64_t odu_wl_duration : 32;
    uint64_t odu_wl_id : 16;
    uint64_t odu_dpu_id : 16;
    uint64_t odu_tstamp;
};
static_assert(sizeof(HwpDpuIduOduData_t) == 32);

struct DMA27Data_t {
    uint64_t startCycle;
    uint64_t endCycle;
};

struct WorkpointConfiguration_t {
    uint16_t pllMultiplier;
    uint16_t configId;
};
