//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @PrintParseConfigureBarrier
func.func @PrintParseConfigureBarrier() -> () {
    %bar0 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    %bar1 = VPURT.ConfigureBarrier<1> <{isStartBarrier}> -> !VPURT.Barrier
    %bar2 = VPURT.ConfigureBarrier<1> <{isFinalBarrier}> -> !VPURT.Barrier
    %bar3 = VPURT.ConfigureBarrier<1> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.ConfigureBarrier<1> <{barrierIndex = 2}> -> !VPURT.Barrier
    %bar5 = VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64, barrierIndex = 1}> -> !VPURT.Barrier

    return

    // CHECK: VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK: VPURT.ConfigureBarrier<1> <{isStartBarrier}> -> !VPURT.Barrier
    // CHECK: VPURT.ConfigureBarrier<1> <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK: VPURT.ConfigureBarrier<1> <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK: VPURT.ConfigureBarrier<1> <{barrierIndex = 2}> -> !VPURT.Barrier
    // CHECK: VPURT.ConfigureBarrier<1> <{wlmPage = 0 : i64, barrierIndex = 1}> -> !VPURT.Barrier
}

// -----

// CHECK-LABEL: @PrintParseDeclareVirtualBarrier
func.func @PrintParseDeclareVirtualBarrier()
        -> (!VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier <{isStartBarrier}> -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier <{barrierIndex = 2}> -> !VPURT.Barrier
    %bar5 = VPURT.DeclareVirtualBarrier <{isFinalBarrier, isStartBarrier}> -> !VPURT.Barrier
    %bar6 = VPURT.DeclareVirtualBarrier <{isFinalBarrier, isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    %bar7 = VPURT.DeclareVirtualBarrier <{isFinalBarrier, isStartBarrier, wlmPage = 0 : i64, barrierIndex = 1}> -> !VPURT.Barrier

    return %bar0, %bar1, %bar2, %bar3, %bar4, %bar5, %bar6, %bar7
        : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier

    // CHECK: [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier}> -> !VPURT.Barrier
    // CHECK: [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier <{isStartBarrier}> -> !VPURT.Barrier
    // CHECK: [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier <{wlmPage = 1 : i64}> -> !VPURT.Barrier
    // CHECK: [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier <{barrierIndex = 2}> -> !VPURT.Barrier
    // CHECK: [[BAR5:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier, isStartBarrier}> -> !VPURT.Barrier
    // CHECK: [[BAR6:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier, isStartBarrier, wlmPage = 0 : i64}> -> !VPURT.Barrier
    // CHECK: [[BAR7:%.+]] = VPURT.DeclareVirtualBarrier <{isFinalBarrier, isStartBarrier, wlmPage = 0 : i64, barrierIndex = 1}> -> !VPURT.Barrier
    // return [[BAR0]], [[BAR1]], [[BAR2]], [[BAR3]], [[BAR4]], [[BAR5]], [[BAR6]], [[BAR7]]
}
