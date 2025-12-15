//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --init-compiler="vpu-arch=%arch%" --init-compiler="vpu-arch=%arch%" -verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// expected-error@+1 {{Target platform is already set, probably you run '--init-compiler' twice}}
module @test {
}
