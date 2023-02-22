//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt %s --init-compiler="vpu-arch=%arch%" --init-compiler="vpu-arch=%arch%" -verify-diagnostics
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @test {
}
