//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% unbounded-dynamism-mode=DISABLED" %s | FileCheck --check-prefix=CHECK-UDM-DISABLED %s
// RUN: vpux-opt --init-compiler="platform=%platform% unbounded-dynamism-mode=LOWER" %s | FileCheck --check-prefix=CHECK-UDM-LOWER %s
// RUN: vpux-opt --init-compiler="platform=%platform% unbounded-dynamism-mode=UPPER" %s | FileCheck --check-prefix=CHECK-UDM-UPPER %s
// RUN: vpux-opt --init-compiler="platform=%platform% unbounded-dynamism-mode=LOWER_UPPER" %s | FileCheck --check-prefix=CHECK-UDM-LOWER-UPPER %s
// RUN: vpux-opt --init-compiler="platform=%platform%" %s | FileCheck --check-prefix=CHECK-UDM-UNSET %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-UDM-DISABLED: module @CheckUnboundedDynamismMode
// CHECK-UDM-LOWER: module @CheckUnboundedDynamismMode
// CHECK-UDM-UPPER: module @CheckUnboundedDynamismMode
// CHECK-UDM-LOWER-UPPER: module @CheckUnboundedDynamismMode
// CHECK-UDM-UNSET: module @CheckUnboundedDynamismMode
module @CheckUnboundedDynamismMode {
    // The values are defined in vpux::UnboundedDynamismMode as:
    // DISABLED = 0, LOWER = 1, UPPER = 2, LOWER_UPPER = 3

    // CHECK-UDM-DISABLED-DAG:    {{    }}config.Option @config.UnboundedDynamismMode : 0 : ui64
    // CHECK-UDM-UNSET-DAG:       {{    }}config.Option @config.UnboundedDynamismMode : 0 : ui64

    // CHECK-UDM-LOWER-DAG:       {{    }}config.Option @config.UnboundedDynamismMode : 1 : ui64
    // CHECK-UDM-UPPER-DAG:       {{    }}config.Option @config.UnboundedDynamismMode : 2 : ui64
    // CHECK-UDM-LOWER-UPPER-DAG: {{    }}config.Option @config.UnboundedDynamismMode : 3 : ui64
}
