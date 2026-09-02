//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="platform=NPU4000 allow-custom-values=true"  --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX-actshave-fifo.mlir.txt | vpux-translate --platform=NPU4000 --export-ELF -o %t
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-40XX.bin -f json | FileCheck %s
// REQUIRES: platform-NPU4000

// CHECK: {"name": "thread_name", "ph": "M", "pid":1, "tid":[[TID:[0-9]+]], "args": {"name" : "Shave0"}}
// CHECK: {"name":"Conv_0?t_Convert/cluster_0",
// CHECK-SAME: "cat":"Shave",
// CHECK-SAME: "ph":"X",
// CHECK-SAME: "ts":35.833,
// CHECK-SAME: "dur":5.468,
// CHECK-SAME: "pid":1,
// CHECK-SAME: "tid":[[TID]],
// CHECK-SAME: "args":{"Total cycles": "5761", "Active cycles": "379", "Stall cycles": "5382", "LSU0 stalls": "372", "LSU1 stalls": "87", "Instruction stalls": "4936", "Input tensors": "[1x4x1x1xf16]", "Output tensors": "[1x4x1x1xui8]", "FIFO ID": "0"}}
