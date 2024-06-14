//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --export-VPUIP -o %t %data_path_npu%/profiling-30XX.mlir.txt
// RUN: prof_parser -b %t -p %data_path_npu%/profiling-0-30XX.bin -f json -vv | FileCheck %s
// REQUIRES: arch-VPUX30XX

// CHECK: {"traceEvents":[
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":0, "args": {"name" : "DMA"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":0, "args": {"sort_index" : "0"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":0, "tid":0, "args": {"name" : "DMA"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":1, "args": {"name" : "Cluster (0)"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":1, "args": {"sort_index" : "1"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":0, "args": {"name" : "DPU"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":1, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":2, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":3, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":4, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":1, "tid":5, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":2, "args": {"name" : "Cluster (1)"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":2, "args": {"sort_index" : "2"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":0, "args": {"name" : "DPU"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":1, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":2, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":3, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":4, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":2, "tid":5, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":3, "args": {"name" : "Cluster (2)"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":3, "args": {"sort_index" : "3"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":3, "tid":0, "args": {"name" : "DPU"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":3, "tid":1, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":3, "tid":2, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":3, "tid":3, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":3, "tid":4, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":3, "tid":5, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":4, "args": {"name" : "Cluster (3)"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":4, "args": {"sort_index" : "4"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":4, "tid":0, "args": {"name" : "DPU"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":4, "tid":1, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":4, "tid":2, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":4, "tid":3, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":4, "tid":4, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":4, "tid":5, "args": {"name" : "DPU / Variants"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":5, "args": {"name" : "UPA"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":5, "args": {"sort_index" : "5"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":5, "tid":0, "args": {"name" : "SW"}},
// CHECK-NEXT: {"name": "process_name", "ph": "M", "pid":6, "args": {"name" : "Layers"}},
// CHECK-NEXT: {"name": "process_sort_index", "ph": "M", "pid":6, "args": {"sort_index" : "6"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":6, "tid":0, "args": {"name" : "Layers"}},
// CHECK-NEXT: {"name": "thread_name", "ph": "M", "pid":6, "tid":1, "args": {"name" : "Layers"}},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/_expand_copy_1_13", "cat":"DMA", "ph":"X", "ts":0.000, "dur":17.495, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"DMA", "ph":"X", "ts":65.593, "dur":4.032, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]", "cat":"DMA", "ph":"X", "ts":69.845, "dur":1.355, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_0", "cat":"DMA", "ph":"X", "ts":228.856, "dur":2.584, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_1", "cat":"DMA", "ph":"X", "ts":231.592, "dur":2.852, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_2", "cat":"DMA", "ph":"X", "ts":234.590, "dur":2.724, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/_cluster_3", "cat":"DMA", "ph":"X", "ts":237.465, "dur":2.452, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_fused_constant/_fused_tile/_broadcast_copy_to_CMX[0,3]", "cat":"DMA", "ph":"X", "ts":240.062, "dur":0.561, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_cluster_0", "cat":"DMA", "ph":"X", "ts":299.670, "dur":2.024, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_cluster_1", "cat":"DMA", "ph":"X", "ts":301.903, "dur":2.022, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_cluster_2", "cat":"DMA", "ph":"X", "ts":304.283, "dur":2.151, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_cluster_3", "cat":"DMA", "ph":"X", "ts":306.666, "dur":1.445, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool", "cat":"DMA", "ph":"X", "ts":308.392, "dur":6.504, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":315.040, "dur":36.390, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":351.580, "dur":36.390, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":388.115, "dur":36.397, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/_unrolled_permuteDMA", "cat":"DMA", "ph":"X", "ts":424.656, "dur":18.852, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool", "cat":"DMA", "ph":"X", "ts":443.653, "dur":7.104, "pid":0, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0", "cat":"DPU", "ph":"X", "ts":239.918, "dur":42.810, "pid":1, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_4", "cat":"DPU", "ph":"X", "ts":239.918, "dur":9.758, "pid":1, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_1", "cat":"DPU", "ph":"X", "ts":239.956, "dur":9.995, "pid":1, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_0", "cat":"DPU", "ph":"X", "ts":239.995, "dur":9.841, "pid":1, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_2", "cat":"DPU", "ph":"X", "ts":240.110, "dur":9.764, "pid":1, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_3", "cat":"DPU", "ph":"X", "ts":240.188, "dur":7.245, "pid":1, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_9", "cat":"DPU", "ph":"X", "ts":248.388, "dur":9.782, "pid":1, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_5", "cat":"DPU", "ph":"X", "ts":250.608, "dur":9.730, "pid":1, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_7", "cat":"DPU", "ph":"X", "ts":250.735, "dur":7.484, "pid":1, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_8", "cat":"DPU", "ph":"X", "ts":250.928, "dur":9.758, "pid":1, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_6", "cat":"DPU", "ph":"X", "ts":251.039, "dur":9.695, "pid":1, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_10", "cat":"DPU", "ph":"X", "ts":259.112, "dur":9.780, "pid":1, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_12", "cat":"DPU", "ph":"X", "ts":259.160, "dur":9.852, "pid":1, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_11", "cat":"DPU", "ph":"X", "ts":261.266, "dur":7.451, "pid":1, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_13", "cat":"DPU", "ph":"X", "ts":261.619, "dur":9.810, "pid":1, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_14", "cat":"DPU", "ph":"X", "ts":261.668, "dur":9.810, "pid":1, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_17", "cat":"DPU", "ph":"X", "ts":269.650, "dur":10.490, "pid":1, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_15", "cat":"DPU", "ph":"X", "ts":269.820, "dur":7.371, "pid":1, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_16", "cat":"DPU", "ph":"X", "ts":269.980, "dur":10.484, "pid":1, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_18", "cat":"DPU", "ph":"X", "ts":272.369, "dur":10.358, "pid":1, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_0/variant_19", "cat":"DPU", "ph":"X", "ts":272.446, "dur":7.941, "pid":1, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0", "cat":"DPU", "ph":"X", "ts":289.829, "dur":6.862, "pid":1, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_0", "cat":"DPU", "ph":"X", "ts":289.829, "dur":1.804, "pid":1, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_3", "cat":"DPU", "ph":"X", "ts":289.906, "dur":2.154, "pid":1, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_2", "cat":"DPU", "ph":"X", "ts":289.983, "dur":1.611, "pid":1, "tid":3},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_1", "cat":"DPU", "ph":"X", "ts":290.099, "dur":1.572, "pid":1, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_4", "cat":"DPU", "ph":"X", "ts":290.176, "dur":1.922, "pid":1, "tid":5},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_7", "cat":"DPU", "ph":"X", "ts":292.512, "dur":1.875, "pid":1, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_5", "cat":"DPU", "ph":"X", "ts":292.550, "dur":1.237, "pid":1, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_8", "cat":"DPU", "ph":"X", "ts":292.589, "dur":1.355, "pid":1, "tid":3},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_6", "cat":"DPU", "ph":"X", "ts":292.979, "dur":1.757, "pid":1, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_9", "cat":"DPU", "ph":"X", "ts":293.098, "dur":1.885, "pid":1, "tid":5},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_11", "cat":"DPU", "ph":"X", "ts":294.785, "dur":0.902, "pid":1, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_0/variant_10", "cat":"DPU", "ph":"X", "ts":295.335, "dur":1.357, "pid":1, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1", "cat":"DPU", "ph":"X", "ts":240.033, "dur":43.785, "pid":2, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_2", "cat":"DPU", "ph":"X", "ts":240.033, "dur":10.817, "pid":2, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_3", "cat":"DPU", "ph":"X", "ts":240.072, "dur":8.097, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_4", "cat":"DPU", "ph":"X", "ts":240.149, "dur":9.764, "pid":2, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_1", "cat":"DPU", "ph":"X", "ts":240.265, "dur":10.547, "pid":2, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_0", "cat":"DPU", "ph":"X", "ts":240.475, "dur":10.298, "pid":2, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_6", "cat":"DPU", "ph":"X", "ts":249.138, "dur":9.735, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_7", "cat":"DPU", "ph":"X", "ts":250.889, "dur":7.378, "pid":2, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_9", "cat":"DPU", "ph":"X", "ts":252.012, "dur":9.794, "pid":2, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_5", "cat":"DPU", "ph":"X", "ts":252.099, "dur":9.850, "pid":2, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_8", "cat":"DPU", "ph":"X", "ts":252.138, "dur":9.717, "pid":2, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_11", "cat":"DPU", "ph":"X", "ts":259.236, "dur":7.407, "pid":2, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_10", "cat":"DPU", "ph":"X", "ts":259.840, "dur":9.761, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_12", "cat":"DPU", "ph":"X", "ts":263.033, "dur":9.741, "pid":2, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_14", "cat":"DPU", "ph":"X", "ts":263.082, "dur":9.741, "pid":2, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_13", "cat":"DPU", "ph":"X", "ts":263.162, "dur":9.832, "pid":2, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_15", "cat":"DPU", "ph":"X", "ts":267.605, "dur":7.374, "pid":2, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_16", "cat":"DPU", "ph":"X", "ts":270.568, "dur":9.767, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_17", "cat":"DPU", "ph":"X", "ts":273.938, "dur":9.755, "pid":2, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_18", "cat":"DPU", "ph":"X", "ts":274.049, "dur":9.770, "pid":2, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_1/variant_19", "cat":"DPU", "ph":"X", "ts":274.233, "dur":7.415, "pid":2, "tid":5},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1", "cat":"DPU", "ph":"X", "ts":289.559, "dur":7.210, "pid":2, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_0", "cat":"DPU", "ph":"X", "ts":289.559, "dur":2.347, "pid":2, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_4", "cat":"DPU", "ph":"X", "ts":289.598, "dur":2.347, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_1", "cat":"DPU", "ph":"X", "ts":289.713, "dur":2.308, "pid":2, "tid":3},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_3", "cat":"DPU", "ph":"X", "ts":289.790, "dur":1.685, "pid":2, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_2", "cat":"DPU", "ph":"X", "ts":290.060, "dur":0.995, "pid":2, "tid":5},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_9", "cat":"DPU", "ph":"X", "ts":292.179, "dur":1.842, "pid":2, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_8", "cat":"DPU", "ph":"X", "ts":292.630, "dur":0.795, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_5", "cat":"DPU", "ph":"X", "ts":293.538, "dur":1.148, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_6", "cat":"DPU", "ph":"X", "ts":293.649, "dur":1.797, "pid":2, "tid":3},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_7", "cat":"DPU", "ph":"X", "ts":293.865, "dur":1.784, "pid":2, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_11", "cat":"DPU", "ph":"X", "ts":294.458, "dur":1.104, "pid":2, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_1/variant_10", "cat":"DPU", "ph":"X", "ts":295.140, "dur":1.628, "pid":2, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2", "cat":"DPU", "ph":"X", "ts":240.226, "dur":45.585, "pid":3, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_3", "cat":"DPU", "ph":"X", "ts":240.226, "dur":9.291, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_0", "cat":"DPU", "ph":"X", "ts":240.436, "dur":12.287, "pid":3, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_2", "cat":"DPU", "ph":"X", "ts":240.552, "dur":12.105, "pid":3, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_1", "cat":"DPU", "ph":"X", "ts":240.629, "dur":11.967, "pid":3, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_4", "cat":"DPU", "ph":"X", "ts":240.668, "dur":9.322, "pid":3, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_5", "cat":"DPU", "ph":"X", "ts":250.559, "dur":9.874, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_9", "cat":"DPU", "ph":"X", "ts":251.105, "dur":9.768, "pid":3, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_8", "cat":"DPU", "ph":"X", "ts":254.040, "dur":9.811, "pid":3, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_7", "cat":"DPU", "ph":"X", "ts":254.089, "dur":7.412, "pid":3, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_6", "cat":"DPU", "ph":"X", "ts":254.156, "dur":9.838, "pid":3, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_10", "cat":"DPU", "ph":"X", "ts":261.572, "dur":9.771, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_11", "cat":"DPU", "ph":"X", "ts":262.075, "dur":7.357, "pid":3, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_13", "cat":"DPU", "ph":"X", "ts":262.622, "dur":9.708, "pid":3, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_12", "cat":"DPU", "ph":"X", "ts":265.108, "dur":9.764, "pid":3, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_14", "cat":"DPU", "ph":"X", "ts":265.269, "dur":9.758, "pid":3, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_16", "cat":"DPU", "ph":"X", "ts":270.496, "dur":9.692, "pid":3, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_15", "cat":"DPU", "ph":"X", "ts":272.408, "dur":7.372, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_17", "cat":"DPU", "ph":"X", "ts":273.418, "dur":9.712, "pid":3, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_18", "cat":"DPU", "ph":"X", "ts":276.120, "dur":9.691, "pid":3, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_2/variant_19", "cat":"DPU", "ph":"X", "ts":276.279, "dur":7.334, "pid":3, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2", "cat":"DPU", "ph":"X", "ts":289.675, "dur":8.785, "pid":3, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_4", "cat":"DPU", "ph":"X", "ts":289.675, "dur":2.308, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_1", "cat":"DPU", "ph":"X", "ts":289.752, "dur":2.465, "pid":3, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_3", "cat":"DPU", "ph":"X", "ts":289.868, "dur":2.388, "pid":3, "tid":3},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_0", "cat":"DPU", "ph":"X", "ts":290.022, "dur":1.502, "pid":3, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_2", "cat":"DPU", "ph":"X", "ts":290.138, "dur":0.998, "pid":3, "tid":5},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_9", "cat":"DPU", "ph":"X", "ts":292.685, "dur":2.535, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_8", "cat":"DPU", "ph":"X", "ts":293.018, "dur":0.808, "pid":3, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_5", "cat":"DPU", "ph":"X", "ts":294.216, "dur":0.844, "pid":3, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_6", "cat":"DPU", "ph":"X", "ts":294.823, "dur":1.432, "pid":3, "tid":3},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_7", "cat":"DPU", "ph":"X", "ts":294.903, "dur":1.467, "pid":3, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_11", "cat":"DPU", "ph":"X", "ts":295.610, "dur":1.120, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_2/variant_10", "cat":"DPU", "ph":"X", "ts":296.860, "dur":1.600, "pid":3, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3", "cat":"DPU", "ph":"X", "ts":240.359, "dur":48.818, "pid":4, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_3", "cat":"DPU", "ph":"X", "ts":240.359, "dur":11.701, "pid":4, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_1", "cat":"DPU", "ph":"X", "ts":240.398, "dur":15.594, "pid":4, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_2", "cat":"DPU", "ph":"X", "ts":240.513, "dur":15.227, "pid":4, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_4", "cat":"DPU", "ph":"X", "ts":240.590, "dur":9.134, "pid":4, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_0", "cat":"DPU", "ph":"X", "ts":240.706, "dur":15.237, "pid":4, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_8", "cat":"DPU", "ph":"X", "ts":251.385, "dur":9.692, "pid":4, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_5", "cat":"DPU", "ph":"X", "ts":253.499, "dur":9.757, "pid":4, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_7", "cat":"DPU", "ph":"X", "ts":257.265, "dur":7.351, "pid":4, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_9", "cat":"DPU", "ph":"X", "ts":257.446, "dur":9.844, "pid":4, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_6", "cat":"DPU", "ph":"X", "ts":257.573, "dur":9.825, "pid":4, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_10", "cat":"DPU", "ph":"X", "ts":262.132, "dur":9.758, "pid":4, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_11", "cat":"DPU", "ph":"X", "ts":264.329, "dur":7.411, "pid":4, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_12", "cat":"DPU", "ph":"X", "ts":265.688, "dur":9.732, "pid":4, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_13", "cat":"DPU", "ph":"X", "ts":268.523, "dur":9.758, "pid":4, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_14", "cat":"DPU", "ph":"X", "ts":268.593, "dur":9.807, "pid":4, "tid":5},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_16", "cat":"DPU", "ph":"X", "ts":272.946, "dur":9.861, "pid":4, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_15", "cat":"DPU", "ph":"X", "ts":273.153, "dur":7.360, "pid":4, "tid":4},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_17", "cat":"DPU", "ph":"X", "ts":276.489, "dur":9.630, "pid":4, "tid":2},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_18", "cat":"DPU", "ph":"X", "ts":279.539, "dur":9.638, "pid":4, "tid":3},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution/output tile [0, 0, 0, 0]/cluster_3/variant_19", "cat":"DPU", "ph":"X", "ts":279.645, "dur":7.370, "pid":4, "tid":5},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_3", "cat":"DPU", "ph":"X", "ts":289.636, "dur":5.385, "pid":4, "tid":0},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_3/variant_4", "cat":"DPU", "ph":"X", "ts":289.636, "dur":3.128, "pid":4, "tid":1},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_3/variant_0", "cat":"DPU", "ph":"X", "ts":289.945, "dur":2.968, "pid":4, "tid":2},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_3/variant_2", "cat":"DPU", "ph":"X", "ts":290.215, "dur":1.534, "pid":4, "tid":3},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_3/variant_1", "cat":"DPU", "ph":"X", "ts":290.253, "dur":2.178, "pid":4, "tid":4},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_3/variant_3", "cat":"DPU", "ph":"X", "ts":290.606, "dur":2.745, "pid":4, "tid":5},
// CHECK-NEXT: {"name":"pool1?t_MaxPool/cluster_3/variant_5", "cat":"DPU", "ph":"X", "ts":293.983, "dur":1.038, "pid":4, "tid":1},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"UPA", "ph":"X", "ts":23.175, "dur":37.684, "pid":5, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases?t_Convolution", "cat":"UPA", "ph":"X", "ts":69.627, "dur":154.490, "pid":5, "tid":0},
// CHECK-NEXT: {"name":"output?t_Output", "cat":"UPA", "ph":"X", "ts":454.155, "dur":63.481, "pid":5, "tid":0},
// CHECK-NEXT: {"name":"conv1/WithoutBiases", "cat":"Layer", "ph":"X", "ts":0.000, "dur":289.177, "pid":6, "tid":0, "args":{"Layer type": "Convolution"}},
// CHECK-NEXT: {"name":"pool1", "cat":"Layer", "ph":"X", "ts":240.062, "dur":210.695, "pid":6, "tid":1, "args":{"Layer type": "MaxPool"}},
// CHECK-NEXT: {"name":"output", "cat":"Layer", "ph":"X", "ts":454.155, "dur":63.481, "pid":6, "tid":0, "args":{"Layer type": "Convert"}}
// CHECK-NEXT: ],
// CHECK-NEXT: "taskStatistics": {
// CHECK-NEXT: "total duration":517.636,
// CHECK-NEXT: "DMA duration":183.334,
// CHECK-NEXT: "DPU duration":58.160,
// CHECK-NEXT: "SW duration":255.655,
// CHECK-NEXT: "M2I duration":0.000,
// CHECK-NEXT: "DMA-DPU overlap":0.561,
// CHECK-NEXT: "DMA-SW overlap":1.355,
// CHECK-NEXT: "SW-DPU overlap":0.000,
// CHECK-NEXT: "all tasks union":495.233,
// CHECK-NEXT: "total idle":22.403,
// CHECK-NEXT: "SW duration without DPU overlap":255.655,
// CHECK-NEXT: "DMA duration without overlaps":181.418,
// CHECK-NEXT: "Sum of DMA task durations":183.334,
// CHECK-NEXT: "Sum of DPU task durations":209.240,
// CHECK-NEXT: "Sum of SW task durations":255.655,
// CHECK-NEXT: "Sum of M2I task durations":0.000
// CHECK-NEXT: },
// CHECK-NEXT: "displayTimeUnit": "ns"
// CHECK-NEXT: }
