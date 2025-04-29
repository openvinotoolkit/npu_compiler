//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -m | FileCheck %s
// REQUIRES: arch-NPU40XX

//CHECK: {
//CHECK: majorVersion: 2,
//CHECK: platform: {
//CHECK: device: 4
//CHECK: },
//CHECK: profilingBuffer: {
//CHECK: sections: [ {
//CHECK: type: 1,
//CHECK: size: 1152
//CHECK: }, {
//CHECK: type: 3,
//CHECK: offset: 1152,
//CHECK: size: 384
//CHECK: }, {
//CHECK: type: 5,
//CHECK: offset: 1536,
//CHECK: size: 64
//CHECK: } ],
//CHECK: size: 1600
//CHECK: },
//CHECK: dpuTasks: [ {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_0",
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [ 16 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [ 16 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [ 16 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [ 16 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_4",
//CHECK: clusterId: 4,
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [ 16 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/cluster_5",
//CHECK: clusterId: 5,
//CHECK: taskId: 1,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 4 ],
//CHECK: updateBarriers: [ 5 ],
//CHECK: workloadIds: [ 16 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0",
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 17 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 17 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 17 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 17 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_4",
//CHECK: clusterId: 4,
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 17 ]
//CHECK: }, {
//CHECK: name: "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_5",
//CHECK: clusterId: 5,
//CHECK: taskId: 2,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 7 ],
//CHECK: updateBarriers: [ 8 ],
//CHECK: workloadIds: [ 17 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_0",
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 18, 19 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 18, 19 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 18, 19 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 18, 19 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_4",
//CHECK: clusterId: 4,
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 18, 19 ]
//CHECK: }, {
//CHECK: name: "relu1?t_Relu/cluster_5",
//CHECK: clusterId: 5,
//CHECK: taskId: 3,
//CHECK: numVariants: 2,
//CHECK: maxVariants: 2,
//CHECK: waitBarriers: [ 8 ],
//CHECK: updateBarriers: [ 9 ],
//CHECK: workloadIds: [ 18, 19 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_0",
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 20 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 20 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 20 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 20 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_4",
//CHECK: clusterId: 4,
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 20 ]
//CHECK: }, {
//CHECK: name: "conv2/WithoutBiases?t_Convolution/cluster_5",
//CHECK: clusterId: 5,
//CHECK: taskId: 4,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 9 ],
//CHECK: updateBarriers: [ 10 ],
//CHECK: workloadIds: [ 20 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_0",
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 21 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_1",
//CHECK: clusterId: 1,
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 21 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_2",
//CHECK: clusterId: 2,
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 21 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_3",
//CHECK: clusterId: 3,
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 21 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_4",
//CHECK: clusterId: 4,
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 21 ]
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/cluster_5",
//CHECK: clusterId: 5,
//CHECK: taskId: 5,
//CHECK: numVariants: 1,
//CHECK: maxVariants: 1,
//CHECK: waitBarriers: [ 10 ],
//CHECK: updateBarriers: [ 11 ],
//CHECK: workloadIds: [ 21 ]
//CHECK: } ],
//CHECK: swTasks: [ {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_0",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_1",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: clusterId: 1
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_2",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: clusterId: 2
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_3",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: clusterId: 3
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_4",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: clusterId: 4
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_0/cluster_5",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: clusterId: 5
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_0",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_1",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1,
//CHECK: clusterId: 1
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_2",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1,
//CHECK: clusterId: 2
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_3",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1,
//CHECK: clusterId: 3
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_4",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1,
//CHECK: clusterId: 4
//CHECK: }, {
//CHECK: name: "relu2?t_Relu/converted_to_f32/tile_1/cluster_5",
//CHECK: waitBarriers: [ 13 ],
//CHECK: updateBarriers: [ 14 ],
//CHECK: taskType: "",
//CHECK: clusterSize: 2,
//CHECK: dataIndex: 1,
//CHECK: tileId: 1,
//CHECK: clusterId: 5
//CHECK: } ]
//CHECK: }
