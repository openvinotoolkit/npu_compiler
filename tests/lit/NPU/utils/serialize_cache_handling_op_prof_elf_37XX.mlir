//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/network_GRUSequence_37XX.mlir.txt | vpux-translate --vpu-arch=%arch% --export-ELF -o %t
// RUN: prof_parser -b %t -m | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK: {
// CHECK: majorVersion: 2,
// CHECK: platform: {
// CHECK: device: 2
// CHECK: },
// CHECK: profilingBuffer: {
// CHECK: sections: [ {
// CHECK: type: 3,
// CHECK: size: 192
// CHECK: }, {
// CHECK: type: 4,
// CHECK: offset: 192,
// CHECK: size: 144
// CHECK: }, {
// CHECK: type: 5,
// CHECK: offset: 384,
// CHECK: size: 64
// CHECK: } ],
// CHECK: size: 448
// CHECK: },
// CHECK: dmaTasks: [ {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [ 0 ]
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [ 2 ],
// CHECK: dataIndex: 5
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [ 4 ],
// CHECK: dataIndex: 1
// CHECK: }, {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape",
// CHECK: waitBarriers: [ 1 ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [ 2 ],
// CHECK: dataIndex: 6
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [ 1 ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [ 2 ],
// CHECK: dataIndex: 7
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [ 6 ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [ 7 ],
// CHECK: dataIndex: 2
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [ 6 ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [  ],
// CHECK: dataIndex: 8
// CHECK: }, {
// CHECK: name: "output?t_Output",
// CHECK: waitBarriers: [ 6 ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "output?t_Output",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [ 7 ],
// CHECK: dataIndex: 3
// CHECK: }, {
// CHECK: name: "output?t_Output",
// CHECK: waitBarriers: [ 8 ],
// CHECK: updateBarriers: [  ],
// CHECK: isProfBegin: true
// CHECK: }, {
// CHECK: name: "output?t_Output",
// CHECK: waitBarriers: [  ],
// CHECK: updateBarriers: [  ],
// CHECK: dataIndex: 4
// CHECK: } ],
// CHECK: swTasks: [ {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape/tile_0/cluster_0",
// CHECK: waitBarriers: [ 0 ],
// CHECK: updateBarriers: [ 1 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6
// CHECK: }, {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape/tile_0/cluster_1",
// CHECK: waitBarriers: [ 0 ],
// CHECK: updateBarriers: [ 1 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: clusterId: 1
// CHECK: }, {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape/tile_1/cluster_0",
// CHECK: waitBarriers: [ 0 ],
// CHECK: updateBarriers: [ 1 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 1,
// CHECK: tileId: 1
// CHECK: }, {
// CHECK: name: "/Gru2/Unsqueeze?t_Reshape/tile_1/cluster_1",
// CHECK: waitBarriers: [ 0 ],
// CHECK: updateBarriers: [ 1 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 1,
// CHECK: tileId: 1,
// CHECK: clusterId: 1
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence/cluster_0",
// CHECK: waitBarriers: [ 2 ],
// CHECK: updateBarriers: [ 3 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 2
// CHECK: }, {
// CHECK: name: "GRUSequence_154?t_GRUSequence/Duplicated_2/cluster_0",
// CHECK: waitBarriers: [ 4 ],
// CHECK: updateBarriers: [ 5 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 3
// CHECK: }, {
// CHECK: name: "output?t_Output/tile_0/cluster_0",
// CHECK: waitBarriers: [ 7 ],
// CHECK: updateBarriers: [ 8 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 4
// CHECK: }, {
// CHECK: name: "output?t_Output/tile_0/cluster_1",
// CHECK: waitBarriers: [ 7 ],
// CHECK: updateBarriers: [ 8 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 4,
// CHECK: clusterId: 1
// CHECK: }, {
// CHECK: name: "output?t_Output/tile_1/cluster_0",
// CHECK: waitBarriers: [ 7 ],
// CHECK: updateBarriers: [ 8 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 5,
// CHECK: tileId: 1
// CHECK: }, {
// CHECK: name: "output?t_Output/tile_1/cluster_1",
// CHECK: waitBarriers: [ 7 ],
// CHECK: updateBarriers: [ 8 ],
// CHECK: taskType: "",
// CHECK: clusterSize: 6,
// CHECK: dataIndex: 5,
// CHECK: tileId: 1,
// CHECK: clusterId: 1
// CHECK: } ]
// CHECK: }
