//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="platform=%platform% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/network_GRUSequence_37XX.mlir.txt | vpux-translate --platform=%platform% --export-ELF -o %t
// RUN: prof_parser -b %t -m | FileCheck %s
// REQUIRES: platform-NPU3720

// CHECK: {
//CHECK-NEXT:   "majorVersion": 2,
//CHECK-NEXT:   "minorVersion": 4,
//CHECK-NEXT:   "platform": {
//CHECK-NEXT:     "device": 2
//CHECK-NEXT:   },
//CHECK-NEXT:   "profilingBuffer": {
//CHECK-NEXT:     "sections": [ {
//CHECK-NEXT:       "type": 3,
//CHECK-NEXT:       "size": 192,
//CHECK-NEXT:       "typeLabel": "actshave"
//CHECK-NEXT:     }, {
//CHECK-NEXT:       "type": 4,
//CHECK-NEXT:       "offset": 192,
//CHECK-NEXT:       "size": 144,
//CHECK-NEXT:       "typeLabel": "dma"
//CHECK-NEXT:     }, {
//CHECK-NEXT:       "type": 5,
//CHECK-NEXT:       "offset": 384,
//CHECK-NEXT:       "size": 64,
//CHECK-NEXT:       "typeLabel": "pll"
//CHECK-NEXT:     } ],
//CHECK-NEXT:     "size": 448
//CHECK-NEXT:   },
//CHECK-NEXT:   "dmaTasks": [ {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [ 0 ],
//CHECK-NEXT:     "portId": 0,
//CHECK-NEXT:     "channelType": "DDR",
//CHECK-NEXT:     "sourceMemoryKind": "DDR",
//CHECK-NEXT:     "destinationMemoryKind": "CMX",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [ 2 ],
//CHECK-NEXT:     "dataIndex": 5,
//CHECK-NEXT:     "portId": 1,
//CHECK-NEXT:     "channelType": "DDR",
//CHECK-NEXT:     "sourceMemoryKind": "DDR",
//CHECK-NEXT:     "destinationMemoryKind": "CMX",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [ 4 ],
//CHECK-NEXT:     "dataIndex": 1,
//CHECK-NEXT:     "portId": 0,
//CHECK-NEXT:     "channelType": "DDR",
//CHECK-NEXT:     "sourceMemoryKind": "DDR",
//CHECK-NEXT:     "destinationMemoryKind": "CMX",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape",
//CHECK-NEXT:     "waitBarriers": [ 1 ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [ 2 ],
//CHECK-NEXT:     "dataIndex": 6,
//CHECK-NEXT:     "portId": 1,
//CHECK-NEXT:     "channelType": "CMX",
//CHECK-NEXT:     "sourceMemoryKind": "CMX",
//CHECK-NEXT:     "destinationMemoryKind": "DDR",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [ 1 ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [ 2 ],
//CHECK-NEXT:     "dataIndex": 7,
//CHECK-NEXT:     "portId": 1,
//CHECK-NEXT:     "channelType": "DDR",
//CHECK-NEXT:     "sourceMemoryKind": "DDR",
//CHECK-NEXT:     "destinationMemoryKind": "CMX",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [ 6 ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [ 7 ],
//CHECK-NEXT:     "dataIndex": 2,
//CHECK-NEXT:     "portId": 0,
//CHECK-NEXT:     "channelType": "CMX",
//CHECK-NEXT:     "sourceMemoryKind": "CMX",
//CHECK-NEXT:     "destinationMemoryKind": "DDR",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [ 6 ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "dataIndex": 8,
//CHECK-NEXT:     "portId": 1,
//CHECK-NEXT:     "channelType": "CMX",
//CHECK-NEXT:     "sourceMemoryKind": "CMX",
//CHECK-NEXT:     "destinationMemoryKind": "DDR",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output",
//CHECK-NEXT:     "waitBarriers": [ 6 ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [ 7 ],
//CHECK-NEXT:     "dataIndex": 3,
//CHECK-NEXT:     "portId": 0,
//CHECK-NEXT:     "channelType": "DDR",
//CHECK-NEXT:     "sourceMemoryKind": "DDR",
//CHECK-NEXT:     "destinationMemoryKind": "CMX",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output",
//CHECK-NEXT:     "waitBarriers": [ 8 ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "isProfBegin": true,
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output",
//CHECK-NEXT:     "waitBarriers": [  ],
//CHECK-NEXT:     "updateBarriers": [  ],
//CHECK-NEXT:     "dataIndex": 4,
//CHECK-NEXT:     "portId": 0,
//CHECK-NEXT:     "channelType": "CMX",
//CHECK-NEXT:     "sourceMemoryKind": "CMX",
//CHECK-NEXT:     "destinationMemoryKind": "DDR",
//CHECK-NEXT:     "tensorShapeInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     },
//CHECK-NEXT:     "tensorStrideInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 8 ],
//CHECK-NEXT:         "elemType": "ui8"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   } ],
//CHECK-NEXT:   "swTasks": [ {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape/tile_0/cluster_0",
//CHECK-NEXT:     "waitBarriers": [ 0 ],
//CHECK-NEXT:     "updateBarriers": [ 1 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape/tile_0/cluster_1",
//CHECK-NEXT:     "waitBarriers": [ 0 ],
//CHECK-NEXT:     "updateBarriers": [ 1 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "clusterId": 1,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape/tile_1/cluster_0",
//CHECK-NEXT:     "waitBarriers": [ 0 ],
//CHECK-NEXT:     "updateBarriers": [ 1 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 1,
//CHECK-NEXT:     "tileId": 1,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "/Gru2/Unsqueeze?t_Reshape/tile_1/cluster_1",
//CHECK-NEXT:     "waitBarriers": [ 0 ],
//CHECK-NEXT:     "updateBarriers": [ 1 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 1,
//CHECK-NEXT:     "tileId": 1,
//CHECK-NEXT:     "clusterId": 1,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence/cluster_0",
//CHECK-NEXT:     "waitBarriers": [ 2 ],
//CHECK-NEXT:     "updateBarriers": [ 3 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 2,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 768 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       }, {
//CHECK-NEXT:         "dimensions": [ 1, 2304, 768 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 2304 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "GRUSequence_154?t_GRUSequence/Duplicated_2/cluster_0",
//CHECK-NEXT:     "waitBarriers": [ 4 ],
//CHECK-NEXT:     "updateBarriers": [ 5 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 3,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 2304 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       }, {
//CHECK-NEXT:         "dimensions": [ 1, 1, 768 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       }, {
//CHECK-NEXT:         "dimensions": [ 1, 2304, 768 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       }, {
//CHECK-NEXT:         "dimensions": [ 1, 3072 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 768 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       }, {
//CHECK-NEXT:         "dimensions": [ 1, 1, 768 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output/tile_0/cluster_0",
//CHECK-NEXT:     "waitBarriers": [ 7 ],
//CHECK-NEXT:     "updateBarriers": [ 8 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 4,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output/tile_0/cluster_1",
//CHECK-NEXT:     "waitBarriers": [ 7 ],
//CHECK-NEXT:     "updateBarriers": [ 8 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 4,
//CHECK-NEXT:     "clusterId": 1,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output/tile_1/cluster_0",
//CHECK-NEXT:     "waitBarriers": [ 7 ],
//CHECK-NEXT:     "updateBarriers": [ 8 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 5,
//CHECK-NEXT:     "tileId": 1,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   }, {
//CHECK-NEXT:     "name": "output?t_Output/tile_1/cluster_1",
//CHECK-NEXT:     "waitBarriers": [ 7 ],
//CHECK-NEXT:     "updateBarriers": [ 8 ],
//CHECK-NEXT:     "taskType": "",
//CHECK-NEXT:     "clusterSize": 6,
//CHECK-NEXT:     "dataIndex": 5,
//CHECK-NEXT:     "tileId": 1,
//CHECK-NEXT:     "clusterId": 1,
//CHECK-NEXT:     "tensorInfo": {
//CHECK-NEXT:       "inputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f16"
//CHECK-NEXT:       } ],
//CHECK-NEXT:       "outputs": [ {
//CHECK-NEXT:         "dimensions": [ 1, 1, 1, 384 ],
//CHECK-NEXT:         "elemType": "f32"
//CHECK-NEXT:       } ]
//CHECK-NEXT:     }
//CHECK-NEXT:   } ]
//CHECK-NEXT: }
