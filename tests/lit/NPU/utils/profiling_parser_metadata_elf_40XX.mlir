//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="platform=%platform% allow-custom-values=true" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --lower-VPUIP-to-ELF %data_path_npu%/profiling-40XX.mlir.txt | vpux-translate --platform=%platform% --export-ELF -o %t
// RUN: prof_parser -b %t -m | FileCheck %s
// REQUIRES: platform-NPU4000

// CHECK: {
// CHECK-NEXT:  "majorVersion": 2,
// CHECK-NEXT:  "minorVersion": 4,
// CHECK-NEXT:  "platform": {
// CHECK-NEXT:    "device": 4
// CHECK-NEXT:  },
// CHECK-NEXT:  "profilingBuffer": {
// CHECK-NEXT:    "sections": [ {
// CHECK-NEXT:      "type": 1,
// CHECK-NEXT:      "size": 64,
// CHECK-NEXT:      "typeLabel": "dpu"
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "type": 3,
// CHECK-NEXT:      "offset": 64,
// CHECK-NEXT:      "size": 32,
// CHECK-NEXT:      "typeLabel": "actshave"
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "type": 6,
// CHECK-NEXT:      "offset": 128,
// CHECK-NEXT:      "size": 512,
// CHECK-NEXT:      "typeLabel": "dmahw"
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "type": 5,
// CHECK-NEXT:      "offset": 640,
// CHECK-NEXT:      "size": 64,
// CHECK-NEXT:      "typeLabel": "pll"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    "size": 704
// CHECK-NEXT:  },
// CHECK-NEXT:  "dmaTasks": [ {
// CHECK-NEXT:    "name": "Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input",
// CHECK-NEXT:    "waitBarriers": [ 0 ],
// CHECK-NEXT:    "updateBarriers": [  ],
// CHECK-NEXT:    "hwpId": 1,
// CHECK-NEXT:    "dataIndex": 1,
// CHECK-NEXT:    "portId": 0,
// CHECK-NEXT:    "channelType": "DDR",
// CHECK-NEXT:    "sourceMemoryKind": "DDR",
// CHECK-NEXT:    "destinationMemoryKind": "CMX",
// CHECK-NEXT:    "tensorShapeInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 32 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 8, 4 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "tensorStrideInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 32 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 256, 32 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "name": "Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/_expand_input/_expand_copy_3_14",
// CHECK-NEXT:    "waitBarriers": [  ],
// CHECK-NEXT:    "updateBarriers": [ 1 ],
// CHECK-NEXT:    "hwpId": 2,
// CHECK-NEXT:    "dataIndex": 2,
// CHECK-NEXT:    "portId": 0,
// CHECK-NEXT:    "channelType": "DDR",
// CHECK-NEXT:    "sourceMemoryKind": "DDR",
// CHECK-NEXT:    "destinationMemoryKind": "CMX",
// CHECK-NEXT:    "tensorShapeInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 224 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 8, 28 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "tensorStrideInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 224 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 256, 32 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "name": "Convolution_6?t_Convolution/expand_act_channels/input-1-CMX",
// CHECK-NEXT:    "waitBarriers": [  ],
// CHECK-NEXT:    "updateBarriers": [  ],
// CHECK-NEXT:    "hwpId": 3,
// CHECK-NEXT:    "dataIndex": 3,
// CHECK-NEXT:    "portId": 0,
// CHECK-NEXT:    "channelType": "DDR",
// CHECK-NEXT:    "sourceMemoryKind": "DDR",
// CHECK-NEXT:    "destinationMemoryKind": "CMX",
// CHECK-NEXT:    "tensorShapeInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 512 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 512 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "tensorStrideInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 512 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 512 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "name": "Convolution_6?t_Convolution/expand_act_channels/input-2-CMX",
// CHECK-NEXT:    "waitBarriers": [  ],
// CHECK-NEXT:    "updateBarriers": [ 3 ],
// CHECK-NEXT:    "hwpId": 5,
// CHECK-NEXT:    "dataIndex": 5,
// CHECK-NEXT:    "portId": 0,
// CHECK-NEXT:    "channelType": "DDR",
// CHECK-NEXT:    "sourceMemoryKind": "DDR",
// CHECK-NEXT:    "destinationMemoryKind": "CMX",
// CHECK-NEXT:    "tensorShapeInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 256 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 256 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "tensorStrideInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 256 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 256 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "name": "Convolution_6?t_Convolution/expand_act_channels/input-0-CMX",
// CHECK-NEXT:    "waitBarriers": [ 2 ],
// CHECK-NEXT:    "updateBarriers": [ 3 ],
// CHECK-NEXT:    "hwpId": 4,
// CHECK-NEXT:    "dataIndex": 4,
// CHECK-NEXT:    "portId": 0,
// CHECK-NEXT:    "channelType": "CMX",
// CHECK-NEXT:    "sourceMemoryKind": "CMX",
// CHECK-NEXT:    "destinationMemoryKind": "CMX",
// CHECK-NEXT:    "tensorShapeInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 2, 16 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 32 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "tensorStrideInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 256, 128 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 32 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "name": "Conv_0?t_Convert/input-0-CMX",
// CHECK-NEXT:    "waitBarriers": [  ],
// CHECK-NEXT:    "updateBarriers": [ 5 ],
// CHECK-NEXT:    "hwpId": 6,
// CHECK-NEXT:    "dataIndex": 6,
// CHECK-NEXT:    "portId": 0,
// CHECK-NEXT:    "channelType": "CMX",
// CHECK-NEXT:    "sourceMemoryKind": "CMX",
// CHECK-NEXT:    "destinationMemoryKind": "CMX",
// CHECK-NEXT:    "tensorShapeInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 8 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 8 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "tensorStrideInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 32 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 8 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "name": "495/sink_port_0?t_Result",
// CHECK-NEXT:    "waitBarriers": [  ],
// CHECK-NEXT:    "updateBarriers": [ 7 ],
// CHECK-NEXT:    "hwpId": 7,
// CHECK-NEXT:    "dataIndex": 7,
// CHECK-NEXT:    "portId": 0,
// CHECK-NEXT:    "channelType": "CMX",
// CHECK-NEXT:    "sourceMemoryKind": "CMX",
// CHECK-NEXT:    "destinationMemoryKind": "DDR",
// CHECK-NEXT:    "tensorShapeInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 4 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 4 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "tensorStrideInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 4 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 4 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  } ],
// CHECK-NEXT:  "dpuTasks": [ {
// CHECK-NEXT:    "name": "Convolution_6?t_Convolution/reorder_in_0/PermuteQuantize/cluster_0",
// CHECK-NEXT:    "taskId": 1,
// CHECK-NEXT:    "numVariants": 1,
// CHECK-NEXT:    "maxVariants": 1,
// CHECK-NEXT:    "waitBarriers": [ 1 ],
// CHECK-NEXT:    "updateBarriers": [ 2 ],
// CHECK-NEXT:    "workloadIds": [ 0 ],
// CHECK-NEXT:    "tensorInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 1, 16, 4, 2 ],
// CHECK-NEXT:        "elemType": "f16"
// CHECK-NEXT:      }, {
// CHECK-NEXT:        "dimensions": [ 1, 16, 4, 2 ],
// CHECK-NEXT:        "elemType": "f16"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 1, 16, 4, 2 ],
// CHECK-NEXT:        "elemType": "f16"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "variantInfo": [ {
// CHECK-NEXT:      "inStart": [ 0, 0, 0 ],
// CHECK-NEXT:      "inEnd": [ 1, 3, 15 ],
// CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
// CHECK-NEXT:      "outEnd": [ 1, 3, 15 ]
// CHECK-NEXT:    } ]
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "name": "Convolution_6?t_Convolution/expand_act_channels/cluster_0",
// CHECK-NEXT:    "taskId": 2,
// CHECK-NEXT:    "numVariants": 1,
// CHECK-NEXT:    "maxVariants": 1,
// CHECK-NEXT:    "waitBarriers": [ 3 ],
// CHECK-NEXT:    "updateBarriers": [ 4 ],
// CHECK-NEXT:    "workloadIds": [ 1 ],
// CHECK-NEXT:    "tensorInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 1, 16, 2, 2 ],
// CHECK-NEXT:        "elemType": "f16"
// CHECK-NEXT:      }, {
// CHECK-NEXT:        "dimensions": [ 16, 16, 2, 2 ],
// CHECK-NEXT:        "elemType": "f16"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 1, 16, 1, 1 ],
// CHECK-NEXT:        "elemType": "f16"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "variantInfo": [ {
// CHECK-NEXT:      "inStart": [ 0, 0, 0 ],
// CHECK-NEXT:      "inEnd": [ 1, 1, 15 ],
// CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
// CHECK-NEXT:      "outEnd": [ 0, 0, 15 ]
// CHECK-NEXT:    } ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  "swTasks": [ {
// CHECK-NEXT:    "name": "Conv_0?t_Convert/cluster_0",
// CHECK-NEXT:    "waitBarriers": [ 5 ],
// CHECK-NEXT:    "updateBarriers": [ 6 ],
// CHECK-NEXT:    "taskType": "",
// CHECK-NEXT:    "clusterSize": 1,
// CHECK-NEXT:    "tensorInfo": {
// CHECK-NEXT:      "inputs": [ {
// CHECK-NEXT:        "dimensions": [ 1, 4, 1, 1 ],
// CHECK-NEXT:        "elemType": "f16"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      "outputs": [ {
// CHECK-NEXT:        "dimensions": [ 1, 4, 1, 1 ],
// CHECK-NEXT:        "elemType": "ui8"
// CHECK-NEXT:      } ]
// CHECK-NEXT:    }
// CHECK-NEXT:  } ]
// CHECK-NEXT:}
