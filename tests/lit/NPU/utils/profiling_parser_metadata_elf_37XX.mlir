//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --mlir-print-debuginfo --init-compiler="platform=%platform% allow-custom-values=true" --lower-VPUIP-to-ELF %data_path_npu%/profiling-37XX.mlir.txt | vpux-translate --platform=%platform% --export-ELF -o %t
// RUN: prof_parser -b %t -m | FileCheck %s
// REQUIRES: platform-NPU3720

//CHECK: {
//CHECK-NEXT:  "majorVersion": 2,
//CHECK-NEXT:  "minorVersion": 4,
//CHECK-NEXT:  "platform": {
//CHECK-NEXT:    "device": 2
//CHECK-NEXT:  },
//CHECK-NEXT:  "profilingBuffer": {
//CHECK-NEXT:    "sections": [ {
//CHECK-NEXT:      "type": 1,
//CHECK-NEXT:      "size": 192,
//CHECK-NEXT:      "typeLabel": "dpu"
//CHECK-NEXT:    }, {
//CHECK-NEXT:      "type": 3,
//CHECK-NEXT:      "offset": 192,
//CHECK-NEXT:      "size": 256,
//CHECK-NEXT:      "typeLabel": "actshave"
//CHECK-NEXT:    }, {
//CHECK-NEXT:      "type": 4,
//CHECK-NEXT:      "offset": 448,
//CHECK-NEXT:      "size": 480,
//CHECK-NEXT:      "typeLabel": "dma"
//CHECK-NEXT:    }, {
//CHECK-NEXT:      "type": 5,
//CHECK-NEXT:      "offset": 960,
//CHECK-NEXT:      "size": 64,
//CHECK-NEXT:      "typeLabel": "pll"
//CHECK-NEXT:    } ],
//CHECK-NEXT:    "size": 1024
//CHECK-NEXT:  },
//CHECK-NEXT:  "dmaTasks": [ {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 0 ],
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 0 ],
//CHECK-NEXT:    "dataIndex": 16,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 0 ],
//CHECK-NEXT:    "dataIndex": 1,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 0 ],
//CHECK-NEXT:    "dataIndex": 17,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 1 ],
//CHECK-NEXT:    "dataIndex": 2,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_expand_copy_3_2/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 1 ],
//CHECK-NEXT:    "dataIndex": 18,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 2 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 1 ],
//CHECK-NEXT:    "dataIndex": 3,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 2 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 1 ],
//CHECK-NEXT:    "dataIndex": 19,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 1 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 3 ],
//CHECK-NEXT:    "dataIndex": 4,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 1 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 3 ],
//CHECK-NEXT:    "dataIndex": 20,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 3 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 4 ],
//CHECK-NEXT:    "dataIndex": 5,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 3 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 4 ],
//CHECK-NEXT:    "dataIndex": 21,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK-NEXT:    "waitBarriers": [ 3 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 5 ],
//CHECK-NEXT:    "dataIndex": 6,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 6 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 5 ],
//CHECK-NEXT:    "dataIndex": 7,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 6 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 5 ],
//CHECK-NEXT:    "dataIndex": 22,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 5 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 7 ],
//CHECK-NEXT:    "dataIndex": 8,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 5 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 7 ],
//CHECK-NEXT:    "dataIndex": 23,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK-NEXT:    "waitBarriers": [ 5 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv2/WithoutBiases?t_Convolution/_fused_constant/_fused_tile",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 8 ],
//CHECK-NEXT:    "dataIndex": 9,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 11 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 12 ],
//CHECK-NEXT:    "dataIndex": 10,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 11 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 12 ],
//CHECK-NEXT:    "dataIndex": 24,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 11 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 12 ],
//CHECK-NEXT:    "dataIndex": 11,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 11 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 12 ],
//CHECK-NEXT:    "dataIndex": 25,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 12 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 13 ],
//CHECK-NEXT:    "dataIndex": 12,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 12 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 13 ],
//CHECK-NEXT:    "dataIndex": 26,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 12 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 13 ],
//CHECK-NEXT:    "dataIndex": 13,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 12 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [ 13 ],
//CHECK-NEXT:    "dataIndex": 27,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "DDR",
//CHECK-NEXT:    "sourceMemoryKind": "DDR",
//CHECK-NEXT:    "destinationMemoryKind": "CMX",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 14 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "dataIndex": 14,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 14 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "dataIndex": 28,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 14 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_0",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "dataIndex": 15,
//CHECK-NEXT:    "portId": 0,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 14 ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "isProfBegin": true,
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/_cluster_1",
//CHECK-NEXT:    "waitBarriers": [  ],
//CHECK-NEXT:    "updateBarriers": [  ],
//CHECK-NEXT:    "dataIndex": 29,
//CHECK-NEXT:    "portId": 1,
//CHECK-NEXT:    "channelType": "CMX",
//CHECK-NEXT:    "sourceMemoryKind": "CMX",
//CHECK-NEXT:    "destinationMemoryKind": "DDR",
//CHECK-NEXT:    "tensorShapeInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "tensorStrideInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 8 ],
//CHECK-NEXT:        "elemType": "ui8"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  } ],
//CHECK-NEXT:  "dpuTasks": [ {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/cluster_0",
//CHECK-NEXT:    "taskId": 1,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 4 ],
//CHECK-NEXT:    "updateBarriers": [ 6 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 31 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      }, {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 31 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 16, 31 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
//CHECK-NEXT:      "outEnd": [ 30, 2, 63 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/cluster_1",
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "taskId": 1,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 4 ],
//CHECK-NEXT:    "updateBarriers": [ 6 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 31 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      }, {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 31 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 16, 31 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
//CHECK-NEXT:      "outEnd": [ 30, 2, 63 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_0",
//CHECK-NEXT:    "taskId": 2,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 7 ],
//CHECK-NEXT:    "updateBarriers": [ 9 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 16, 32, 62 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      }, {
//CHECK-NEXT:        "dimensions": [ 48, 16, 3, 3 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 30, 60 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
//CHECK-NEXT:      "outEnd": [ 59, 29, 47 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv1/WithoutBiases?t_Convolution/Duplicated_2/cluster_1",
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "taskId": 2,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 7 ],
//CHECK-NEXT:    "updateBarriers": [ 9 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 16, 30, 62 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      }, {
//CHECK-NEXT:        "dimensions": [ 48, 16, 3, 3 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 30, 60 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 30, 0 ],
//CHECK-NEXT:      "outEnd": [ 59, 59, 47 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu1?t_Relu/cluster_0",
//CHECK-NEXT:    "taskId": 3,
//CHECK-NEXT:    "numVariants": 2,
//CHECK-NEXT:    "maxVariants": 2,
//CHECK-NEXT:    "waitBarriers": [ 9 ],
//CHECK-NEXT:    "updateBarriers": [ 8 ],
//CHECK-NEXT:    "workloadIds": [ 0, 1 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 30, 60 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 16, 30 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
//CHECK-NEXT:      "outEnd": [ 29, 15, 31 ]
//CHECK-NEXT:    }, {
//CHECK-NEXT:      "outStart": [ 0, 0, 32 ],
//CHECK-NEXT:      "outEnd": [ 29, 15, 47 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu1?t_Relu/cluster_1",
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "taskId": 3,
//CHECK-NEXT:    "numVariants": 2,
//CHECK-NEXT:    "maxVariants": 2,
//CHECK-NEXT:    "waitBarriers": [ 9 ],
//CHECK-NEXT:    "updateBarriers": [ 8 ],
//CHECK-NEXT:    "workloadIds": [ 0, 1 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 30, 60 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 14, 30 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 16, 0 ],
//CHECK-NEXT:      "outEnd": [ 29, 29, 31 ]
//CHECK-NEXT:    }, {
//CHECK-NEXT:      "outStart": [ 0, 16, 32 ],
//CHECK-NEXT:      "outEnd": [ 29, 29, 47 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv2/WithoutBiases?t_Convolution/cluster_0",
//CHECK-NEXT:    "taskId": 4,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 8 ],
//CHECK-NEXT:    "updateBarriers": [ 10 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 16, 30 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      }, {
//CHECK-NEXT:        "dimensions": [ 64, 48, 3, 3 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 14, 28 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
//CHECK-NEXT:      "outEnd": [ 27, 13, 63 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "conv2/WithoutBiases?t_Convolution/cluster_1",
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "taskId": 4,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 8 ],
//CHECK-NEXT:    "updateBarriers": [ 10 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 48, 14, 30 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      }, {
//CHECK-NEXT:        "dimensions": [ 64, 48, 3, 3 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 14, 28 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 14, 0 ],
//CHECK-NEXT:      "outEnd": [ 27, 27, 63 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/cluster_0",
//CHECK-NEXT:    "taskId": 5,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 10 ],
//CHECK-NEXT:    "updateBarriers": [ 11 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 14, 28 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 7, 14 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 0, 0 ],
//CHECK-NEXT:      "outEnd": [ 13, 6, 63 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/cluster_1",
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "taskId": 5,
//CHECK-NEXT:    "numVariants": 1,
//CHECK-NEXT:    "maxVariants": 1,
//CHECK-NEXT:    "waitBarriers": [ 10 ],
//CHECK-NEXT:    "updateBarriers": [ 11 ],
//CHECK-NEXT:    "workloadIds": [ 0 ],
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 14, 28 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 7, 14 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    },
//CHECK-NEXT:    "variantInfo": [ {
//CHECK-NEXT:      "outStart": [ 0, 7, 0 ],
//CHECK-NEXT:      "outEnd": [ 13, 13, 63 ]
//CHECK-NEXT:    } ]
//CHECK-NEXT:  } ],
//CHECK-NEXT:  "swTasks": [ {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/tile_0/cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 0 ],
//CHECK-NEXT:    "updateBarriers": [ 2 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 16, 62 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 16, 62 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/tile_0/cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 0 ],
//CHECK-NEXT:    "updateBarriers": [ 2 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 16, 62 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 16, 62 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/tile_1/cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 0 ],
//CHECK-NEXT:    "updateBarriers": [ 2 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "dataIndex": 1,
//CHECK-NEXT:    "tileId": 1,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 15, 62 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 15, 62 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "data?t_Parameter/converted_to_f16/tile_1/cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 0 ],
//CHECK-NEXT:    "updateBarriers": [ 2 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "dataIndex": 1,
//CHECK-NEXT:    "tileId": 1,
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 15, 62 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 3, 15, 62 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/tile_0/cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 13 ],
//CHECK-NEXT:    "updateBarriers": [ 14 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "dataIndex": 2,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 4, 14 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 4, 14 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/tile_0/cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 13 ],
//CHECK-NEXT:    "updateBarriers": [ 14 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "dataIndex": 2,
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 4, 14 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 4, 14 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/tile_1/cluster_0",
//CHECK-NEXT:    "waitBarriers": [ 13 ],
//CHECK-NEXT:    "updateBarriers": [ 14 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "dataIndex": 3,
//CHECK-NEXT:    "tileId": 1,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 14 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 14 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  }, {
//CHECK-NEXT:    "name": "relu2?t_Relu/converted_to_f32/tile_1/cluster_1",
//CHECK-NEXT:    "waitBarriers": [ 13 ],
//CHECK-NEXT:    "updateBarriers": [ 14 ],
//CHECK-NEXT:    "taskType": "",
//CHECK-NEXT:    "clusterSize": 4,
//CHECK-NEXT:    "dataIndex": 3,
//CHECK-NEXT:    "tileId": 1,
//CHECK-NEXT:    "clusterId": 1,
//CHECK-NEXT:    "tensorInfo": {
//CHECK-NEXT:      "inputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 14 ],
//CHECK-NEXT:        "elemType": "f16"
//CHECK-NEXT:      } ],
//CHECK-NEXT:      "outputs": [ {
//CHECK-NEXT:        "dimensions": [ 1, 64, 3, 14 ],
//CHECK-NEXT:        "elemType": "f32"
//CHECK-NEXT:      } ]
//CHECK-NEXT:    }
//CHECK-NEXT:  } ]
//CHECK-NEXT:}
