// RUN: vpux-translate --import-HWTEST %s

{
    "case_type": "ZMajorConvolution",
    "input": {
        "shape": [
            1,
            32,
            16,
            16
        ],
        "dtype": "int4",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": -8,
            "high_range": 7
        }
    },
    "weight": {
        "shape": [
            64,
            32,
            1,
            1
        ],
        "dtype": "int4",
        "quantization": {
            "scale": 1.0,
            "zeropoint": 0,
            "low_range": -8,
            "high_range": 7
        }
    },
    "output": {
        "shape": [
            1,
            64,
            16,
            16
        ],
        "dtype": "int4",
        "quantization": {
            "scale": 64,
            "zeropoint": 0,
            "low_range": -8,
            "high_range": 7
        }
    },
    "conv_op": {
        "stride": [
            1,
            1
        ],
        "pad": [
            0,
            0,
            0,
            0
        ],
        "group": 1,
        "dilation": 1,
        "compress": false,
        "mpe_cub": "CUBOID_16x16"
    },
    "activation": {
        "name": null
    }
}
