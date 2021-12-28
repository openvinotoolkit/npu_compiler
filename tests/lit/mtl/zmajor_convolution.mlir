// RUN: vpux-translate --import-HWTEST %s

{
    "case_type": "ZMajorConvolution",
    "network": "",
    "layer_name": "conv2d_u8_to_u8_unit_test",
    "input": {
        "shape": [
            1,
            256,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 0.01,
            "zeropoint": 127,
            "low_range": 0,
            "high_range": 63
        }
    },
    "weight": {
        "shape": [
            64,
            256,
            1,
            1
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 0.01,
            "zeropoint": 127,
            "low_range": 0,
            "high_range": 63
        }
    },
    "output": {
        "shape": [
            1,
            64,
            16,
            16
        ],
        "dtype": "uint8",
        "quantization": {
            "scale": 0.01,
            "zeropoint": 127,
            "low_range": 0,
            "high_range": 63
        }
    },
    "conv_op": {
        "stride": [
            1,
            1
        ],
        "pad": [
            0,
            0
        ],
        "group": 1,
        "dilation": 1
    },
    "output_order": "nhwc",
    "activation": {
        "name": null
    }
}
