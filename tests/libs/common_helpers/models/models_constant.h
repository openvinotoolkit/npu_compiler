//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

namespace Models {

struct ModelDesc {
    char pathToModel[FILENAME_MAX];
    size_t width;
    size_t height;
};

const ModelDesc squeezenet1_1 = {
        "/KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8",
        227,
        227
};

const ModelDesc googlenet_v1 = {
        "/KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8",
        224,
        224
};

const ModelDesc yolov3 = {
        "/KMB_models/INT8/public/yolo_v3/yolo_v3_tf_dense_int8_IRv10",
        416,
        416
};
}
