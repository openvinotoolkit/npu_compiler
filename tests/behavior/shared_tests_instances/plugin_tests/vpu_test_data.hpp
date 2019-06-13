// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

// correct params
#define BEH_KMB BehTestParams("kmbPlugin", model_path_fp16, weights_path_fp16, Precision::FP32)
#define BEH_HETERO BehTestParams("HeteroPlugin", model_path_fp32, weights_path_fp32, Precision::FP32)

// all parameters are unsupported - reversed
#define BEH_US_ALL_KMB     BehTestParams("kmbPlugin", model_path_q78, weights_path_q78, Precision::Q78)

const BehTestParams supportedValues[] = {
        BEH_KMB,
};

const BehTestParams requestsSupportedValues[] = {
        BEH_KMB,
};

const BehTestParams allInputSupportedValues[] = {
BEH_KMB, BEH_KMB.withIn(Precision::U8), BEH_KMB.withIn(Precision::FP16),
};

const BehTestParams allOutputSupportedValues[] = {
        BEH_KMB, BEH_KMB.withOut(Precision::FP16),
};

const BehTestParams typeUnSupportedValues[] = {
BEH_KMB.withIn(Precision::Q78), BEH_KMB.withIn(Precision::U16), BEH_KMB.withIn(Precision::I8),
    BEH_KMB.withIn(Precision::I16), BEH_KMB.withIn(Precision::I32),
};

const BehTestParams allUnSupportedValues[] = {
        BEH_US_ALL_KMB,
};

const std::vector<BehTestParams> withCorrectConfValues = {
    BEH_KMB.withConfig({ { KEY_VPU_COPY_OPTIMIZATION, NO } }),
    BEH_KMB.withConfig({ { KEY_VPU_LOG_LEVEL, LOG_DEBUG } }),
    BEH_KMB.withConfig({ { KEY_VPU_IGNORE_UNKNOWN_LAYERS, YES } }),
    BEH_KMB.withConfig({ { KEY_VPU_INPUT_NORM, "0.5" } }),
    BEH_KMB.withConfig({ { KEY_VPU_INPUT_BIAS, "0.1" } }),
    BEH_KMB.withConfig({ { KEY_VPU_HW_STAGES_OPTIMIZATION, YES } }),
    BEH_KMB.withConfig({ { KEY_VPU_NONE_LAYERS, "Tile" } }),
    BEH_KMB.withConfig({ { KEY_VPU_HW_ADAPTIVE_MODE, "YES" } }),
    BEH_KMB.withConfig({ { KEY_VPU_NUMBER_OF_SHAVES, "5" }, { KEY_VPU_NUMBER_OF_CMX_SLICES, "5" } }),
    BEH_KMB.withConfig({ { KEY_VPU_HW_INJECT_STAGES, "YES" } }),
    BEH_KMB.withConfig({ { KEY_VPU_HW_POOL_CONV_MERGE, "YES" } }),
};

const std::vector<BehTestParams> withCorrectConfValuesPluginOnly = {
};

const std::vector<BehTestParams> withCorrectConfValuesNetworkOnly = {
};

const BehTestParams withIncorrectConfValues[] = {
    BEH_KMB.withConfig({ { KEY_VPU_COPY_OPTIMIZATION, "ON" } }),
    BEH_KMB.withConfig({ { KEY_VPU_LOG_LEVEL, "VERBOSE" } }),
    BEH_KMB.withConfig({ { KEY_VPU_INPUT_NORM, "0.0" } }),
    BEH_KMB.withConfig({ { KEY_VPU_IGNORE_UNKNOWN_LAYERS, "ON" } }),
};
