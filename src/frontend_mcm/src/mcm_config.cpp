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

#include <cpp_interfaces/exception2status.hpp>
#include <map>
#include <mcm_config.hpp>
#include <private_vpu_compiler_config.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace vpu;

const std::unordered_set<std::string>& MCMConfig::getCompileOptions() const {
    static const std::unordered_set<std::string> options = merge(
            vpux::VPUXConfig::getCompileOptions(), {
                                                           VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR_PATH),
                                                           VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR),
                                                           VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR_PATH),
                                                           VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR),
                                                           VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB),
                                                           VPU_COMPILER_CONFIG_KEY(PARSING_ONLY),
                                                           VPU_COMPILER_CONFIG_KEY(GENERATE_JSON),
                                                           VPU_COMPILER_CONFIG_KEY(GENERATE_DOT),
                                                           VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS_PATH),
                                                           VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS),
                                                           VPU_COMPILER_CONFIG_KEY(LOG_LEVEL),
                                                           VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT),
                                                           VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT),
                                                           VPU_COMPILER_CONFIG_KEY(SERIALIZE_CNN_BEFORE_COMPILE_FILE),
                                                           VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS),
                                                           VPU_COMPILER_CONFIG_KEY(COMPILATION_PASS_BAN_LIST),
                                                           VPU_COMPILER_CONFIG_KEY(SCALE_FUSE_INPUT),
                                                           VPU_COMPILER_CONFIG_KEY(REFERENCE_MODE),
                                                           VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT),
                                                           VPU_COMPILER_CONFIG_KEY(ALLOW_U8_INPUT_FOR_FP16_MODELS),
                                                           VPU_COMPILER_CONFIG_KEY(SCALESHIFT_FUSING),
                                                           VPU_COMPILER_CONFIG_KEY(REMOVE_PERMUTE_NOOP),
                                                           VPU_COMPILER_CONFIG_KEY(ALLOW_PERMUTE_ND),
                                                           VPU_COMPILER_CONFIG_KEY(NUM_CLUSTER),
                                                           VPU_COMPILER_CONFIG_KEY(OPTIMIZE_INPUT_PRECISION),
                                                   });

    return options;
}

void MCMConfig::parse(const std::map<std::string, std::string>& config) {
    static const std::unordered_map<std::string, LogLevel> logLevels = {
            {CONFIG_VALUE(LOG_NONE), LogLevel::None},       {CONFIG_VALUE(LOG_ERROR), LogLevel::Error},
            {CONFIG_VALUE(LOG_WARNING), LogLevel::Warning}, {CONFIG_VALUE(LOG_INFO), LogLevel::Info},
            {CONFIG_VALUE(LOG_DEBUG), LogLevel::Debug},     {CONFIG_VALUE(LOG_TRACE), LogLevel::Trace}};

    vpux::VPUXConfig::parse(config);

    setOption(_mcmLogLevel, logLevels, config, VPU_COMPILER_CONFIG_KEY(LOG_LEVEL));

    setOption(_mcmTargetDesciptorPath, config, VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR_PATH));
    setOption(_mcmTargetDesciptor, config, VPU_COMPILER_CONFIG_KEY(TARGET_DESCRIPTOR));

    setOption(_mcmCompilationDesciptorPath, config, VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR_PATH));
    setOption(_mcmCompilationDesciptor, config, VPU_COMPILER_CONFIG_KEY(COMPILATION_DESCRIPTOR));

    setOption(_mcmGenerateBlob, switches, config, VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB));
    setOption(_mcmGenerateJSON, switches, config, VPU_COMPILER_CONFIG_KEY(GENERATE_JSON));
    setOption(_mcmGenerateDOT, switches, config, VPU_COMPILER_CONFIG_KEY(GENERATE_DOT));

    setOption(_mcmParseOnly, switches, config, VPU_COMPILER_CONFIG_KEY(PARSING_ONLY));

    setOption(_mcmCompilationResultsPath, config, VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS_PATH));
    setOption(_mcmCompilationResults, config, VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS));

    setOption(_eltwiseScalesAlignment, switches, config, VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT));
    setOption(_concatScalesAlignment, switches, config, VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT));

    setOption(_serializeCNNBeforeCompileFile, config, VPU_COMPILER_CONFIG_KEY(SERIALIZE_CNN_BEFORE_COMPILE_FILE));

    setOption(_customLayers, config, VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS));

    setOption(_mcmCompilationPassBanList, config, VPU_COMPILER_CONFIG_KEY(COMPILATION_PASS_BAN_LIST));

    setOption(_scaleFuseInput, switches, config, VPU_COMPILER_CONFIG_KEY(SCALE_FUSE_INPUT));

    setOption(_referenceMode, switches, config, VPU_COMPILER_CONFIG_KEY(REFERENCE_MODE));

    setOption(_allowNCHWLayoutForMcmModelInput, switches, config, VPU_COMPILER_CONFIG_KEY(ALLOW_NCHW_MCM_INPUT));

    setOption(_allowU8InputForFp16Models, switches, config, VPU_COMPILER_CONFIG_KEY(ALLOW_U8_INPUT_FOR_FP16_MODELS));

    setOption(_scaleShiftFusing, switches, config, VPU_COMPILER_CONFIG_KEY(SCALESHIFT_FUSING));

    setOption(_optimizeInputPrecision, switches, config, VPU_COMPILER_CONFIG_KEY(OPTIMIZE_INPUT_PRECISION));

    setOption(_removePermuteNoOp, switches, config, VPU_COMPILER_CONFIG_KEY(REMOVE_PERMUTE_NOOP));
    setOption(_allowPermuteND, switches, config, VPU_COMPILER_CONFIG_KEY(ALLOW_PERMUTE_ND));
    setOption(_numberOfClusters, config, VPU_COMPILER_CONFIG_KEY(NUM_CLUSTER), parseInt);
    IE_ASSERT(0 <= _numberOfClusters && _numberOfClusters <= 4)
            << "MCMConfig::parse attempt to set invalid number of clusters: '" << _numberOfClusters
            << "', valid numbers are from 0 to 4";
}
