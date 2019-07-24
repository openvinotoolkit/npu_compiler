//
// Copyright 2017-2018 Intel Corporation.
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

#include <cstdint>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <set>
#include <utility>

#include <ie_icnn_network.hpp>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/perf_report.hpp>
#include <vpu/utils/logger.hpp>

#include <frontend_mcm.hpp>
#include <kmb_config.h>

#ifdef ENABLE_MCM_COMPILER
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include "include/mcm/target/target_descriptor.hpp"

namespace vpu {
namespace KmbPlugin {

namespace ie = InferenceEngine;

//
// CompilationConfig
//

struct CompiledGraph final {
    using Ptr = std::shared_ptr<CompiledGraph>;

    std::vector<char> blob;
    std::pair<char*, size_t> blobHeader;

    std::string networkName;

    int networkBatch = 0;

    std::vector<StageMetaInfo> stagesMeta;
    int numActiveStages = 0;

    int inputBufSize = 0;
    int outputBufSize = 0;
};

//
// compileNetwork
//

void compileMcm(
        const ie::ICNNNetwork& network,
        const KmbConfig& config,
        mv::CompilationUnit& unit,
        std::vector<char>& blob);
//
// getSupportedLayers
//

std::set<std::string> getSupportedLayersMcm(
        const ie::ICNNNetwork& network,
        mv::OpModel& pCompiler);

//
// Blob version and checks
//

const int BLOB_MAGIC_NUMBER  = 9709;
const int BLOB_VERSION_MAJOR = 2;
const int BLOB_VERSION_MINOR = 0;

}  // namespace KmbPlugin

}  // namespace vpu
#endif
