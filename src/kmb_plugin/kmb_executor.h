//
// Copyright 2019 Intel Corporation.
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

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <iomanip>
#include <utility>

#include <ie_icnn_network.hpp>

#ifdef ENABLE_VPUAL
#include <GraphManagerPlg.h>
#include <PlgStreamResult.h>
#include <PlgTensorSource.h>

#include <NNFlicPlg.h>
#include <Pool.h>
#include <cma_allocation_helper.h>
#include <MemAllocator.h>
#include <mvMacros.h>
#endif

#include <kmb_config.h>
#include "kmb_allocator.h"

namespace vpu {
namespace KmbPlugin {

class KmbExecutor {
public:
    explicit KmbExecutor(const std::shared_ptr<KmbConfig>& config);
    ~KmbExecutor() = default;

    void allocateGraph(const std::vector<char> &graphFileContent, const char* networkName);

    void deallocateGraph();

    void queueInference(void *input_data, size_t input_bytes, void *result_data, size_t result_bytes);

    void getResult(void *result_data, unsigned int result_bytes);

    const InferenceEngine::InputsDataMap& getNetworkInputs() const { return m_networkInputs; }
    const InferenceEngine::OutputsDataMap& getNetworkOutputs() const { return m_networkOutputs; }

    const std::shared_ptr<KmbConfig>& _config;

private:
        unsigned int _numStages = 0;
        Logger::Ptr _logger;
#ifdef ENABLE_VPUAL
        std::shared_ptr<GraphManagerPlg> gg;
        std::shared_ptr<PlgTensorSource> plgTensorInput_;
        std::shared_ptr<PlgStreamResult> plgTensorOutput_;
        std::shared_ptr<RgnAllocator> RgnAlloc;

        std::shared_ptr<NNFlicPlg> nnPl;

        void *blob_file;
        void *rgnAllocatorBuffer;
        std::shared_ptr<BlobHandle_t> BHandle;

        std::shared_ptr<PlgPool<TensorMsg>> plgPoolOutputs;

        std::shared_ptr<Pipeline> pipe;
#endif
        InferenceEngine::InputsDataMap  m_networkInputs;
        InferenceEngine::OutputsDataMap m_networkOutputs;

        std::shared_ptr<KmbAllocator> allocator;
        void initVpualObjects();

        int xlinkChannelIn;
        int xlinkChannelOut;

        uint32_t _outTensorLen;
        uint32_t _outTensorAddr;
};

typedef std::shared_ptr<KmbExecutor> KmbExecutorPtr;

}  // namespace KmbPlugin
}  // namespace vpu
