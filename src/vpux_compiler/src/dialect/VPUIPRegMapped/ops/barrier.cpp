//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "host_parsed_inference.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/utils/core/format.hpp"


using namespace vpux;

//
// ConfigureBarrierOp
//

void vpux::VPUIPRegMapped::ConfigureBarrierOp::serialize(std::vector<char>& buffer) {

    host_parsing::BarrierWrapper barrier;

    barrier.next_same_id = next_same_id();
    barrier.real_id = id();
    barrier.consumer_count = consumer_count().getValueOr(0); //make it fixed after dialect refactor......
    barrier.producer_count = producer_count().getValueOr(0);

    char* ptrCharTmp = reinterpret_cast<char*>(&barrier);
    for (size_t i = 0; i < sizeof(barrier); i++) {
        buffer.push_back(ptrCharTmp[i]);
    }
}

size_t vpux::VPUIPRegMapped::ConfigureBarrierOp::getBinarySize() {
    return sizeof(host_parsing::BarrierWrapper);
}
