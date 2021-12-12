//
// Copyright 2020 Intel Corporation.
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

//#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include <mlir/IR/BuiltinTypes.h>
#include <host_parsed_inference.h>

using namespace vpux;

//
// MappedInferenceOp
//

void vpux::VPUIPRegMapped::MappedInferenceOp::serialize(std::vector<char>& buffer) {

    host_parsing::MappedInference mi;
    memset(reinterpret_cast<void*>(&mi), 0, sizeof(host_parsing::MappedInference));

    mi.dmaTasks[0].count = dmaCount();
    mi.invariants.count = invariantCount();
    mi.variants.count = variantCount();
    mi.actKInvocations.count = actInvocationsCount();

    char* ptrCharTmp = reinterpret_cast<char*>(&mi);
    for (long unsigned i = 0; i < sizeof(host_parsing::MappedInference); i++) {
        buffer.push_back(*(ptrCharTmp + i));
    }

    return;
}
