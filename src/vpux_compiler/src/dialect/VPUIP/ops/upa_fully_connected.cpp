//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::FullyConnectedUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::FullyConnectedParamsBuilder builder(writer);
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_FullyConnectedParams});
}
