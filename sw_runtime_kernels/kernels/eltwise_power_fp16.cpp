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

#include <math.h>
#include <param_eltwise.h>
#include <eltwise_base.h>

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

#define ELTWISE_FN(a,b) (powf(a,b))
ELTWISE_BINARY_OP(power_fp16);

}
}  // namespace shave_lib
}  // namespace nn