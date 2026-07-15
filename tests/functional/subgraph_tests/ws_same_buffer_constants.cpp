//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common/functions.h"

#include "shared_test_classes/subgraph/ws_base.hpp"

namespace ov::test::behavior {

class WsSameBufferConstantsSubGraphTest : public WsBaseTest {
public:
    WsSameBufferConstantsSubGraphTest(): WsBaseTest("same_buf_constants_test") {
    }

private:
    void SetUp() override {
        inType = ov::element::f32;
        const ov::Shape inputShape{1, 2, 3};

        init_input_shapes(static_shapes_to_test_representation({inputShape, inputShape}));

        const auto model = buildSingleWsFriendlyNetwork(inType, inputShape);

        // Note: this test requires model serialization and de-serialization,
        // with compression enabled. so that the OV constants end up pointing to
        // the same buffer.
        ov::pass::Serialize(_xmlPath, _binPath).run_on_model(model);
        function = core->read_model(_xmlPath, _binPath);
    }
};

TEST_F(WsSameBufferConstantsSubGraphTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(WsSameBufferConstantsSubGraphTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_F(WsSameBufferConstantsSubGraphTest, NPU5020_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test::behavior
