/*
 * {% copyright %}
 */
#pragma once

#include <mv_types.h>
#include <nn_resource_locator.h>
//#include <nn_inference_runtime_types.h>
#include <ShaveCtrl.h>

using namespace nn::common_runtime;

namespace nn {
namespace inference_runtime {
namespace shaves {

using namespace act_runtime;

class ShaveManager {
public:
    ShaveManager(const common_runtime::StaticMapping &sMapping);
    ~ShaveManager();

    void startNNShavesForTile(const uint32_t tile);
    void startNNShavesForTiles();
    void stopNNShavesForTileMask(uint32_t mask);
    void startNNShavesForTileMask(uint32_t mask);

    /// These functions reduce to a noop if the shaves are already running, they match the requested configuration, and
    /// forceRestart = false
    /// @param tile description
    void startActShavesForTile(const uint32_t tile, const ActKernelRuntimeConfigs &cfgs, bool forceRestart = false);
    void startActShavesForTiles(const ActKernelRuntimeConfigs &cfgs, bool forceRestart = false);

    void stopNNShavesForTile(const uint32_t tile);
    void stopNNShavesForTiles();

    void stopActShavesForTile(const uint32_t tile);
    void stopActShavesForTiles();

private:
    void startActShaves(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs);
    bool processConfigChanges(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs);
    void initActRtCodeBuffer(const uint8_t tile);

#ifdef CONFIG_VALIDATION_APP_ENABLED
    void initActRtStacksAndDatas(const uint8_t tile, const ActKernelRuntimeConfigs &cfgs);
#endif

    const StaticMapping &cmxMapping;

    ShaveHandle *actShvHnd[4/*AS_TOTAL*/];
    ShaveHandle *nnShvHnd[SNN_TOTAL];

    uint32_t actShvStacks[4/*AS_TOTAL*/]{0};

    actRuntimeEntry actShvEntries[MAX_TILES]{0};
    uint8_t *actShvTextsBuffers[MAX_TILES]{nullptr};
    uint32_t actShvTxtBuffSizes[MAX_TILES]{0};
};

} // namespace shaves
} // namespace inference_runtime
} // namespace nn
