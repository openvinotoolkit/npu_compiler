# Auto Tuning in NPU Compiler

This guide explains what auto tuning is in the NPU compiler and when to use it.

## What is auto tuning?

Auto tuning is a strategy-search method used by the compiler to find a better
execution plan for selected workloads.

At a high level, auto tuning explores a large candidate space that combines:

- multi-cluster strategy selection
- temporal tiling strategy selection

For each candidate, the compiler relies on DPU cost estimation to compare
options and choose the best strategy.

## Why auto tuning exists

A fixed heuristic can be fast, but it may miss the best strategy for some
workloads. Auto tuning spends additional compile-time effort to search a wider
space and improve strategy quality.

The main objective is better runtime performance through better strategy
selection.

## Where auto tuning is most useful

Auto tuning is most beneficial for compute-bound models, especially when the
model contains large single-layer executions (for example, large convolution
layers).

In these cases, choosing the right multi-cluster and temporal tiling strategy
can have a clear impact on end-to-end latency.

## Current scope and limitations

The current search space is tuned primarily for convolution workloads.

As a result:

- It can often provide near-optimal or optimal strategies for compute-bound
  convolution layers.
- It is not expected to be equally optimal for non-convolution operations.

## How to use auto tuning

This section describes a practical workflow to collect auto tuning strategies,
merge useful entries into the platform cache, and regenerate the binary table.

### 1. Collect auto tuning strategies from model compilation

Before collecting strategies, ensure the build and compile configuration below:

- Export `ENABLE_VPUNN_PROFILING_SERVICE=TRUE` to enable VPUNN online
  profiling. For service configuration and overrides, CSV serialization, and
  HTTP diagnostics, refer to the [VPUNN online-profiling environment variables](https://github.com/intel/npu-nn-cost-model/blob/accdd017657ed58d9e656f64e5db827360c3243d/docs/online_profiling.md).
- Enable full-search tiling during compilation:
  `enable-tiling-full-search-space=true`.

Without online profiling, DPU cost is predicted by the VPUNN trained model.
With online profiling enabled, full-search decisions can rely on more accurate
DPU cost data.

Then enable strategy-table dump and compile the model as usual:

```bash
export ENABLE_VPUNN_PROFILING_SERVICE=TRUE
export VPUX_PTC_WRITE_ENABLED=1
# Optional: override the output path (default: precomputed_strategy_table.json)
export VPUX_PTC_WRITE_LOCATION=/path/to/precomputed_strategy_table.json
```

Example compile option in config (`NPU_COMPILATION_MODE_PARAMS`):

```text
enable-tiling-full-search-space=true
```

Note: `VPUX_PTC_WRITE_ENABLED` is available in developer builds only
(`VPUX_DEVELOPER_BUILD`).

After compilation, auto tuning strategies are written to a JSON file
(typically `precomputed_strategy_table.json`, depending on environment setup).

For details about JSON fields and semantics, refer to:
`src/vpux_compiler/docs/design/precomputed_strategy_table.md`.

### 2. Select and merge candidate entries

For now, it is recommended to select only convolution entries and append them to
the platform cache JSON:

- PTL: `artifacts/precomputed_strategy_cache/npu50xx_precomputed_strategy.json`

Each entry is keyed by a 64-bit descriptor hash. Before appending a new
entry, check whether the same hash key already exists in the platform cache.

If an entry with the same layer hash already exists, it usually means the layer
is already cached and the new entry can be ignored.

However, a different strategy for the same layer may still improve full-model
performance due to interactions with other passes and stages, for example:

- vertical fusion
- scheduling
- DMA optimizations

In this case, validate whether the new strategy is better at the single-layer
execution level than the existing cached one. Update the cache only if the new
strategy wins this single-layer comparison.

### 3. Regenerate platform PTC

Convert the updated JSON cache into the binary `.ptc` file.

For PTL:

```bash
python3 tools/precomputed-strategy-table/convert_precomputed_strategy_table.py \
  artifacts/precomputed_strategy_cache/npu50xx_precomputed_strategy.json \
  -o artifacts/precomputed_strategy_cache/npu50xx_precomputed_strategy.ptc
```

### 4. Rebuild compiler and retest model performance

Rebuild the compiler so the updated `.ptc` is embedded, then rerun performance
evaluation on the target model.

## Troubleshooting

### No strategy JSON is generated

- Confirm developer build is used (`VPUX_DEVELOPER_BUILD`).
- Confirm `VPUX_PTC_WRITE_ENABLED=1` is exported in the same shell that runs
  model compilation.
- If `VPUX_PTC_WRITE_LOCATION` is set, verify the path is writable.

### PTC hit count is lower than expected

- Ensure INFO-level compiler logs are enabled, then check whether the model
  compile log contains the `[PTC] Done` summary from
  `tiling-strategy-assignment`:

  ```text
  [INFO] 22:59:02.033 [tiling-strategy-assignment]   [PTC] Done: 5 hit(s), 13 miss(es) out of 18 operation(s)
  ```

  In this example, 5 operations matched entries in the embedded PTC and 13 did
  not.
- Compare the hash-affecting fields between the generated JSON entry and the
  operation being compiled.

  The following fields affect the hash:

  - op name
  - input 0 descriptor: rank, shape, layout, element type
  - input 1 descriptor: rank, shape, layout, element type, or a missing-input
    sentinel
  - output 0 descriptor: rank, shape, layout, element type
  - strides for NCE operations
  - padding for NCE operations
  - target device string, for example `NPU50XX`
  - engine kind: `SHAVE`, `SCL`, or `DCIM`
  - DPU tile count

- Pay special attention to device, DPU tile count, tensor element type, layout,
  shape, strides, and padding.

### Existing cache entry differs from new result

- A hash collision is unlikely; most conflicts are the same layer descriptor
  with different selected strategy.
- The strategy fields (`spatial_tiling`, `temporal_tiling`, `pipeline_mode`) and
  the inspection-only fields (`layer_name`, `strategy_cost_cycles`) are not part
  of the hash key. See `precomputed_strategy_table.md` for the full schema.
- Keep the existing entry by default.
- Replace it only if single-layer validation confirms the new strategy is
  better.

### PTC conversion command fails

- Verify input JSON path and output `.ptc` path are correct for the target
  platform (`npu50xx` for PTL).
- Ensure the JSON is valid and entries are not manually malformed.
- Re-run conversion after fixing invalid entries.

### Full-model performance does not improve

- Auto tuning strategy quality is evaluated at single-layer level, while
  end-to-end model latency also depends on vertical fusion, scheduling,
  memory pressure, DMA behavior, and cost-model accuracy.
- Keep strategy updates that are validated at single-layer level and re-check
  full-model performance after rebuild to confirm net impact.

### PTC-embedded blob underperforms the auto-tuning blob

- Confirm the PTC hit count first. A miss means the compiler did not apply the
  cached strategy for that operation.
- If the entry hits, check the cost model used to generate the PTC-embedded blob. Auto tuning
  runs with `ENABLE_VPUNN_PROFILING_SERVICE=TRUE`, which uses VPUNN online
  profiling and provides more accurate DPU cost than the VPUNN trained model.
