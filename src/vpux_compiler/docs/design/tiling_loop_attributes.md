# Tiling Loop Attributes

## Purpose

Static tiling unrolls one logical operation into multiple operations. After unrolling, the IR order alone
does not reliably identify either the tiled region an operation belongs to or the logical tile it computes.

Loop attributes preserve that identity for downstream loop-region analysis, scheduling, allocation, and copy
optimization. They are a compiler IR contract for every pass that materializes static tiles, including generic tiling
and specialized tiling passes.

## Attributes

| Attribute | Meaning | Used by |
|---|---|---|
| `tiling_loop_index` | Identity of a normal static tiling region. | `ApplyTilingPass`, FlashSDPA tiling. |
| `vf_loop_index` | Identity of a Vertical Fusion (VF) region. | `VfTilingPass`. |
| `vf_loop_tile_index` | Physical VF unroll lane. | VF-specific transformations. |
| `loop_tile_coord` | Logical coordinate of a materialized operation's output tile. | Loop-region analysis, scheduling, allocation, and copy optimization. |

`tiling_loop_index` and `vf_loop_index` are separate namespaces. A normal tiled region uses
`tiling_loop_index`; a VF region uses `vf_loop_index`. A materialized compute operation must also carry
`loop_tile_coord` so downstream consumers can identify the logical tile without deriving it from IR order.

`loop_tile_coord` is a dictionary with two arrays:

```mlir
loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 2, 1>}
```

- `axes` are ascending logical `Dim` values, independent of physical tensor layout and unroll order.
- `indices` are zero-based offset ordinals for the corresponding axes.
- An axis absent from `axes` is fully covered by the operation and is not tiled for that operation.

For example, a 4D activation with logical shape `[N, C, H, W]` and strategy `[1, 4, 2, 1]` has tiled axes C and H.
The example above identifies C tile 2 and H tile 1. Channel-first and spatial-first unrolling can materialize those
coordinates in a different sequence, but both must retain the same `axes` and `indices` meaning.

## From Unrolling to Scheduling

Consider a normal tiling pass that unrolls a convolution over H into two operations:

```mlir
// Before unrolling
%0 = VPU.NCE.Convolution %input, %weights {tilingStrategy = [1, 1, 2, 1]}

// After unrolling
%tile0 = VPU.NCE.Convolution %input0, %weights {
  tiling_loop_index = 7 : i64,
  loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}
}
%tile1 = VPU.NCE.Convolution %input1, %weights {
  tiling_loop_index = 7 : i64,
  loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 1>}
}
%result = VPU.Concat %tile0, %tile1
```

The shared `tiling_loop_index` identifies one logical tiled region. `loop_tile_coord` distinguishes its H tiles.
When VPU operations are bufferized, the conversion copies the attributes to the resulting VPUIP compute operations.
Loop-region analysis then groups operations by the region identity and uses the explicit coordinates to associate
compute operations, operands, and values with their logical tiles. Scheduling, allocation, and copy optimization use
those facts instead of reconstructing tile identity from the order in which unrolled operations appear in IR.

VF follows the same principle with `vf_loop_index`. `vf_loop_tile_index` retains the physical VF unroll lane, while
`loop_tile_coord` represents each operation's actual output coverage. A shared internal producer may cover only H in
a C/H tiled VF region; its coordinate must contain only H, rather than copying the final output's C/H coordinate.

## Producer Contract

Any pass that materializes static tiles must do all of the following:

1. Assign a module-wide unused `tiling_loop_index` for a normal tiled region, or a module-wide unused
   `vf_loop_index` for a VF region.
2. Attach `loop_tile_coord` to every materialized compute producer, derived from that producer's actual output tiles.
3. For VF, also attach `vf_loop_tile_index` when the physical VF lane is required by VF-specific consumers.
4. Preserve all existing loop attributes when cloning, rewriting, or lowering a tiled operation. The common helpers
   `getLoopAttributes()` and `copyLoopAttributes()` handle VPU-to-VPU rewrites; VPU-to-VPUIP conversions must forward
   the returned attributes to their newly created compute operation.
5. Define the coordinate after any specialized reindexing. A no-op tiling stage must not leave a coordinate that
   describes an earlier loop domain.

Current producers are:

| Producer | Region identity assignment | Tile coordinate attachment |
|---|---|---|
| `ApplyTilingPass` | Assigns `tiling_loop_index` before generic unrolling. | `generateTiling()` assigns the coordinate to each clone. |
| `VfTilingPass` | Assigns `vf_loop_index` before VF unrolling. | VF rewriters assign per-operation coordinates when they materialize clones. |
| FlashSDPA tiling | Assigns or reassigns `tiling_loop_index` for its final loop domain. | FlashSDPA tiling assigns coordinates after specialized tiling and reindexing. |

## Adding a Tiling-Like Pass

Before adding a pass that unrolls, clones, or otherwise materializes operations for static tiling:

1. Identify the final logical loop domain and choose the normal or VF region identity attribute.
2. Obtain a module-wide index with `getNextTilingLoopIndex()` or `getNextVFLoopIndex()`.
3. Derive each materialized compute operation's coordinate with `getTileCoordinates()` and serialize it with
   `getTileCoordinateAttr()`.
4. Ensure any pass that creates replacement compute operations propagates the attrs.
5. Add exact LIT checks for the region index and complete `loop_tile_coord`; do not use a broad match that can hide
   an incorrect coordinate.

At minimum, the tests must cover every relevant multi-axis, shared-producer, or reindex/no-op path. Consumers must
not infer tile identity from IR order when a region index and `loop_tile_coord` are present.

## SCF Direction

These attributes are the current contract for statically materialized tiled regions. SCF can represent iteration
identity structurally through loop regions and induction variables, but is not yet a replacement for every generic
and specialized static tiling path. A future SCF migration must provide an equivalent downstream identity contract,
or migrate all producers and consumers together.