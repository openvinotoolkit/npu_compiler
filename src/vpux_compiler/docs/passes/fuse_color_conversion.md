# FuseColorConversion — YUV to RGB coefficient derivation

## 1. Brief

`FuseColorConversion` (`src/vpux_compiler/src/dialect/IE/transforms/passes/fuse_color_conversion.cpp`)
recognises a YUV to RGB colour-space conversion that a framework exported as ordinary tensor
operations, and replaces it with a single `IE.YuvToRgb` op that lowers to the `nv12_to_rgb` SHAVE
kernel.

The important consequence, and the reason this document exists: **the pass discards the convolution
matrix**. The kernel has BT.601 hardcoded. Only two things survive the rewrite:

| what the pass reads | where | what it becomes |
| --- | --- | --- |
| `\|w[8]\|` vs `\|w[2]\|` | `detectOutputColorFormat` | `outFmt` = RGB or BGR |
| the bias, as `mean(bias[i] / standardBias[i])` | `fuseColorConversionPattern` | the `scale` argument passed to the kernel |

Everything else — all nine weights, the input FakeQuantizes, the output FakeQuantize — is deleted.

So a model whose coefficients are *not* BT.601 will still fuse, still compile, and then compute
something different on the NPU than the framework reference computes on the CPU. Nothing warns you.
Any test or model that goes through this pass must therefore use coefficients that agree with the
kernel. The rest of this document derives them.

## 2. Pattern

```
  Y  -> Convert -> AffineReshape / DynamicReshape ------------------+
                                                                    +-> Concat -> [FQ] -> Conv -> Add -> Clamp|FQ
  UV -> Convert -> Transpose(NHWC->NCHW) -> Interpolate(2x) -> [FQ]-+

                                     becomes

  Y  --+
       +-> IE.YuvToRgb(NV12 -> RGB|BGR, scale) -> Transpose -> [Multiply(scale)] -> [Clamp|FQ]
  UV --+
```

`Convert` and `FakeQuantize` are skipped transparently. The tail must be a `Clamp` or `FakeQuantize`
whose output range is `[0, 255]` or `[0, 1]` (`isValidColorOutputRange`).

Whether the scale is folded into the kernel or emitted as a separate `IE.Multiply` depends on the
`enableYuvToRgbShaveScale` option. Host compilation sets it to `true`
(`src/vpux_compiler/src/pipelines/pipeline_strategies.cpp`), so the kernel applies the scale itself.

## 3. What the kernel computes

`sw_runtime_kernels/kernels/src/asm/50xx/nv12_to_rgb.asm`:

```
R = clip( 1.164*(Y-16) +     0*(U-128) + 1.596*(V-128) ) * scale
G = clip( 1.164*(Y-16) - 0.391*(U-128) - 0.813*(V-128) ) * scale
B = clip( 1.164*(Y-16) + 2.018*(U-128) +     0*(V-128) ) * scale
```

with `clip` to `[0, 255]`. This is BT.601 **limited range** (studio swing): Y occupies 16..235 and
U/V are centred on 128. The ASM stores the coefficients as

```
.set Y_off  16.0F32     .set Y_mul  1.164F32
.set VCTA  +1.596F32    .set VCTB  -0.813F32
.set UCTB  -0.391F32    .set UCTC  +2.018F32      // UCTA, VCTC = ZERO
```

and clamps against `0x437F0000` = 255.0. The ASM pre-multiplies the coefficients by `scale` and
scales the clamp limit to `255 * scale`, which is algebraically the same as clipping first and
scaling after. `i420_to_rgb` uses identical coefficients on three planes instead of two.

The channel written first in memory depends on the format argument: `color_format == 0` writes
R, G, B; anything else writes B, G, R.

## 4. Deriving the convolution weights

### Step 1 — the graph's algebraic form

A 1x1 convolution over three input channels followed by a bias add is, per pixel:

```
out[c] = w[c][0]*Y + w[c][1]*U + w[c][2]*V + bias[c]
```

Both index orders are dictated by the graph, not chosen:

- **Columns are Y, U, V.** The concat puts the reshaped Y plane in channel 0; the UV plane is
  `[1, H/2, W/2, 2]` transposed by `[0, 3, 1, 2]`, and the interleaved NV12 pair is U at even offset,
  V at odd — the kernel reads it the same way (`u_val = uv[i]`, `v_val = uv[i+1]`).
- **Rows follow `outFmt`.** The conv emits `[1, 3, H, W]`; the tail transpose `[0, 2, 3, 1]` makes
  out-channel *c* the *c*-th component in memory. For `outFmt = BGR` the kernel writes B first, so
  row 0 is B, row 1 G, row 2 R. For `outFmt = RGB` the rows are R, G, B.

### Step 2 — multiply out the offsets

The kernel keeps `-16` and `-128` inside the parentheses; the graph has no offsets, only a matrix
and a bias. Expanding, using R as the example:

```
R = ( 1.164*(Y-16) + 1.596*(V-128) ) * scale
  = ( 1.164*Y + 0*U + 1.596*V - 1.164*16 - 1.596*128 ) * scale
  = ( 1.164*Y + 0*U + 1.596*V - 18.624 - 204.288 ) * scale
  = 1.164*scale*Y  +  0*U  +  1.596*scale*V  +  (-222.912 * scale)
     └ w[R][Y] ┘            └ w[R][V] ┘         └── bias[R] ──┘
```

So every weight is a **kernel coefficient times `scale`**, and every bias is the **folded constant
term times `scale`**. The offsets do not vanish — they become the bias.

### Step 3 — fix `scale`

The kernel clips to `[0, 255]`, so its output spans 0..255 before scaling. `scale` is whatever maps
that onto the range the graph's tail `Clamp`/`FakeQuantize` declares:

| tail output range | scale |
| --- | --- |
| `[0, 255]` | 1 |
| `[0, 1]` | 1/255 |

The worked example below uses `[0, 1]`, so `scale = 1/255`.

### Step 4 — the target weights

Divide each kernel coefficient by 255. In BGR row order:

```
  B.Y:   1.164 / 255  =   0.004564705882
  B.U:   2.018 / 255  =   0.007913725490
  B.V:   0     / 255  =   0.000000000000
  G.Y:   1.164 / 255  =   0.004564705882
  G.U:  -0.391 / 255  =  -0.001533333333
  G.V:  -0.813 / 255  =  -0.003188235294
  R.Y:   1.164 / 255  =   0.004564705882
  R.U:   0     / 255  =   0.000000000000
  R.V:   1.596 / 255  =   0.006258823529
```

Note the **Y column is identical in all three rows**. That is true of any YUV to RGB matrix and is
the quickest sanity check on a candidate set of coefficients.

### Step 5 — the bias

Collect the constant terms, i.e. evaluate the pre-clip expression at `Y = U = V = 0`:

```
  B:  -(1.164*16) - (2.018*128)                =  -18.624 - 258.304            =  -276.928
  G:  -(1.164*16) + (0.391*128) + (0.813*128)  =  -18.624 + 50.048 + 104.064   =   135.488
  R:  -(1.164*16) - (1.596*128)                =  -18.624 - 204.288            =  -222.912
```

then multiply by `scale`:

```
  B:  -276.928 / 255  =  -1.085992156863
  G:   135.488 / 255  =   0.531325490196
  R:  -222.912 / 255  =  -0.874164705882
```

`-276.928 / 135.488 / -222.912` are exactly the `standardBiasValues` the pass hardcodes, which is
how it recovers `scale`. Equivalently `bias = -M * (16, 128, 128) * scale`.

This is the constant term of the affine map, *not* the kernel's observable output at zero input — at
`Y = U = V = 0` the R and B expressions are negative and the kernel clips them to 0. You extract the
bias algebraically, not by probing the kernel.

### Step 6 — quantise, if the model is QDQ

A quantised export feeds the filter through `Convert(f32) -> Subtract(zp) -> Multiply(scale_w)`, so a
stored code `q` means `(q - zp) * scale_w`. Inverting gives `q = round(v / scale_w) + zp`.

Note that `findUnderlyingConstant` walks back through that chain to the **raw u8 constant**, so
`detectOutputColorFormat` compares quantised codes, not dequantised values. With a zero point in the
usual range this still resolves correctly, because a zero coefficient sits at code `zp` while the
large V coefficient sits far above it — but it is comparing codes, so keep it in mind if the zero
point is unusual.

With `zp = 73` and `scale_w = 0.000043523261` (see section 6):

```
              target          / scale_w    round     q     dequantised           error
  B.Y   0.004564705882      104.8797       105    178   0.004569942405   0.000005236523
  B.U   0.007913725490      181.8275       182    255   0.007921233502   0.000007508012
  B.V   0.000000000000        0.0000         0     73   0.000000000000   0.000000000000
  G.Y   0.004564705882      104.8797       105    178   0.004569942405   0.000005236523
  G.U  -0.001533333333      -35.2302       -35     38  -0.001523314135   0.000010019198
  G.V  -0.003188235294      -73.2536       -73      0  -0.003177198053   0.000011037241
  R.Y   0.004564705882      104.8797       105    178   0.004569942405   0.000005236523
  R.U   0.000000000000        0.0000         0     73   0.000000000000   0.000000000000
  R.V   0.006258823529      143.8041       144    217   0.006267349584   0.000008526055
```

giving `{178, 255, 73, 178, 38, 0, 178, 73, 217}`, worst representation error 0.000011.
Zero coefficients land on code 73 because that is the zero point.

### Step 7 — check what the compiler recovers

- `detectOutputColorFormat`: `|w[8]| = 217` > `|w[2]| = 73` → **BGR**, matching the row order chosen
  in step 1.
- `scaleFactor = mean(bias[i] / standardBias[i])` = mean of `-1.085992/-276.928`,
  `0.531325/135.488`, `-0.874165/-222.912` = **1/255** in all three terms.

Dumping the IR (`IE_NPU_IR_PRINTING_LOCATION`, `IE_NPU_IR_PRINTING_FILTER=".*"`) should show:

```mlir
IE.YuvToRgb(%arg0, %arg1) {inFmt = NV12, outFmt = BGR, scale = 0.00392156886}
...
VPUIP.SW.Kernel.run {attrs = [1, 0.00392156886]}    kernel_type("nv12_to_rgb")
```

`attrs[0]` is the format code (0 = RGB, 1 = BGR); `attrs[1]` is the scale.

## 5. Reference values

For `scale = 1/255`. Multiply by 255 for `scale = 1` (a `[0, 255]` tail), or by any other factor,
**applying the same factor to weights and bias**.

RGB row order:

```
  weights                 Y                 U                 V
      R    0.004564705882    0.000000000000    0.006258823529
      G    0.004564705882   -0.001533333333   -0.003188235294
      B    0.004564705882    0.007913725490    0.000000000000

  bias     R  -0.874164705882    G   0.531325490196    B  -1.085992156863
```

BGR row order — the same rows, reversed:

```
  weights                 Y                 U                 V
      B    0.004564705882    0.007913725490    0.000000000000
      G    0.004564705882   -0.001533333333   -0.003188235294
      R    0.004564705882    0.000000000000    0.006258823529

  bias     B  -1.085992156863    G   0.531325490196    R  -0.874164705882
```

## 6. Note on the zero point

`zp = 73` in the dynamic test is not a free choice — it came with the source model, and it is a
**weight** quantisation parameter, unrelated to the expected Y/U/V input range. (The input range is
described by the FakeQuantize ops on the activations, `[0, 239.704999]` in that model.)

It is also not trained. A post-training quantiser fits a u8 grid to the weight tensor:

```
  scale_w = (max - min) / 255
  zp      = round(-min / scale_w)
```

so `zp` is simply where the value 0.0 falls on the 0..255 code axis. Running the model's stored
parameters backwards recovers the range its original weights occupied:

```
                            code 0            code 255
  model zp/scale      -0.003177198053     0.007921233502
  BT.601 / 255        -0.003188235294     0.007913725490
                          0.35 % off          0.09 % off
```

and textbook calibration on BT.601/255 yields `scale_w = 0.000043537101`, `zp = round(73.2303) = 73`
— the same zero point and a scale within 0.03%. The quantisation parameters are a fossil of the real
network, and they confirm it was doing standard BT.601 even when the exported weight codes are
obfuscated.

## 7. Tests

- `tests/functional/subgraph_tests/fuse_color_conversion.cpp` — static shapes, parameterised over
  RGB/BGR, over `Clamp` vs `FakeQuantize` tails, and over the scale factor via `getYuvScaleFactor()`.
  Uses the coefficients literally and scales weights and bias by the same factor.
- `tests/functional/common/dyn_color_conversion_pattern.cpp` — the dynamic-shape QDQ variant, shared
  by `fuse_dyn_color_conversion.cpp` and `dynamic_resize_conv_subgraph.cpp`.
- `tests/lit/NPU/dialect/IE/passes/fuse_color_conversion.mlir` and
  `fuse_color_conversion_scale_on_shave.mlir` — pattern-matching lit tests.

### Why an accuracy failure here is easy to misread

Because the pass throws the matrix away, wrong coefficients do not fail at compile time. They
appear as a plain accuracy mismatch, and often a partial one: for inputs near zero the R and B rows
clip to 0 on both the reference and the NPU, so only G carries any error and the failure looks like
a small, oddly channel-specific drift rather than a wrong colour matrix.

If a CSC test fails on accuracy, check in this order:

1. Dump the IR and read `outFmt` and `scale` off the `IE.YuvToRgb` op.
2. Confirm `scale` equals the factor implied by the tail range (1/255 for a `[0, 1]` tail).
3. Confirm the Y column of the test's dequantised matrix is constant across the three rows.
4. Confirm each `bias[i] / standardBias[i]` individually equals that same scale — the pass averages
   them, so three inconsistent ratios silently produce a meaningless scale.
