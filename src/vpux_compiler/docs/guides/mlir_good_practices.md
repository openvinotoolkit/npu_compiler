# MLIR: Good Practices

MLIR is a framework for building compilers. This document tries to capture the
practices that make the MLIR-based code more understandable and less
error-prone. In particular, these practices are gathered during VPUX compiler
development, but they could as well be universal.

Note, however, that the way one writes code is something that may change over
time. If you see that something contradicts with your understanding or your own
practices, feel free to raise questions or (better) amend this guide so that
everyone could benefit from the better guidelines.

For the introduction to MLIR, see the [MLIR primer](./primer_mlir.md).

---

## Passes

### Running multiple rewriters within one pass

**General rule**: the IR traversal direction of a pass and rewriters within this
pass must match. This becomes essential when there are multiple "concurrent"
rewriters that modify overlapping graph patterns: aligned IR traversal ensures
the rewriters apply *in order* and according to their *priorities* (if any are
specified). If this rule is not followed, observers may experience the behavior
that differs from their understanding of the code. This stems from the fact that
the IR traversal direction of the pass may favor the IR traversal of one
rewriter's `match` over another rewriter's `match`, *even when the priorities of
rewriters would suggest otherwise*.

---

Usually, when you are to add a new rewriter to an existing pass, you'd have to
think about the following: how should the matching logic go? Consider a simple
pattern that has to be matched, written in pseudo-valid `.mlir` format:
```mlir
func.func @GoodPractices1(%external : i64) {
    // pattern: A -> B -> C
    %a = A(%external) -> !typeA
    %b = B(%a) -> !typeB
    %c = C(%b) -> !typeC

    return %c
}
```

There are typically three strategies to go with:
* "Top-down" matching: subclass `mlir::OpRewritePattern<A>`; matching semantics
  is "start at A. A's output consumer op is B. B's output consumer op is C"
* "Bottom-up" matching: subclass `mlir::OpRewritePattern<C>`; matching semantics
  is "start at C. C's operand defining op is B. B's operand defining op is A"
* "Mixed" matching: subclass `mlir::OpRewritePattern<B>`; matching semantics is
  "start at B. B's operand defining op is A. B's output consumer op is C"

All three strategies are generally fine but special caution has to be taken when
**multiple** rewriters are present within the same pass. Especially, when some
of such rewriters match "overlapping" patterns (e.g. two rewriters that subclass
`mlir::OpRewritePattern<B>`, one works solely on `B`, another looks for `A -> B`
pattern).

> In the cases of multiple rewriters, the rule of thumb is to use the same
> matching strategy for all rewriters, avoiding the "mixed" matching strategy
> (thus, either do "top-down" or "bottom-up" for all).

Moreover, you may wish to consider the following *additional* parameters when
dealing with multiple overlapping rewriters:
* The IR traversal direction used by e.g. `mlir::applyPatternsGreedily()`
  (controlled using `GreedyRewriteConfig::useTopDownTraversal` flag)
* [mlir::PatternBenefit](https://mlir.llvm.org/doxygen/classmlir_1_1PatternBenefit.html)
  that controls the priority of rewriters

Showing how these parameters affect the results is easier using an example.
Consider the following (again, in pseudo-valid `.mlir` format):
```mlir
func.func @GoodPractices2(%external : i64) {
    // pattern: A { is_special } -> A
    %a1 = A(%external) { is_special } -> !typeA
    %a2 = A(%a1) -> !typeA

    return %a2
}
```

Now, let's assume we have 2 rewriters and 2 passes (that differ in the
parameters). The rewriters would replace `A -> A` pair with another operation,
but would do it slightly differently:
```cpp
// Replaces `A { is_special } -> A` pair with operation X
struct ReplaceAAWithX : mlir::OpRewritePattern<A> {
    ReplaceAAWithX(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit)
        : mlir::OpRewritePattern<A>(ctx, benefit) {} // pass benefit to base class

    // use "bottom-up" matching strategy
    mlir::LogicalResult matchAndRewrite(AOp aOp, mlir::PatternRewriter& rewriter) const final {
        // when given A {is_specia} -> A, match 'A' initially
        const bool isSpecial = aOp.isSpecial();
        if (isSpecial) { // skip when special
            return mlir::failure();
        }
        // reject parent op unless it is 'A {is_special}'
        auto parentAOp = aOp.getOperand(0).getDefiningOp<AOp>();
        if (!parentAOp || !parentOp.isSpecial()) {
            return mlir::failure();
        }

        /* replace with X... */

        return mlir::success();
    }
};

// Replaces `A { is_special } -> A` pair with operation Y
struct ReplaceAAWithY : mlir::OpRewritePattern<B> {
    ReplaceAAWithY(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit)
        : mlir::OpRewritePattern<A>(ctx, benefit) {} // pass benefit to base class

    // use "top-down" matching strategy
    mlir::LogicalResult matchAndRewrite(AOp aOp, mlir::PatternRewriter& rewriter) const final {
        // when given A {is_specia} -> A, match 'A {is_special}' initially
        const bool isSpecial = aOp.isSpecial();
        if (!isSpecial) { // skip when not special
            return mlir::failure();
        }
        // reject consumer op unless it is 'A' (without is_special attribute)
        auto consumerOp = *aOp.getResult(0).getUsers().begin();
        if (!consumerOp || consumerOp.isSpecial()) {
            return mlir::failure();
        }

        /* replace with Y... */

        return mlir::success();
    }
};

const mlir::PatternBenefit benefitLow(1);
const mlir::PatternBenefit benefitHigh(2);

// use top-down IR traversal
// ReplaceAAWithX has _higher_ benefit than ReplaceAAWithY
void Pass1::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReplaceAAWithX>(&ctx, benefitHigh); // !!!
    patterns.add<ReplaceAAWithY>(&ctx, benefitLow);

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true; // !!!

    const auto res = mlir::applyPatternsGreedily(
        func, std::move(patterns), config);
    if (mlir::failed(res)) {
        signalPassFailure();
    }
}

// use bottom-up IR traversal
// ReplaceAAWithX has _lower_ benefit than ReplaceAAWithY
void Pass2::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReplaceAAWithX>(&ctx, benefitLow); // !!!
    patterns.add<ReplaceAAWithY>(&ctx, benefitHigh);

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = false; // !!!

    const auto res = mlir::applyPatternsGreedily(
        func, std::move(patterns), config);
    if (mlir::failed(res)) {
        signalPassFailure();
    }
}
```

So, what happens when we run these passes?

`vpux-opt --pass1` would give:
```mlir
func.func @GoodPractices2(%external : i64) {
    %y = Y(%external) -> !typeY
    return %y
}
```

`vpux-opt --pass2` would give:
```mlir
func.func @GoodPractices2(%external : i64) {
    %x = X(%external) -> !typeX
    return %x
}
```

**But Why?** The IR traversal is important: when we go top-down in `Pass1`,
`ReplaceAAWithX` would fail to match the first 'A {is_special}'. But
`ReplaceAAWithY` would succeed and thus would rewrite the code to `Y`, *despite
higher benefit* of `ReplaceAAWithX`.

Same happens in the other case: when we go bottom-up in `Pass2`,
`ReplaceAAWithY` would fail to match the second 'A' but `ReplaceAAWithX` would
succeed and thus would rewrite the code to `X`.

> Thus, the rule of thumb is: if your overlapping rewriters all match the IR in
> a "top-down" fashion, your pass config's direction must also be "top-down". If
> rewriters do "bottom-up" matching, the pass config's should specify
> "bottom-up" IR traversal as well. Only in such configurations the pattern
> benefits would behave correctly.

**Note**: In practice, the overlapping cases would be much harder to spot: the
rewriters could have different operations involved and have only partial
overlaps (e.g. `A -> B -> C -> B` with one rewriter starting at `B` and finding
`C -> B` and another starting at `B` and finding `A -> B -> C`).
