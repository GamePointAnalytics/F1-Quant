def expanding_folds(items: list, min_train: int, test_size: int) -> list[tuple[list, list]]:
    """
    Walk-forward / expanding-window CV splits: train grows every fold, test is
    always the next `test_size` items — never overlapping with, and never
    appearing before, its own training set. Generic over any ordered list
    (race-laps DataFrames, season year-ints, ...) so every backtest in this
    repo can share one fold-generation implementation instead of each having
    its own single, fixed train/test split.

    A single fixed split only tells you "did this work once" — walk-forward
    folds give a mean *and a spread* across several out-of-sample windows,
    which is what's actually needed to trust a number for live decisions
    rather than a one-off result.
    """
    folds = []
    n = len(items)
    start = min_train
    while start + test_size <= n:
        train = items[:start]
        test = items[start:start + test_size]
        folds.append((train, test))
        start += test_size
    return folds
