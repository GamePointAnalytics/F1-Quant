from backtesting.rolling_cv import expanding_folds


def test_expanding_folds_boundaries_and_count():
    items = list(range(1, 23))  # 22 "races", like a full season
    folds = expanding_folds(items, min_train=10, test_size=3)
    assert len(folds) == 4  # train sizes 10/13/16/19, test blocks of 3 -> 4 folds fit in 22

    expected_train_sizes = [10, 13, 16, 19]
    for (train, test), expected_size in zip(folds, expected_train_sizes):
        assert len(train) == expected_size
        assert len(test) == 3


def test_expanding_folds_train_only_grows_by_prior_test_block():
    items = list(range(1, 23))
    folds = expanding_folds(items, min_train=10, test_size=3)
    for i in range(1, len(folds)):
        prev_train, prev_test = folds[i - 1]
        train, _ = folds[i]
        assert train == prev_train + prev_test  # expanding window, not sliding


def test_expanding_folds_no_overlap_and_no_lookahead():
    items = list(range(1, 23))
    folds = expanding_folds(items, min_train=10, test_size=3)
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        # every train item must come before every test item in the original order
        assert max(items.index(t) for t in train) < min(items.index(t) for t in test)


def test_expanding_folds_empty_when_too_few_items():
    items = list(range(1, 8))  # only 7 items, min_train=10 can't even start
    assert expanding_folds(items, min_train=10, test_size=3) == []


def test_expanding_folds_works_on_season_year_ints():
    years = [2021, 2022, 2023, 2024, 2025]
    folds = expanding_folds(years, min_train=2, test_size=1)
    assert folds == [
        ([2021, 2022], [2023]),
        ([2021, 2022, 2023], [2024]),
        ([2021, 2022, 2023, 2024], [2025]),
    ]
