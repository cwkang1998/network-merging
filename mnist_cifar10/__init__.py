from .logits_stats import test, naive, std, ratio, overall_ratio, thirdQ
from .multi_pass_aug import mean_agg_test, voting_agg_test
from .smart_coord import smart_coord_test
from augments import (
    apply_gaussian,
    apply_poisson,
    apply_hflip,
    apply_vflip,
    apply_random_crop,
    apply_sharpen
)

AUGS_DESCP = [
    "1 sharpen",
    "diff alpha sharpen",
    "1 gauss",
    "5 gauss",
    "diff std gauss",
    "1 poisson",
    "5 poisson",
    "diff rate poisson",
    "h and v flip",
    "random crop",
]

AUGS = [
    [{"iter": 1, "func": apply_sharpen, "kwargs": {}}],
    [
        {"iter": 1, "func": apply_sharpen, "kwargs": {"alpha": 0.1}},
        {"iter": 1, "func": apply_sharpen, "kwargs": {"alpha": 0.3}},
        {"iter": 1, "func": apply_sharpen, "kwargs": {"alpha": 0.5}},
        {"iter": 1, "func": apply_sharpen, "kwargs": {"alpha": 0.7}},
        {"iter": 1, "func": apply_sharpen, "kwargs": {"alpha": 0.9}},
    ],
    [{"iter": 1, "func": apply_gaussian, "kwargs": {}}],
    [{"iter": 5, "func": apply_gaussian, "kwargs": {}}],
    [
        {"iter": 1, "func": apply_gaussian, "kwargs": {"std": 0.05}},
        {"iter": 1, "func": apply_gaussian, "kwargs": {"std": 0.1}},
        {"iter": 1, "func": apply_gaussian, "kwargs": {"std": 0.3}},
        {"iter": 1, "func": apply_gaussian, "kwargs": {"std": 0.5}},
        {"iter": 1, "func": apply_gaussian, "kwargs": {"std": 0.7}},
        {"iter": 1, "func": apply_gaussian, "kwargs": {"std": 1}},
    ],
    [{"iter": 1, "func": apply_poisson, "kwargs": {}}],
    [{"iter": 5, "func": apply_poisson, "kwargs": {}}],
    [
        {"iter": 1, "func": apply_poisson, "kwargs": {"rate": 0.05}},
        {"iter": 1, "func": apply_poisson, "kwargs": {"rate": 0.1}},
        {"iter": 1, "func": apply_poisson, "kwargs": {"rate": 0.3}},
        {"iter": 1, "func": apply_poisson, "kwargs": {"rate": 0.5}},
        {"iter": 1, "func": apply_poisson, "kwargs": {"rate": 0.7}},
        {"iter": 1, "func": apply_poisson, "kwargs": {"rate": 1}},
    ],
    [
        {"iter": 1, "func": apply_hflip, "kwargs": {}},
        {"iter": 1, "func": apply_vflip, "kwargs": {}},
    ],
    [{"iter": 1, "func": apply_random_crop, "kwargs": {}}],
]


def logits_statistics(args, model1, model2, device, test_loaders):
    model_evals = [naive, std, ratio, overall_ratio, thirdQ]
    result = []
    for e in model_evals:
        print(f"Applied function: {e.__name__}")
        test_loss, acc = test(args, model1, model2, e, device, test_loaders)
        result.append({"func": e.__name__, "test_loss": test_loss, "acc": acc})
    return result


def multi_pass_aug_mean(args, model1, model2, device, test_loaders):
    result = []
    for i in range(len(AUGS)):
        a = AUGS[i]
        print(f"Applied methods: {AUGS_DESCP[i]}, Agg Method: mean")
        test_loss, acc = mean_agg_test(args, model1, model2, a, device, test_loaders)
        result.append({"methods": AUGS_DESCP[i], "test_loss": test_loss, "acc": acc})

    return result


def multi_pass_aug_voting(args, model1, model2, device, test_loaders):
    result = []
    for i in range(len(AUGS)):
        a = AUGS[i]
        print(f"Applied methods: {AUGS_DESCP[i]}, Agg Method: voting")
        test_loss, acc = voting_agg_test(args, model1, model2, a, device, test_loaders)
        result.append({"methods": AUGS_DESCP[i], "test_loss": test_loss, "acc": acc})

    return result


def smart_coordinator(args, model1, model2, pan1, pan2, device, test_loaders):
    result = []
    print(f"PAN type: {args.pan_type}")
    test_loss, acc = smart_coord_test(
        args, model1, model2, pan1, pan2, device, test_loaders
    )
    result.append({"test_loss": test_loss, "acc": acc})
    return result
