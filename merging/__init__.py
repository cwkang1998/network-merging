from .logits_stats import test, naive, std, ratio, overall_ratio, thirdQ


def logits_statistics(args, model1, model2, device, test_loader):
    model_evals = [naive, std, ratio, overall_ratio, thirdQ]
    result = []
    for e in model_evals:
        print(f"Applied function: {e.__name__}")
        test_loss, acc = test(args, model1, model2, e, device, test_loader)
        result.append({"func": e.__name__, "test_loss": test_loss, "acc": acc})
    return result
