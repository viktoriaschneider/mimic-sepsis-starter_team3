import flwr as fl
import cloudpickle
import base64
from typing import List, Tuple, Dict
from flwr.common import Metrics
from baseline_model.custom_strategy import SaveModelStrategy
from baseline_model.model import get_model, get_model_parameters

# from baseline_model.model import get_model, get_model_parameters


def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
    """
    Aggregates the rich metrics sent by the 5 hospitals.
    Counts (FP, FN, Cost) are summed. Ratios (Accuracy, AUROC) are weighted averaged.
    """
    if not metrics:
        return {}

    total_examples = sum([num_examples for num_examples, _ in metrics])

    # 1. Sum up the absolute counts across all hospitals
    total_cost = sum([m.get("cost_score", 0) for _, m in metrics])
    total_fp = sum([m.get("false_positives", 0) for _, m in metrics])
    total_fn = sum([m.get("false_negatives", 0) for _, m in metrics])
    total_tp = sum([m.get("true_positives", 0) for _, m in metrics])
    total_tn = sum([m.get("true_negatives", 0) for _, m in metrics])

    # 2. Calculate the weighted average for rates/ratios
    weighted_auroc = (
        sum([num * m.get("auroc", 0) for num, m in metrics]) / total_examples
    )
    weighted_log_loss = (
        sum([num * m.get("log_loss", 0) for num, m in metrics]) / total_examples
    )
    weighted_accuracy = (
        sum([num * m.get("accuracy", 0) for num, m in metrics]) / total_examples
    )
    weighted_f1 = (
        sum([num * m.get("f1_score", 0) for num, m in metrics]) / total_examples
    )
    weighted_precision = (
        sum([num * m.get("precision", 0) for num, m in metrics]) / total_examples
    )
    weighted_recall = (
        sum([num * m.get("recall", 0) for num, m in metrics]) / total_examples
    )

    return {
        "TOTAL_COST": total_cost,
        "FP": total_fp,
        "FN": total_fn,
        "TP": total_tp,
        "TN": total_tn,
        "auroc": weighted_auroc,
        "log_loss": weighted_log_loss,
        "accuracy": weighted_accuracy,
        "f1_score": weighted_f1,
        "precision": weighted_precision,
        "recall": weighted_recall,
    }


def get_on_fit_config_fn():
    model_obj = get_model()
    model_bytes = cloudpickle.dumps(model_obj)
    model_b64 = base64.b64encode(model_bytes).decode("utf-8")

    def fit_config(_server_round: int):
        return {"model_bytes": model_b64}

    return fit_config


def main():
    print("Starting Central FL Server with Dynamic Pipelines...")

    init_model = get_model()
    initial_params = fl.common.ndarrays_to_parameters(get_model_parameters(init_model))

    strategy = SaveModelStrategy(
        min_fit_clients=5,
        min_available_clients=5,
        min_evaluate_clients=5,
        on_fit_config_fn=get_on_fit_config_fn(),
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
