import cloudpickle
import flwr as fl
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import (
    Parameters,
    FitRes,
    EvaluateRes,
    Scalar,
    parameters_to_ndarrays,
)
from baseline_model.model import get_model, set_model_parameters


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(evaluate_res.num_examples for _, evaluate_res in results)
        if total_examples == 0:
            print(
                f"Round {server_round}: All clients returned 0 examples in evaluate — skipping aggregation."
            )
            return None, {}

        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            params_numpy = parameters_to_ndarrays(aggregated_parameters)

            full_pipeline = get_model()
            set_model_parameters(full_pipeline, params_numpy)

            # Save a per-round checkpoint so no round is ever lost
            checkpoint_path = f"model_round_{server_round}.pkl"
            with open(checkpoint_path, "wb") as f:
                cloudpickle.dump(full_pipeline, f)

            # Always overwrite final_model.pkl so it points to the latest round
            with open("final_model.pkl", "wb") as f:
                cloudpickle.dump(full_pipeline, f)

            print(
                f"Round {server_round}: Saved checkpoint → {checkpoint_path}"
                f"  (final_model.pkl updated)"
            )

        return aggregated_parameters, aggregated_metrics
