from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_classifier, evaluate_regressor, split_data, train_repeat_buyer, train_freight_value, train_delivery_time


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_repeat_buyer"],
                outputs=["X_train_repeat", "X_test_repeat", "y_train_repeat", "y_test_repeat"],
                name="split_repeat_buyer_data_node",
            ),
            node(
                func=train_repeat_buyer,
                inputs=["X_train_repeat", "y_train_repeat"],
                outputs="repeat_buyer_classifier",
                name="train_repeat_buyer_node",
            ),
            node(
                func=evaluate_classifier,
                inputs=["repeat_buyer_classifier", "X_test_repeat", "y_test_repeat", "X_train_repeat", "y_train_repeat"],
                outputs="repeat_buyer_test_metrics",
                name="evaluate_repeat_buyer_model_node",
            ),
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_freight_value"],
                outputs=["X_train_freight", "X_test_freight", "y_train_freight", "y_test_freight"],
                name="split_freight_value_data_node",
            ),
            node(
                func=train_freight_value,
                inputs=["X_train_freight", "y_train_freight"],
                outputs="freight_value_regressor",
                name="train_freight_value_node",
            ),
            node(
                func=evaluate_regressor,
                inputs=["freight_value_regressor", "X_test_freight", "y_test_freight", "X_train_freight", "y_train_freight"],
                outputs="freight_value_test_metrics",
                name="evaluate_freight_value_model_node",
            ),
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_delivery_time"],
                outputs=["X_train_delivery", "X_test_delivery", "y_train_delivery", "y_test_delivery"],
                name="split_delivery_time_data_node",
            ),
            node(
                func=train_delivery_time,
                inputs=["X_train_delivery", "y_train_delivery"],
                outputs="delivery_time_regressor",
                name="train_delivery_time_node",
            ),
            node(
                func=evaluate_regressor,
                inputs=["delivery_time_regressor", "X_test_delivery", "y_test_delivery", "X_train_delivery", "y_train_delivery"],
                outputs="delivery_time_test_metrics",
                name="evaluate_delivery_time_model_node",
            ),
        ]
    )
