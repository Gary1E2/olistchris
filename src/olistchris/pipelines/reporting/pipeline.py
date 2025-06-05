from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_confusion_matrix,
    create_roc_auc,
    create_cont_eval
)


def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            node(
                func=create_confusion_matrix,
                inputs=["repeat_buyer_classifier", "X_test_repeat", "y_test_repeat",
                        "X_train_repeat", "y_train_repeat", "params:confusion_matrix"],
                outputs="repeat_buyer_cm",
                name="repeat_confusion_matrix_node"
            ),
            node(
                func=create_roc_auc,
                inputs=["repeat_buyer_classifier", "X_test_repeat", "y_test_repeat",
                        "X_train_repeat", "y_train_repeat", "params:roc_auc"],
                outputs="repeat_buyer_rocauc",
                name="repeat_roc_auc_node"
            ),
            node(
                func=create_cont_eval,
                inputs=["freight_value_regressor", "X_test_freight", "y_test_freight",
                        "X_train_freight", "y_train_freight", "params:cont_eval"],
                outputs="freight_value_conteval",
                name="freight_cont_eval_node"
            ),
            node(
                func=create_cont_eval,
                inputs=["delivery_time_regressor", "X_test_delivery", "y_test_delivery",
                        "X_train_delivery", "y_train_delivery", "params:cont_eval"],
                outputs="delivery_time_conteval",
                name="delivery_cont_eval_node"
            ),
        ]
    )
