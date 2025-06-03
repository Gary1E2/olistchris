from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles, preprocess_orders


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_orders,
                inputs="orders",
                outputs="preprocessed_orders",
                name="preprocess_orders_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["customers", "preprocessed_orders", "order_items", "payments", "reviews",
                        "products", "sellers", "geolocation", "product_translation"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
