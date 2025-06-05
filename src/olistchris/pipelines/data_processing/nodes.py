import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
    

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Summer'   # Brazil (Southern Hemisphere)
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:  # [9, 10, 11]
        return 'Spring'
    

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth’s radius in kilometers
    # convert degrees → radians
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    # Haversine computation
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def preprocess_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for orders.

    Args:
        orders: Raw data.
    Returns:
        
    """
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    orders['year'] = orders['order_purchase_timestamp'].dt.year
    orders['month'] = orders['order_purchase_timestamp'].dt.month
    orders['weekday'] = orders['order_purchase_timestamp'].dt.day_name()
    orders['hour'] = orders['order_purchase_timestamp'].dt.hour
    orders['part_of_day'] = orders['hour'].apply(get_part_of_day)
    orders = orders[orders['order_status'] == 'delivered']
    orders = orders.dropna()
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    return orders


def create_model_input_table(
    customers: pd.DataFrame, orders: pd.DataFrame, order_items: pd.DataFrame,
    payments: pd.DataFrame, reviews: pd.DataFrame, products: pd.DataFrame,
    sellers: pd.DataFrame, geolocation: pd.DataFrame, product_translation: pd.DataFrame
    ) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    # merging datasets
    df = pd.merge(orders, order_items, on='order_id')
    df = pd.merge(df, payments, on='order_id')
    df = pd.merge(df, reviews, on='order_id')
    df = pd.merge(df, products, on='product_id')
    df = pd.merge(df, customers, on='customer_id')
    order_counts = df.groupby('customer_unique_id')['order_id'].nunique().reset_index(name='order_count')

    df = df.drop(['order_id', 'order_status', 'order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date',
                'order_item_id', 'seller_id', 'review_id', 'review_comment_title', 'payment_sequential', 'product_id',
                'product_name_lenght', 'product_description_lenght', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp'], axis=1)

    geo_clean = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean()

    df = pd.merge(df, geo_clean, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')

    df = df.drop(['customer_zip_code_prefix', 'customer_id'], axis=1)

    # ENGINEERING FEATURES:
    df['season'] = df['month'].apply(month_to_season)
    # Calculate last purchase date per customer
    last_purchase = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index(name='last_purchase_date')

    # Get the latest date in the entire dataset
    max_date = df['order_purchase_timestamp'].max()

    # Calculate months inactive
    last_purchase['months_inactive'] = ((max_date - last_purchase['last_purchase_date']) / pd.Timedelta(days=30)).round().astype(int)

    # Time it took to deliver item
    df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    distinct_products_per_customer = df.groupby('customer_unique_id')['product_category_name'].nunique().reset_index(name='distinct_product_categories')

    df = df.drop(['order_purchase_timestamp', 'order_delivered_customer_date', 'month', 'year', 'part_of_day'], axis=1)

    # AGGREGATION:
    # get most common value (mode) in specific columns 
    # If multiple equally common values, get first. 
    # If empty, return None
    agg_df = df.groupby('customer_unique_id').agg({
    'weekday': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'hour': 'median',
    'price': ['sum', 'mean'],
    'freight_value': 'mean',
    'payment_type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'payment_installments': 'mean',
    'payment_value': ['sum', 'mean'],
    'review_score': 'mean',
    'product_category_name': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'product_photos_qty': 'mean',
    'product_weight_g': 'mean',
    'product_length_cm': 'mean',
    'product_height_cm': 'mean',
    'product_width_cm': 'mean',
    'customer_city': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'customer_state': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean',
    'season': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'delivery_time': 'mean'
    }).reset_index()

    # flatten multi index columns
    agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_df.columns]

    # translate product to english
    agg_df = pd.merge(agg_df, product_translation, left_on='product_category_name_<lambda>', right_on='product_category_name', how='left')
    agg_df = agg_df.drop(['product_category_name', 'product_category_name_<lambda>'], axis=1) # Drop the redundant Portuguese column after merging

    # ADDING ENGINEERED FEATURES:
    agg_df = pd.merge(agg_df, order_counts, on='customer_unique_id', how='left')

    # define repeat buyers
    agg_df['repeat_buyer'] = agg_df['order_count'].apply(lambda x: 1 if x > 1 else 0)
    agg_df = pd.merge(agg_df, distinct_products_per_customer, on='customer_unique_id')
    agg_df = pd.merge(agg_df, last_purchase[['customer_unique_id', 'months_inactive']], on='customer_unique_id', how='left')

    # FEATURE ENGINEERING 2:
    # Reduce geolocation data to one latitude/longitude per zip code prefix
    # (take the average of all points sharing the same prefix)
    geo_prefix = geolocation.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',   # average latitude
        'geolocation_lng': 'mean'    # average longitude
    }).reset_index()

    # Merge those averaged coords into the customers table
    customers_loc = (
        customers
        .merge(
            geo_prefix,
            left_on='customer_zip_code_prefix',    # match customer prefix
            right_on='geolocation_zip_code_prefix',#to averaged geolocation
            how='left'                             # keep all customers
        )
        .rename(columns={
            'geolocation_lat': 'customer_lat',
            'geolocation_lng': 'customer_lng'
        })
        .drop('geolocation_zip_code_prefix', axis=1)  # drop redundant column
    )

    # Merge averaged coords into the sellers table
    sellers_loc = (
        sellers
        .merge(
            geo_prefix,
            left_on='seller_zip_code_prefix',
            right_on='geolocation_zip_code_prefix',
            how='left'
        )
        .rename(columns={
            'geolocation_lat': 'seller_lat',
            'geolocation_lng': 'seller_lng'
        })
        .drop('geolocation_zip_code_prefix', axis=1)
    )

    # Build one DataFrame linking orders → customers → sellers
    order_merged = (
        orders[['order_id', 'customer_id']] # start with orders and customer IDs
        .merge(
            customers[['customer_id', 'customer_unique_id']], # Attach customer_unique_id from the customers table
            on='customer_id'
        )
        .merge(
            order_items[['order_id', 'seller_id']],     # attach seller IDs via order_items
            on='order_id'
        )
        .merge(
            customers_loc[['customer_id', 'customer_lat', 'customer_lng']],
            on='customer_id',                           # attach customer coords
            how='left'
        )
        .merge(
            sellers_loc[['seller_id', 'seller_lat', 'seller_lng']],
            on='seller_id',                             # attach seller coords
            how='left'
        )
    )

    # Apply the Haversine function row-wise to compute distances
    order_merged['distance_km'] = order_merged.apply(
        lambda row: haversine(
            row['customer_lat'], row['customer_lng'],
            row['seller_lat'],   row['seller_lng']
        )
        if pd.notnull(row['customer_lat']) and pd.notnull(row['seller_lat'])
        else None,  # skip if any coordinate is missing
        axis=1
    )

    # Aggregate average distance per customer
    avg_distance_per_customer = (
        order_merged
        .groupby('customer_unique_id')['distance_km']
        .mean()
        .reset_index()
        .rename(columns={'distance_km': 'avg_distance_km'})
    )

    # Merge back into agg_df
    agg_df = pd.merge(agg_df, avg_distance_per_customer, on='customer_unique_id', how='left')
    
    # get volume mean
    agg_df['volume_mean'] = agg_df['product_length_cm_mean'] * agg_df['product_width_cm_mean'] * agg_df['product_height_cm_mean']

    # replace 0g weights to 0.1g weights for calculations
    agg_df['product_weight_g_mean'] = agg_df['product_weight_g_mean'].replace(0, 0.1)

    # get different ratios
    agg_df['cost_volume'] = agg_df['price_mean'] / agg_df['volume_mean']
    agg_df['density_mean'] = agg_df['product_weight_g_mean'] / agg_df['volume_mean']
    agg_df['cost_weight'] = agg_df['price_mean'] / agg_df['product_weight_g_mean']
    agg_df['lh_ratio'] = agg_df['product_length_cm_mean'] / agg_df['product_height_cm_mean']
    agg_df['lw_ratio'] = agg_df['product_length_cm_mean'] / agg_df['product_width_cm_mean']
    agg_df['hw_ratio'] = agg_df['product_height_cm_mean'] / agg_df['product_width_cm_mean']

    agg_df.dropna(inplace=True)
    return agg_df
