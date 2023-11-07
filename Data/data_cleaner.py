import pandas as pd
import numpy as np
import math
import sklearn.preprocessing

def cleaner(train, feature=None, morph=None, pre_morph=False, submission=False):
    """
    Function that performs data cleaning and feature engineering for the data

    inputs:
    - train: path to training data
    - feature: path to feature data, which is not available for the leaderboard data
    - morph: path to morph data, which is not available for the leaderboard data

    outputs:
    - data: cleaned and feature engineered data
    """

    data = pd.read_csv(train)

    ############## CONCAT FEATURE DATA ##############
    if feature:
    #load in additional features for each neuron
        feature_weights = pd.read_csv(feature)
        feature_weights["feature_weights"] = (
        feature_weights.filter(regex="feature_weight_")
        .sort_index(axis=1)
        .apply(lambda x: np.array(x), axis=1)
        )
        # delete the feature_weight_i columns
        feature_weights.drop(
            feature_weights.filter(regex="feature_weight_").columns, axis=1, inplace=True
        )
        data = (
        data.merge(
            feature_weights.rename(columns=lambda x: "pre_" + x), 
            how="left", 
            validate="m:1",
            copy=False,
        )
        .merge(
            feature_weights.rename(columns=lambda x: "post_" + x),
            how="left",
            validate="m:1",
            copy=False,
        ))

    ############## CONCAT MORPH DATA ##############
    if morph:
        morph_embeddings = pd.read_csv(morph)
        # join all morph_embed_i columns into a single np.array column
        morph_embeddings["morph_embeddings"] = (
            morph_embeddings.filter(regex="morph_emb_")
            .sort_index(axis=1)
            .apply(lambda x: np.array(x), axis=1)
        )
        # delete the morph_embed_i columns
        morph_embeddings.drop(
            morph_embeddings.filter(regex="morph_emb_").columns, axis=1, inplace=True
        )
        data = (
        data.merge(
            morph_embeddings.rename(columns=lambda x: "post_" + x),
            how="left",
            validate="m:1",
            copy=False,
        ))
        if pre_morph:
            data = (
            data.merge(
            morph_embeddings.rename(columns=lambda x: "pre_" + x),
            how="left",
            validate="m:1",
            copy=False,
            ))
    
    ############## FE: SIMILARITY ##############
    data["fw_similarity"] = data.apply(row_feature_similarity, axis=1)
        
    ############## FE: PROJECTION GROUP ##############
    # generate projection group as pre->post
    data["projection_group"] = (
        data["pre_brain_area"].astype(str)
        + "->"
        + data["post_brain_area"].astype(str)
    )

    ############## FE: COMBINE COORDINATES ##############
    data = dist_column(data, "axonal_coords", "axonal_coor_")
    data = dist_column(data, "dendritic_coords", "dendritic_coor_")
    data = dist_column(data, "pre_rf_coords", "pre_rf_")
    data = dist_column(data, "post_rf_coords", "post_rf_")
    data = dist_column(data, "pre_nucleus_coords", "pre_nucleus_[xyz]")
    data = dist_column(data, "post_nucleus_coords", "post_nucleus_[xyz]")

    ############## FE: DISTANCE FROM PRE-SYNAPTIC NUCLEUS TO AXON ##############
    data["nuclei_adp_dist"] =  data[["pre_nucleus_coords", "axonal_coords"]].apply(
    lambda x: math.dist(x["pre_nucleus_coords"], x["axonal_coords"]), axis=1)
        
    ############## FE: PER-NEURON ADP COUNTS ##############
    if not submission:
        counts = data.groupby('pre_nucleus_id').count() # count of each presynaptic neuron
        counts = counts["ID"]
        total_connections = data[["pre_nucleus_id", "connected"]].groupby('pre_nucleus_id').sum()
        total_connections = total_connections["connected"]
        adp_counts = pd.DataFrame([counts, total_connections]).transpose()
        adp_counts = adp_counts.rename(columns={"ID":"ADP_total", "connected":"connect_total"})
        adp_counts["connect_rate"] = adp_counts["connect_total"]/adp_counts["ADP_total"]
        data = data.merge(adp_counts, left_on='pre_nucleus_id', right_on='pre_nucleus_id')

    ############## STANDARDIZE ALL NUMERIC DATA #############
    num_cols = data.select_dtypes(include='number').drop(columns=['ID', 'pre_nucleus_id', 'post_nucleus_id'])
    num_cols = num_cols.columns
    for column in num_cols:
        data[column] = sklearn.preprocessing.StandardScaler().fit_transform(np.array(data[column]).reshape(-1, 1))
    return data


#cosine similarity function
def row_feature_similarity(row):
    pre = row["pre_feature_weights"]
    post = row["post_feature_weights"]
    return (pre * post).sum() / (np.linalg.norm(pre) * np.linalg.norm(post))


# join all distance columns into a single np.array column
def dist_column(df, new_col, old_cols):
    df[new_col] = (
        df.filter(regex=old_cols)
        .sort_index(axis=1)
        .apply(lambda x: np.array(x), axis=1)
    )
    # delete the old columns
    df.drop(
        df.filter(regex=old_cols).columns, axis=1, inplace=True
    )
    return df




