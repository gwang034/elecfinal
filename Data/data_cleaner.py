import pandas as pd
import numpy as np
import math
import sklearn.preprocessing

def cleaner(train, feature=None, imp_morph=None, pre_morph=False, submission=False):
    """
    Function that performs data cleaning and feature engineering for the data

    inputs:
    - train: path to training data
    - feature: path to feature data
    - imp_morph: path to imputed morph embedding data

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
    
    ############## CONCAT IMPUTED MORPH VECTOR DATA ################
    morph_embs = pd.read_csv(imp_morph)
    morph_embs["pre_morph_embeddings"] = (morph_embs.filter(regex="pre_morph_emb_").sort_index(axis=1)
                                          .apply(lambda x: np.array(x), axis=1))
    
    morph_embs["post_morph_embeddings"] = (morph_embs.filter(regex="post_morph_emb_")
                                           .sort_index(axis=1).apply(lambda x: np.array(x), axis=1))
    
    morph_embs.drop(morph_embs.filter(regex="_morph_emb_").columns, axis=1, inplace=True)
    morph_embs["ID"] = data["ID"]
    data = data.merge(morph_embs, on="ID")

    ############## FE: CALCULATE DISTANCES BETWEEN IMPUTED MORPH EMBEDDINGS ###################
    data["me_similarity"] = data.apply(row_morph_similarity, axis=1)

    

    ############## OLD CODE: CONCAT MORPH DATA ##############
    # if morph:
    #     morph_embeddings = pd.read_csv(morph)
    #     # join all morph_embed_i columns into a single np.array column
    #     morph_embeddings["morph_embeddings"] = (
    #         morph_embeddings.filter(regex="morph_emb_")
    #         .sort_index(axis=1)
    #         .apply(lambda x: np.array(x), axis=1)
    #     )
    #     # delete the morph_embed_i columns
    #     morph_embeddings.drop(
    #         morph_embeddings.filter(regex="morph_emb_").columns, axis=1, inplace=True
    #     )
    #     data = (
    #     data.merge(
    #         morph_embeddings.rename(columns=lambda x: "post_" + x),
    #         how="left",
    #         validate="m:1",
    #         copy=False,
    #     ))
    #     if pre_morph:
    #         data = (
    #         data.merge(
    #         morph_embeddings.rename(columns=lambda x: "pre_" + x),
    #         how="left",
    #         validate="m:1",
    #         copy=False,
    #         ))
    
    ############## FE: SIMILARITY ##############
    data["fw_similarity"] = data.apply(row_feature_similarity, axis=1)
        
    ############## FE: PROJECTION GROUP ##############
    # generate projection group as pre->post
    # data["projection_group"] = (
        # data["pre_brain_area"].astype(str)
    #     + "->"
    #     + data["post_brain_area"].astype(str)
    # )

    ############## FE: COMBINE COORDINATES ##############
    data = coord_column(data, "axonal_coords", "axonal_coor_")
    data = coord_column(data, "dendritic_coords", "dendritic_coor_")
    data = coord_column(data, "pre_rf_coords", "pre_rf_[xy]")
    data = coord_column(data, "post_rf_coords", "post_rf_[xy]")
    data = coord_column(data, "pre_nucleus_coords", "pre_nucleus_[xyz]")
    data = coord_column(data, "post_nucleus_coords", "post_nucleus_[xyz]")

    data = coord_column(data, "pre_nucleus_xy", "pre_nucleus_[xy]")
    data = coord_column(data, "post_nucleus_xy", "post_nucleus_[xy]")

    ############## FE: RF SIMILARITY ##############
    data = coord_df(data)
    data["rf_similarity"] = data.apply(rfsimilarity, axis=1)

    ############## FE: BRAIN AREA ##############
    data = one_hot('pre_brain_area', data, '_pre')
    data = one_hot('post_brain_area', data, '_post')

    ############## FE: BRAIN COMPARTMENT GROPUING ##############
    data = area_cols(data)

    ############## FE: MINICOLUMNS? ##############
    data["minicol_dist"] =  data[["pre_nucleus_xy", "post_nucleus_xy"]].apply(
    lambda x: math.dist(x["pre_nucleus_xy"], x["post_nucleus_xy"]), axis=1)

    ############## FE: DISTANCE FROM PRE-SYNAPTIC NUCLEUS TO AXON ##############
    data["nuclei_adp_dist"] =  data[["pre_nucleus_coords", "axonal_coords"]].apply(
    lambda x: math.dist(x["pre_nucleus_coords"], x["axonal_coords"]), axis=1)
        
    ############## FE: PER-NEURON ADP COUNTS ##############
    # if not submission:
    # counts = data.groupby('pre_nucleus_id').count() # count of each presynaptic neuron
    # counts = pd.DataFrame(counts["ID"]).rename(columns={"ID":"ADP_total"})
        
        # total_connections = data[["pre_nucleus_id", "connected"]].groupby('pre_nucleus_id').sum()
        # total_connections = total_connections["connected"]
        # adp_counts = pd.DataFrame([counts, total_connections]).transpose()
        # adp_counts = adp_counts.rename(columns={"ID":"ADP_total", "connected":"connect_total"})
        # adp_counts["connect_rate"] = adp_counts["connect_total"]/adp_counts["ADP_total"]
        # print(data)
    # data = data.merge(counts, how='left', left_on='pre_nucleus_id', right_on='pre_nucleus_id')
        # print(data)

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

#cosine similarity function
def row_morph_similarity(row):
    pre = row["pre_morph_embeddings"]
    post = row["post_morph_embeddings"]
    return (pre * post).sum() / (np.linalg.norm(pre) * np.linalg.norm(post))


# join all distance columns with coordinates into a single np.array column
def coord_column(df, new_col, old_cols):
    df[new_col] = (
        df.filter(regex=old_cols)
        .sort_index(axis=1)
        .apply(lambda x: np.array(x), axis=1)
    )
    # delete the old columns
    # df.drop(
    #     df.filter(regex=old_cols).columns, axis=1, inplace=True
    # )
    return df

# only the x and y coordinates
def coord_df(df):
    df = coord_column(df, "pre_rf_coords_xy", "pre_rf_[xy]")
    df = coord_column(df, "post_rf_coords_xy", "post_rf_[xy]")
    return df

def rfsimilarity(row):
    pre = row["pre_rf_coords_xy"]
    post = row["post_rf_coords_xy"]
    return (pre * post).sum() / (np.linalg.norm(pre) * np.linalg.norm(post))

def one_hot(column, df, suffix=''):
    """
    one-hot encodes this shit
    """
    cats = pd.unique(df[column])

    for cat in cats:
        new_col = cat+suffix
        df[new_col] = df[column]==cat
        df[new_col] = df[new_col].astype('int')
    
    df = df.drop(columns=column)
    return df



def area_cols(df):
    # Encode brain areas
    area1 = ["basal", "soma"]
    area2 = ["axon", "apical", "oblique", "apical_shaft"]
    area3 = ["apical_tuft"]
    df["area1"] = df["compartment"].isin(area1).astype('int')
    df["area2"] = df["compartment"].isin(area2).astype('int')
    df["area3"] = df["compartment"].isin(area3).astype('int')
    return df

