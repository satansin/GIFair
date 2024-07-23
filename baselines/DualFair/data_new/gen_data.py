import os
import json
import pandas as pd

import pdb

def get_cols_json(cols, categorical_features):
    cols_json = {"columns": []}
    for col in cols:
        cols_json["columns"].append({
            "name": col,
            "type": "categorical" if col in categorical_features else "continuous"
        })
    return cols_json

def preprocessing_adult(df):
    def group_edu(x):
        if x <= 5:
            return "<6"
        elif x >= 13:
            return ">12"
        else:
            return x

    def age_cut(x):
        if x >= 70:
            return ">=70"
        else:
            return x

    def group_race(x):
        if x == "White":
            return 1.0
        else:
            return 0.0

    # Cluster education and age attributes.
    # Limit education range
    df["education-num"] = df["education-num"].apply(lambda x: group_edu(x))
    df["education-num"] = df["education-num"].astype("category")

    # Limit age range
    df["age"] = df["age"].apply(lambda x: x // 10 * 10)
    df["age"] = df["age"].apply(lambda x: age_cut(x))

    # Group race
    df["race"] = df["race"].apply(lambda x: group_race(x))

    return df

def gen_adult():
    filepath_data = os.path.join("..", "..", "DCFR-baseline-new", "dcfr", "datasets", "adult", "adult.data")
    filepath_test = os.path.join("..", "..", "DCFR-baseline-new", "dcfr", "datasets", "adult", "adult.test")
    train = pd.read_csv(filepath_data, header=None)
    test = pd.read_csv(filepath_test, header=None)

    protected_attribute = "sex"
    categorical_features = [
        "workclass",
        "education",
        "age",
        "race",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "native-country",
        "result",
    ]
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "result",
    ]

    train.columns = columns
    test.columns = columns
    train = preprocessing_adult(train)
    test = preprocessing_adult(test)

    train.to_csv("adult.data.csv", index=False)
    test.to_csv("adult.test.csv", index=False)

    columns_json = get_cols_json(columns, categorical_features)
    with open("adult.json", "w") as json_file:
        json.dump(columns_json, json_file, indent=4)

def preprocess_compas(df):
    df = df.loc[df["days_b_screening_arrest"] <= 30]
    df = df.loc[df["days_b_screening_arrest"] >= -30]
    df = df.loc[df["is_recid"] != -1]
    df = df.loc[df["c_charge_degree"] != "O"]
    df = df.loc[df["score_text"] != "N/A"]
    return df

def gen_compas():
    filepath = os.path.join("..", "..", "DCFR-baseline-new", "dcfr", "datasets", "compas", "compas-scores-two-years.csv")
    df = pd.read_csv(filepath)
    df = preprocess_compas(df)

    protected_attribute = "race"
    categorical_features = ["sex", "age_cat", "c_charge_degree", "result"]
    columns = [
        "c_charge_degree",
        "race",
        "age_cat",
        "sex",
        "priors_count",
        "days_b_screening_arrest",
        "decile_score",
        "two_year_recid",
    ]

    df = df[columns].copy()
    df = df.rename(columns={"two_year_recid": "result"})
    df.sample(frac=1, random_state=0)
    test = df.tail(df.shape[0] // 10 * 3)
    train = df.head(df.shape[0] - test.shape[0])

    test.to_csv("compas.test.csv", index=False)
    train.to_csv("compas.data.csv", index=False)

    columns[len(columns) - 1] = "result"
    columns_json = get_cols_json(columns, categorical_features)
    with open("compas.json", "w") as json_file:
        json.dump(columns_json, json_file, indent=4)

def gen_german():
    filepath = os.path.join("..", "..", "DCFR-baseline-new", "dcfr", "datasets", "german", "german.csv")
    df = pd.read_csv(filepath)

    protected_attribute = "A13" # Age
    categorical_features = ["A1", "A3", "A4", "A6", "A7", "A9", "A10", "A12", "A14", "A15", "A17", "A19", "A20", "result"]
    columns = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
        "A17",
        "A18",
        "A19",
        "A20",
        "A21"
    ]
    
    df = df[columns].copy()
    df = df.rename(columns={"A21": "result"})
    df.sample(frac=1, random_state=0)
    test = df.tail(df.shape[0] // 10 * 3)
    train = df.head(df.shape[0] - test.shape[0])

    test.to_csv("german.test.csv", index=False)
    train.to_csv("german.data.csv", index=False)

    columns_json = get_cols_json(columns, categorical_features)
    with open("german.json", "w") as json_file:
        json.dump(columns_json, json_file, indent=4)


if __name__ == "__main__":
    gen_german()
    gen_adult()
    gen_compas()