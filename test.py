import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.feature_selection import RFE


class ModelMetrics:
    def __init__(self, model_type:str,train_metrics:dict,test_metrics:dict,feature_importance_df:pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"
    
    def add_train_metric(self,metric_name:str,metric_val:float):
        self.train_metrics[metric_name] = round(metric_val, 4)

    def add_test_metric(self,metric_name:str,metric_val:float):
        self.test_metrics[metric_name] = round(metric_val, 4)

    def __str__(self): 
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str

def find_dataset_statistics(dataset: pd.DataFrame, target_col: str) -> tuple[int, int, int, int, float]:
    # Get the total number of records and columns in the dataset
    n_records, n_columns = dataset.shape

    # Count the number of rows where target is negative (0) and positive (1)
    n_negative = dataset[target_col].value_counts()[0]
    n_positive = dataset[target_col].value_counts()[1]

    # Calculate the percentage of instances of positive target value
    perc_positive = n_positive / n_records * 100

    return n_records, n_columns, n_negative, n_positive, perc_positive
    
    
def train_test_split(dataset: pd.DataFrame,
                     target_col: str,
                     test_size: float,
                     stratify: bool,
                     random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X = dataset.drop(target_col, axis=1)
    y = dataset[target_col]

    if stratify:
        # Use stratified sampling if specified
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, 
                                                                                        stratify=y, 
                                                                                        random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, 
                                                                                        random_state=random_state)

    train_features, test_features, train_targets, test_targets = X_train, X_test, y_train, y_test

    return train_features, test_features, train_targets, test_targets

class PreprocessDataset:
    def __init__(self, 
                 train_features:pd.DataFrame, 
                 test_features:pd.DataFrame,
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        self.one_hot_encode_cols = one_hot_encode_cols
        self.one_hot_encoder = None
        self.train_features = train_features
        self.test_features = test_features
        self.min_max_scale_cols = min_max_scale_cols
        self.scaler = None
        self.n_components = n_components
        self.pca = None
        self.feature_engineering_functions = feature_engineering_functions

    def one_hot_encode_columns_train(self) -> pd.DataFrame:
        # Identify categorical and non-categorical columns
        categorical_cols = self.one_hot_encode_cols
        non_categorical_cols = list(set(self.train_features.columns) - set(categorical_cols))

        # Split the data into categorical and non-categorical columns
        train_features_categorical = self.train_features[categorical_cols]
        train_features_non_categorical = self.train_features[non_categorical_cols]

        # Create and fit the OneHotEncoder
        self.one_hot_encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.one_hot_encoder.fit(train_features_categorical)

        # Transform the training data
        encoded_train_array = self.one_hot_encoder.transform(train_features_categorical).toarray()
        # Get the feature names for each encoded column
        encoded_feature_names = []
        for i, col in enumerate(categorical_cols):
            encoded_feature_names += [f"{col}_{value}" for value in self.one_hot_encoder.categories_[i]]
        encoded_train_df = pd.DataFrame(encoded_train_array, index=train_features_categorical.index, 
                                        columns=encoded_feature_names)

        # Join the encoded and non-encoded dataframes
        one_hot_encoded_dataset = pd.concat([train_features_non_categorical, encoded_train_df], axis=1)

        return one_hot_encoded_dataset

    def one_hot_encode_columns_test(self) -> pd.DataFrame:
        # Identify categorical and non-categorical columns
        categorical_cols = self.one_hot_encode_cols
        non_categorical_cols = list(set(self.test_features.columns) - set(categorical_cols))

        # Split the data into categorical and non-categorical columns
        test_features_categorical = self.test_features[categorical_cols]
        test_features_non_categorical = self.test_features[non_categorical_cols]

        # Transform the test data
        encoded_test_array = self.one_hot_encoder.transform(test_features_categorical).toarray()
        # Get the feature names for each encoded column
        encoded_feature_names = []
        for i, col in enumerate(categorical_cols):
            encoded_feature_names += [f"{col}_{value}" for value in self.one_hot_encoder.categories_[i]]
        encoded_test_df = pd.DataFrame(encoded_test_array, index=test_features_categorical.index, 
                                        columns=encoded_feature_names)

        # Join the encoded and non-encoded dataframes
        one_hot_encoded_dataset = pd.concat([test_features_non_categorical, encoded_test_df], axis=1)

        return one_hot_encoded_dataset


    def min_max_scaled_columns_train(self) -> pd.DataFrame:
        # Identify numerical columns to be scaled
        numerical_cols = self.min_max_scale_cols

        # Split the data into numerical and non-numerical columns
        train_features_numerical = self.train_features[numerical_cols]
        train_features_non_numerical = self.train_features.drop(numerical_cols, axis=1)

        # Create and fit the scaler
        self.scaler = sklearn.preprocessing.MinMaxScaler()
        self.scaler.fit(train_features_numerical)

        # Transform the training data
        scaled_train_array = self.scaler.transform(train_features_numerical)
        scaled_train_df = pd.DataFrame(scaled_train_array, index=train_features_numerical.index, 
                                        columns=numerical_cols)

        # Join the scaled and non-scaled dataframes
        min_max_scaled_dataset = pd.concat([train_features_non_numerical, scaled_train_df], axis=1)

        return min_max_scaled_dataset


    def min_max_scaled_columns_test(self) -> pd.DataFrame:
        # Identify numerical columns
        numerical_cols = self.min_max_scale_cols

        # Split the data into numerical and non-numerical columns
        test_features_numerical = self.test_features[numerical_cols]
        test_features_non_numerical = self.test_features.drop(columns=numerical_cols)

        # Scale the numerical columns
        test_features_numerical_scaled = self.scaler.transform(test_features_numerical)
        test_features_numerical_scaled_df = pd.DataFrame(test_features_numerical_scaled, index=test_features_numerical.index, 
                                                         columns=numerical_cols)

        # Join the scaled and non-scaled dataframes
        min_max_scaled_dataset = pd.concat([test_features_non_numerical, test_features_numerical_scaled_df], axis=1)

        return min_max_scaled_dataset


    def pca_train(self) -> pd.DataFrame:
        # Identify numerical columns
        numerical_cols = self.min_max_scale_cols

        # Split the data into numerical and non-numerical columns
        train_features_numerical = self.min_max_scaled_columns_train()[numerical_cols]

        # Initialize PCA
        self.pca = sklearn.decomposition.PCA(n_components=min(train_features_numerical.shape[0], train_features_numerical.shape[1]), 
                                              random_state=0, svd_solver='full')
        # Fit and transform the training data
        pca_train_array = self.pca.fit_transform(train_features_numerical)
        pca_dataset = pd.DataFrame(pca_train_array, index=train_features_numerical.index, 
                                    columns=[f'component_{i+1}' for i in range(self.pca.n_components_)])

        return pca_dataset



    def pca_test(self) -> pd.DataFrame:
        # Identify numerical columns
        numerical_cols = self.min_max_scale_cols

        # Split the data into numerical and non-numerical columns
        test_features_numerical = self.min_max_scaled_columns_test()[numerical_cols]

        # Transform the test data using the trained PCA
        pca_test_array = self.pca.transform(test_features_numerical)
        pca_dataset = pd.DataFrame(pca_test_array, index=test_features_numerical.index, 
                                    columns=[f'component_{i+1}' for i in range(self.pca.n_components_)])

        return pca_dataset



    def feature_engineering_train(self) -> pd.DataFrame:
        # Create a copy of the train_features dataframe to avoid modifying the original
        feature_engineered_dataset = self.train_features.copy()

        # Apply each feature engineering function to its corresponding column
        for feature, func in self.feature_engineering_functions.items():
            feature_engineered_dataset[feature] = func(feature_engineered_dataset)

        return feature_engineered_dataset

    def feature_engineering_test(self) -> pd.DataFrame:
        # Create a copy of the test_features dataframe to avoid modifying the original
        feature_engineered_dataset = self.test_features.copy()

        # Apply each feature engineering function to its corresponding column
        for feature, func in self.feature_engineering_functions.items():
            feature_engineered_dataset[feature] = func(feature_engineered_dataset)

        return feature_engineered_dataset

    def preprocess(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        # Encode categorical columns for training set
        train_encoded = self.one_hot_encode_columns_train()
        # Scale numerical columns for training set
        train_scaled = self.min_max_scaled_columns_train()
        # Apply PCA to training set
        train_pca = self.pca_train()
        # Apply feature engineering to training set
        train_feat_eng = self.feature_engineering_train()

        # Encode categorical columns for test set
        test_encoded = self.one_hot_encode_columns_test()
        # Scale numerical columns for test set
        test_scaled = self.min_max_scaled_columns_test()
        # Apply PCA to test set
        test_pca = self.pca_test()
        # Apply feature engineering to test set
        test_feat_eng = self.feature_engineering_test()

        # Join all processed dataframes for training set
        train_features = pd.concat([train_encoded, train_scaled, train_pca, train_feat_eng], axis=1)
        # Join all processed dataframes for test set
        test_features = pd.concat([test_encoded, test_scaled, test_pca, test_feat_eng], axis=1)

        return train_features, test_features
        
class KmeansClustering:
    def __init__(self, train_features: pd.DataFrame, test_features: pd.DataFrame, random_state: int):
        self.train_features = train_features
        self.test_features = test_features
        self.random_state = random_state
        self.kmeans = None
    
    def kmeans_train(self) -> list:
        # Initialize KMeans model
        kmeans_model = sklearn.cluster.KMeans(random_state=self.random_state, n_init=10)

        # Initialize KElbowVisualizer
        visualizer = yellowbrick.cluster.KElbowVisualizer(kmeans_model, k=(1,10))

        # Fit KElbowVisualizer on training data
        visualizer.fit(self.train_features)

        # Get optimal value of k
        k_optimal = visualizer.elbow_value_

        # Train KMeans model with optimal k
        self.kmeans = sklearn.cluster.KMeans(n_clusters=k_optimal, random_state=self.random_state)
        self.kmeans.fit(self.train_features)

        # Return cluster ids for each row of training set
        cluster_ids = self.kmeans.labels_.tolist()
        return cluster_ids

    def kmeans_test(self) -> list:
        # Return cluster ids for each row of test set
        cluster_ids = self.kmeans.predict(self.test_features).tolist()
        return cluster_ids

    def train_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # Get cluster ids for each row of training set
        cluster_ids = self.kmeans_train()

        # Add cluster id column to training dataframe
        output_df = self.train_features.copy()
        output_df["kmeans_cluster_id"] = cluster_ids
        return output_df

    def test_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # Get cluster ids for each row of test set
        cluster_ids = self.kmeans_test()

        # Add cluster id column to test dataframe
        output_df = self.test_features.copy()
        output_df["kmeans_cluster_id"] = cluster_ids
        return output_df


class ModelMetrics:
    def __init__(self, model_type:str,train_metrics:dict,test_metrics:dict,feature_importance_df:pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"
    
    def add_train_metric(self,metric_name:str,metric_val:float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self,metric_name:str,metric_val:float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self): 
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str


def calculate_naive_metrics(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target_col: str, naive_assumption: int) -> ModelMetrics:
    # Split out feature and target dataframes
    train_features = train_dataset.drop(columns=[target_col])
    train_target = train_dataset[target_col]
    test_features = test_dataset.drop(columns=[target_col])
    test_target = test_dataset[target_col]

    # Create naive assumption prediction
    naive_pred = np.full(test_target.shape, naive_assumption)

    # Calculate metrics
    train_metrics = {
        "accuracy": round(accuracy_score(train_target, np.full(train_target.shape, naive_assumption)), 4),
        "recall": round(recall_score(train_target, np.full(train_target.shape, naive_assumption)), 4),
        "precision": round(precision_score(train_target, np.full(train_target.shape, naive_assumption)), 4),
        "fscore": round(f1_score(train_target, np.full(train_target.shape, naive_assumption)), 4)
    }
    test_metrics = {
        "accuracy": round(accuracy_score(test_target, naive_pred), 4),
        "recall": round(recall_score(test_target, naive_pred), 4),
        "precision": round(precision_score(test_target, naive_pred), 4),
        "fscore": round(f1_score(test_target, naive_pred), 4)
    }

    naive_metrics = ModelMetrics("Naive", train_metrics, test_metrics, None)
    return naive_metrics

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)

def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)

def calculate_logistic_regression_metrics(train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, target_col:str, logreg_kwargs) -> tuple[ModelMetrics,LogisticRegression]:
    # Split out feature and target dataframes
    train_features = train_dataset.drop(columns=[target_col])
    train_target = train_dataset[target_col]
    test_features = test_dataset.drop(columns=[target_col])
    test_target = test_dataset[target_col]

    # Train logistic regression model on training data
    model = LogisticRegression(**logreg_kwargs)
    rfe = RFE(model, n_features_to_select=10)
    rfe.fit(train_features, train_target)
    model.fit(rfe.transform(train_features), train_target)

    # Get top 10 features selected by RFE
    feature_importance = pd.DataFrame({"Feature": train_features.columns, "Importance": np.abs(model.coef_[0])})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    feature_importance = feature_importance.head(10)
    feature_importance = feature_importance.sort_values(by=["Importance", "Feature"], ascending=[False, True])

    # Predict on training and test data
    train_pred = model.predict(rfe.transform(train_features))
    train_pred_proba = model.predict_proba(rfe.transform(train_features))[:, 1]
    test_pred = model.predict(rfe.transform(test_features))
    test_pred_proba = model.predict_proba(rfe.transform(test_features))[:, 1]

    # Calculate metrics
    train_metrics = {
        "accuracy": round(accuracy_score(train_target, train_pred), 4),
        "recall": round(recall_score(train_target, train_pred), 4),
        "precision": round(precision_score(train_target, train_pred), 4),
        "fscore": round(f1_score(train_target, train_pred), 4),
        "fpr": round(false_positive_rate(train_target, train_pred), 4),
        "fnr": round(false_negative_rate(train_target, train_pred), 4),
        "roc_auc": round(roc_auc_score(train_target, train_pred_proba), 4)
    }
    test_metrics = {
        "accuracy": round(accuracy_score(test_target, test_pred), 4),
        "recall": round(recall_score(test_target, test_pred), 4),
        "precision": round(precision_score(test_target, test_pred), 4),
        "fscore": round(f1_score(test_target, test_pred), 4),
        "fpr": round(false_positive_rate(test_target, test_pred), 4),
        "fnr": round(false_negative_rate(test_target, test_pred), 4),
        "roc_auc": round(roc_auc_score(test_target, test_pred_proba), 4)
    }

    log_reg_metrics = ModelMetrics("Logistic Regression", train_metrics, test_metrics, feature_importance)
    return log_reg_metrics, model



def calculate_random_forest_metrics(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, target_col: str, rf_kwargs) -> tuple[ModelMetrics, RandomForestClassifier]:
    # Split out feature and target dataframes
    train_features = train_dataset.drop(columns=[target_col])
    train_target = train_dataset[target_col]
    test_features = test_dataset.drop(columns=[target_col])
    test_target = test_dataset[target_col]

    # Train random forest model on training data
    model = RandomForestClassifier(**rf_kwargs)
    model.fit(train_features, train_target)

    # Get feature importances
    feature_importance = pd.DataFrame({"Feature": train_features.columns, "Importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    feature_importance["Importance"] = np.abs(feature_importance["Importance"])

    # Predict on training and test data
    train_pred = model.predict(train_features)
    train_pred_proba = model.predict_proba(train_features)[:, 1]
    test_pred = model.predict(test_features)
    test_pred_proba = model.predict_proba(test_features)[:, 1]

    # Calculate metrics
    train_metrics = {
        "accuracy": round(accuracy_score(train_target, train_pred), 4),
        "recall": round(recall_score(train_target, train_pred), 4),
        "precision": round(precision_score(train_target, train_pred), 4),
        "fscore": round(f1_score(train_target, train_pred), 4),
        "fpr": round(false_positive_rate(train_target, train_pred), 4),
        "fnr": round(false_negative_rate(train_target, train_pred), 4),
        "roc_auc": round(roc_auc_score(train_target, train_pred_proba), 4)
    }
    test_metrics = {
        "accuracy": round(accuracy_score(test_target, test_pred), 4),
        "recall": round(recall_score(test_target, test_pred), 4),
        "precision": round(precision_score(test_target, test_pred), 4),
        "fscore": round(f1_score(test_target, test_pred), 4),
        "fpr": round(false_positive_rate(test_target, test_pred), 4),
        "fnr": round(false_negative_rate(test_target, test_pred), 4),
        "roc_auc": round(roc_auc_score(test_target, test_pred_proba), 4)
    }

    rf_metrics = ModelMetrics("Random Forest", train_metrics, test_metrics, feature_importance)
    return rf_metrics, model


def calculate_gradient_boosting_metrics(train_dataset:pd.DataFrame, test_dataset:pd.DataFrame, target_col:str, gb_kwargs) -> tuple[ModelMetrics,GradientBoostingClassifier]:
    # Split out feature and target dataframes
    train_features = train_dataset.drop(columns=[target_col])
    train_target = train_dataset[target_col]
    test_features = test_dataset.drop(columns=[target_col])
    test_target = test_dataset[target_col]

    # Train gradient boosting model on training data
    model = GradientBoostingClassifier(**gb_kwargs)
    model.fit(train_features, train_target)

    # Predict on training and test data
    train_pred = model.predict(train_features)
    train_pred_proba = model.predict_proba(train_features)[:, 1]
    test_pred = model.predict(test_features)
    test_pred_proba = model.predict_proba(test_features)[:, 1]

    # Calculate metrics
    train_metrics = {
        "accuracy": round(accuracy_score(train_target, train_pred), 4),
        "recall": round(recall_score(train_target, train_pred), 4),
        "precision": round(precision_score(train_target, train_pred), 4),
        "fscore": round(f1_score(train_target, train_pred), 4),
        "fpr": round(false_positive_rate(train_target, train_pred), 4),
        "fnr": round(false_negative_rate(train_target, train_pred), 4),
        "roc_auc": round(roc_auc_score(train_target, train_pred_proba), 4)
    }
    test_metrics = {
        "accuracy": round(accuracy_score(test_target, test_pred), 4),
        "recall": round(recall_score(test_target, test_pred), 4),
        "precision": round(precision_score(test_target, test_pred), 4),
        "fscore": round(f1_score(test_target, test_pred), 4),
        "fpr": round(false_positive_rate(test_target, test_pred), 4),
        "fnr": round(false_negative_rate(test_target, test_pred), 4),
        "roc_auc": round(roc_auc_score(test_target, test_pred_proba), 4)
    }

    # Get feature importances
    gb_importance = pd.DataFrame({"Feature": train_features.columns, "Importance": model.feature_importances_})
    gb_importance = gb_importance.sort_values(by=["Importance"], ascending=False)
    gb_importance["Importance"] = np.abs(gb_importance["Importance"])

    gb_metrics = ModelMetrics("Gradient Boosting", train_metrics, test_metrics, gb_importance)
    return gb_metrics, model


def train_model_return_scores(train_df_path,test_df_path) -> pd.DataFrame:
    # TODO: Load and preprocess the train and test dfs
    # Train a sklearn model using training data at train_df_path 
    # Use any sklearn model and return the test index and model scores
    

    # TODO: output dataframe should have 2 columns
    # index : this should be the row index of the test df 
    # malware_score : this should be your model's output for the row in the test df
    test_scores = pd.DataFrame()
    return test_scores 
