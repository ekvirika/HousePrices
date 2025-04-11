from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import mlflow

class NullHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy='median', categorical_strategy='mode', null_threshold=0.8):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_na_values = {}
        self.null_threshold = null_threshold

    def fit(self, X, y=None):
        print("Fitting NullHandler...")
        
        # Identify columns with too many nulls
        null_percentage = X.isnull().mean()
        columns_to_drop = null_percentage[null_percentage > self.null_threshold].index
        print(f"Columns with null ratio > {self.null_threshold}: {list(columns_to_drop)}")

        X_cleaned = X.drop(columns=columns_to_drop)

        # Separate column types
        self.numeric_columns = X_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = X_cleaned.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        print(f"Numeric columns to fill: {self.numeric_columns}")
        print(f"Categorical columns to fill: {self.categorical_columns}")

        # Store fill values
        for col in self.numeric_columns:
            if self.numeric_strategy == 'mean':
                self.fill_na_values[col] = X_cleaned[col].mean()
            elif self.numeric_strategy == 'median':
                self.fill_na_values[col] = X_cleaned[col].median()
            else:
                raise ValueError("Unsupported strategy for numeric columns.")
            print(f"Numeric column '{col}' will be filled with {self.numeric_strategy}: {self.fill_na_values[col]}")

        for col in self.categorical_columns:
            if self.categorical_strategy == 'mode':
                mode_val = X_cleaned[col].mode()
                self.fill_na_values[col] = mode_val[0] if not mode_val.empty else "missing"
            else:
                raise ValueError("Unsupported strategy for categorical columns.")
            print(f"Categorical column '{col}' will be filled with mode: {self.fill_na_values[col]}")

        return self

    def transform(self, X):
        print("Transforming dataset with NullHandler...")

        # Drop columns not in fill_na_values (i.e., dropped during fit)
        valid_columns = list(self.fill_na_values.keys())
        X_cleaned = X.drop(columns=[col for col in X.columns if col not in valid_columns], errors='ignore')
        X_filled = X_cleaned.copy()

        for col, fill_value in self.fill_na_values.items():
            if col in X_filled:
                print(f"Filling column '{col}' with: {fill_value}")
                X_filled[col] = X_filled[col].fillna(fill_value)

        print(f"Shape after null handling: {X_filled.shape}")
        return X_filled




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, target_column=None, null_threshold=0.8, variance_threshold=0.95, drop=True):
        self.target_column = target_column
        self.null_threshold = null_threshold
        self.variance_threshold = variance_threshold
        self.dropped_null_columns = []
        self.dropped_low_variance_columns = []
        self.drop = drop

    def fit(self, X, y=None):
        print("Fitting DataCleaner...")
        df = X.copy()

        # Identify columns with too many nulls
        null_ratio = df.isna().sum() / df.shape[0]
        self.dropped_null_columns = null_ratio[null_ratio > self.null_threshold].index.tolist()
        print(f"Columns with null ratio > {self.null_threshold}: {self.dropped_null_columns}")

        # Identify low-variance columns
        same_value_ratio = df.apply(lambda col: col.value_counts(normalize=True).max() if col.nunique() > 0 else 1)
        self.dropped_low_variance_columns = same_value_ratio[same_value_ratio > self.variance_threshold].index.tolist()
        print(f"Columns with same value ratio > {self.variance_threshold}: {self.dropped_low_variance_columns}")

        return self

    def transform(self, X):
        print("Transforming dataset with DataCleaner...")
        df = X.copy()
        to_drop = list(set(self.dropped_null_columns + self.dropped_low_variance_columns))

        if self.drop:
            print(f"Dropping columns: {to_drop}")
            df.drop(columns=to_drop, inplace=True, errors='ignore')
        else:
            print(f"Replacing NaNs in columns: {self.dropped_null_columns} with 'None'")
            for col in self.dropped_null_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('None')

        print(f"Shape after cleaning: {df.shape}")
        return df

    def plot_missing_values(self, X, threshold=0.0):
        print(f"Plotting missing value proportions for columns with > {threshold} missing data.")
        null_ratio = X.isna().sum() / X.shape[0]
        null_ratio = null_ratio[null_ratio > threshold].sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        null_ratio.plot(kind='bar')
        plt.title('Proportion of Missing Values by Column')
        plt.ylabel('Fraction of Missing Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return null_ratio

    def visualize_dropped_features(self, X, y=None):
        if y is None or self.target_column is None:
            print("Target column not provided for visualization.")
            return

        df = X.copy()
        df[self.target_column] = y
        combined_dropped = list(set(self.dropped_null_columns + self.dropped_low_variance_columns))

        print(f"Visualizing dropped features: {combined_dropped}")
        for col in combined_dropped:
            if col not in df.columns:
                continue  # Already dropped

            plt.figure(figsize=(6, 4))

            if df[col].dtype == 'object' or df[col].nunique() < 10:
                sns.boxplot(x=df[col], y=df[self.target_column])
                plt.xticks(rotation=45)
            else:
                sns.scatterplot(x=df[col], y=df[self.target_column])

            plt.title(f'{col} vs {self.target_column}')
            plt.tight_layout()
            plt.show()


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class SelectiveOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, num_unique):
        self.categorical_cols = []
        self.fill_values = {}
        self.encoder = None
        self.ohe_columns = []
        self.num_unique = num_unique

    def fit(self, X, y=None):
        print("Fitting SelectiveOneHotEncoder...")
        # Identify categorical columns with <= 3 unique non-null values
        self.categorical_cols = [
            col for col in X.select_dtypes(include='object').columns
            if X[col].nunique(dropna=True) <= self.num_unique
        ]
        print(f"Selected columns for one-hot encoding (<=3 unique values): {self.categorical_cols}")

        # Fill NaNs with mode
        for col in self.categorical_cols:
            mode = X[col].mode()[0]
            self.fill_values[col] = mode
            print(f"Filling missing values in '{col}' with mode: {mode}")

        # Fill before fitting encoder
        filled = X[self.categorical_cols].fillna(self.fill_values)
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.encoder.fit(filled)
        print("OneHotEncoder fitted on filled data.")

        self.ohe_columns = self.encoder.get_feature_names_out(self.categorical_cols)
        print(f"One-hot encoded columns will be: {self.ohe_columns.tolist()}")
        return self

    def transform(self, X):
        print("Transforming data with SelectiveOneHotEncoder...")
        X = X.copy()
        
        # Fill NaNs
        for col in self.categorical_cols:
            fill_value = self.fill_values[col]
            print(f"Filling missing values in '{col}' with: {fill_value}")
            X[col] = X[col].fillna(fill_value)

        # One-hot encode the selected columns
        ohe_array = self.encoder.transform(X[self.categorical_cols])
        ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_columns, index=X.index)
        print("One-hot encoded values:\n", ohe_df.head())

        # Drop original encoded columns and add new ones
        X = X.drop(columns=self.categorical_cols)
        X = pd.concat([X, ohe_df], axis=1)
        print(f"Final transformed dataframe with shape {X.shape}")
        return X





# Custom transformer for applying WoE encoding
class WoECategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=2, strategy='quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        self.woe_encoder = None
        
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("WoECategoricalEncoder requires target values for fitting")
        
        # Bin the target
        y_binned = self.binner.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Get categorical columns
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Apply WoE encoding
        self.woe_encoder = WOEEncoder(cols=cat_cols)
        self.woe_encoder.fit(X[cat_cols], y_binned)
        
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        
        # Get categorical columns
        cat_cols = [col for col in self.woe_encoder.cols if col in X.columns]
        if cat_cols:
            X_transformed[cat_cols] = self.woe_encoder.transform(X[cat_cols])
            
        return X_transformed



from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Custom transformer for correlation filtering
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8, target_col=None):
        self.threshold = threshold
        self.target_col = target_col
        self.features_to_drop = []

    def fit(self, X, y=None):
        print("\n[CorrelationFilter] Fitting...")
        if self.target_col is not None and y is not None:
            print(f"Using target column '{self.target_col}' for guided correlation filtering.")
            data = X.copy()
            data[self.target_col] = y

            # Correlation with target
            target_corr = data.corr()[self.target_col].drop(self.target_col).abs()
            print("Correlation with target:")
            print(target_corr.sort_values(ascending=False))

            # Correlation between features
            corr_matrix = X.corr().abs()

            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.threshold:
                        feat1 = corr_matrix.columns[i]
                        feat2 = corr_matrix.columns[j]
                        high_corr_pairs.append((feat1, feat2))

            print(f"Highly correlated feature pairs (>{self.threshold}): {high_corr_pairs}")

            # Drop the one with lower correlation to target
            features_to_drop = []
            for feat1, feat2 in high_corr_pairs:
                drop_feat = feat1 if target_corr.get(feat1, 0) <= target_corr.get(feat2, 0) else feat2
                print(f"Between '{feat1}' and '{feat2}', dropping '{drop_feat}' (lower target corr)")
                features_to_drop.append(drop_feat)

            self.features_to_drop = list(set(features_to_drop))
            print("Final list of features to drop:", self.features_to_drop)

        else:
            print("No target column provided — applying unsupervised correlation filtering.")
            corr_matrix = X.corr().abs()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            high_corr_features = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.threshold:
                        feat1 = corr_matrix.columns[i]
                        feat2 = corr_matrix.columns[j]
                        print(f"High correlation between '{feat1}' and '{feat2}': {corr_matrix.iloc[i, j]}")
                        high_corr_features.append(feat2)

            self.features_to_drop = list(set(high_corr_features))
            print("Final list of features to drop:", self.features_to_drop)

        return self

    def transform(self, X, y=None):
        print("\n[CorrelationFilter] Transforming...")
        print(f"Dropping features: {self.features_to_drop}")
        return X.drop(columns=self.features_to_drop, errors='ignore')

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("Input features not provided")
        return [f for f in input_features if f not in self.features_to_drop]
