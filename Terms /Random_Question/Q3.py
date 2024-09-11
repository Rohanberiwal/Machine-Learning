import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

from sklearn.preprocessing import PolynomialFeatures

def preprocess_data(data):
    numerical_features = ['Construction_Year', 'Number_of_Floors', 'Energy_Consumption_Per_SqM', 
                           'Water_Usage_Per_Building', 'Waste_Recycled_Percentage', 'Occupancy_Rate', 
                           'Indoor_Air_Quality', 'Smart_Devices_Count', 'Green_Certified', 
                           'Maintenance_Resolution_Time', 'Energy_Per_SqM', 'Number_of_Residents']
    categorical_features = ['Building_Type', 'Building_Status', 'Maintenance_Priority']

    data[numerical_features] = data[numerical_features].fillna(data[numerical_features].median())
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])

    # Standardize numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Encode categorical features
    encoder = LabelEncoder()
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])

    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(data[numerical_features])
    poly_feature_names = poly.get_feature_names_out(numerical_features)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    data = pd.concat([data, poly_df], axis=1)
    data.drop(numerical_features, axis=1, inplace=True)  # Remove original numerical features

    return data


def split_data(data):
    X = data.drop('Electricity_Bill', axis=1)
    y = data['Electricity_Bill']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return mse, rmse, mae, r2, adj_r2

def k_fold_cross_validation(X, y, model, k=10):
    scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
    mean_mse = -np.mean(scores)
    mean_rmse = np.sqrt(mean_mse)
    return mean_mse, mean_rmse

def grid_search(X_train, y_train, model, param_grid):
    grid = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, -grid.best_score_

def main(filepath):
    data = load_data(filepath)
    data = preprocess_data(data)
    
    X = data.drop('Electricity_Bill', axis=1)
    y = data['Electricity_Bill']
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature matrix and target variable have different number of samples: {X.shape[0]} vs {y.shape[0]}")
    
    X_train, X_test, y_train, y_test = split_data(data)
    
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor()
    }
    
    for name, model in models.items():
        print(f"\n{name} Performance:")
        mse, rmse, mae, r2, adj_r2 = train_and_evaluate_model(X_train, y_train, X_test, y_test, model)
        mean_mse, mean_rmse = k_fold_cross_validation(X, y, model)

        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared (R²): {r2}')
        print(f'Adjusted R-squared (Adjusted R²): {adj_r2}')
        print(f'Cross-Validated Mean Squared Error (MSE): {mean_mse}')
        print(f'Cross-Validated Root Mean Squared Error (RMSE): {mean_rmse}')
        
        # Grid search for Ridge and Lasso
        if name in ['Ridge', 'Lasso']:
            param_grid = {'alpha': np.logspace(-3, 3, 7)}
            best_model, best_score = grid_search(X_train, y_train, model, param_grid)
            print(f'Best parameters: {best_model.get_params()}')
            print(f'Best cross-validated MSE: {best_score}')
    
if __name__ == "__main__":
    main('/content/Electricity BILL.csv')
