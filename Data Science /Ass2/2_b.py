import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_kddcup99
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.sparse import lil_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_kddcup99
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse.linalg import lsqr

loss_a = []
loss_b = []

data = fetch_kddcup99(as_frame=True)
X = data.data.sample(frac=0.1, random_state=42)
y = data.target[X.index]

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)
    ]
)


X_processed = preprocessor.fit_transform(X)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


n_samples, n_features_encoded = X_processed.shape
JL_dim = 10  
sparsity_param = 3  
iterations = 5  


def generate_sparse_jl_matrix(d, n, projection_dim=10, k=1):
    M = lil_matrix((projection_dim, n))
    prob_nonzero = 1 / 6
    prob_zero = 2 / 3
    row_scale = 3 / d
    root_val = np.sqrt(row_scale)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rand_val = np.random.rand()
            if rand_val < prob_nonzero:
                M[i, j] = root_val if np.random.rand() < 0.5 else -root_val
            elif rand_val < prob_nonzero + prob_zero:
                M[i, j] = 0
            else:
                M[i, j] = 1 if np.random.rand() < 0.5 else -1
    return M

def main():
    for iter in range(iterations):
        M = generate_sparse_jl_matrix(n_features_encoded, n_samples, projection_dim=JL_dim, k=sparsity_param)
        E = M @ X_processed
        z = M @ y.astype(float)
        a = lsqr(E, z)[0]
        b = lsqr(X_processed, y.astype(float))[0]
        y_pred_a = X_processed @ a
        y_pred_b = X_processed @ b
        loss_a.append(np.mean((y_pred_a - y.astype(float)) ** 2))
        loss_b.append(np.mean((y_pred_b - y.astype(float)) ** 2))


    results_df = pd.DataFrame({
        "Iteration": np.arange(1, iterations + 1),
        "Loss_a": loss_a,
        "Loss_b": loss_b
    })


    plt.figure(figsize=(10, 6))
    plt.bar(results_df["Iteration"] - 0.2, results_df["Loss_a"], width=0.4, label="Loss using a (Projected)")
    plt.bar(results_df["Iteration"] + 0.2, results_df["Loss_b"], width=0.4, label="Loss using b (Original)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Comparison: Projected vs Original Regression Solutions (Sparse Representation)")
    plt.xticks(results_df["Iteration"])
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(results_df)
    
main()
