
## Exploratory Data Analysis (EDA)

### 1. Pair Plots
Pair plots are used to visualize the relationships between pairs of numerical features in the dataset. This helps in understanding the correlation and interaction between features.

**Code:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
sns.pairplot(data, hue='Category')  # Replace 'Category' with relevant categorical feature
plt.show()
