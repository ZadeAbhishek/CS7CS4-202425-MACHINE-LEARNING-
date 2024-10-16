# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mtplt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from math import sqrt


# id:17--34--17 
df = pd.read_csv("./data/week3.csv")
df.columns = ['X1', 'X2', 'y']
df.head()

X1 = df['X1']
X2 = df['X2']
X = np.column_stack((X1, X2))
y = df['y']
point = np.column_stack((X1, X2, y))

# (i) 
### (a)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y,color='blue', s=20, label='Original Data')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Data Plot')
plt.show()

marks = np.column_stack((X1, X2, y))
center = np.mean(marks, axis=0)
_, e_values, e_vectors = np.linalg.svd(marks - center, full_matrices=False)
normal = e_vectors[2]
dispersion = e_values[2]
print('normal: ', normal)
print('dispersion: ', dispersion)

### (b) 
polynomial_values = [1, 2, 3, 4, 5]
result_list = []

for poly_degree in polynomial_values:
    poly_instance = PolynomialFeatures(degree=poly_degree)
    X_poly = poly_instance.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20, random_state=42)
    values_of_C = [1, 5, 10, 50, 100, 500, 1000]
    
    for C in values_of_C:
        lasso_reg = Lasso(alpha=1/(2*C))
        lasso_reg.fit(X_train, y_train)
        
        coef = np.around(lasso_reg.coef_, decimals=3)
        intercept = np.around(lasso_reg.intercept_, decimals=3)
        
        result_list.append({
            'Polynomial Degree': poly_degree,
            'C': C,
            'Coefficients': coef,
            'Intercept': intercept
        })

model_results = pd.DataFrame(result_list)

pd.set_option('display.max_colwidth', 600)

for poly_degree in polynomial_values:
    filtered_results = model_results[model_results['Polynomial Degree'] == poly_degree]
    
    styled_table = (
        filtered_results
        .style.set_table_styles(
            [
                {'selector': 'thead th', 'props': [('background-color', 'white'), ('color', 'black'), ('font-size', '12pt')]},
                {'selector': 'tbody td', 'props': [('background-color', 'white'), ('color', 'black'), ('border', '1px solid black')]},
            ]
        )
        .set_caption(f"Model results for Polynomial Degree {poly_degree}")
        .hide(axis='index')  # This hides the index
    )
    
    display(styled_table)



min_X = min(min(X1),min(X2)) - 1
max_X = max((max(X1),max(X2))) + 1
def kFoldModel(model_name):
    
    grid_range = np.linspace(min_X, max_X, 50)

    X1Test, X2Test = np.meshgrid(grid_range, grid_range)
    Xtest = np.c_[X1Test.ravel(), X2Test.ravel()]

    polynomial_values = [1, 2, 3, 4, 5]
    values_of_C = [1, 5, 10, 50, 100, 500, 1000]

    for poly_degree in polynomial_values: 
        poly_instance = PolynomialFeatures(degree=poly_degree)
        X_poly = poly_instance.fit_transform(X)
        X_poly_test = poly_instance.fit_transform(Xtest)

        for C in values_of_C:
            model_reg = None
            if(model_name == "Ridge"):
                model_reg = Ridge(alpha=1/(2*C))
            else:
                model_reg = Lasso(alpha=1/(2*C))
                
            model_reg.fit(X_poly, y)
            predictions = model_reg.predict(X_poly_test)
            Z = predictions.reshape(X1Test.shape)
        
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surface = ax.plot_surface(X1Test, X2Test, Z, color='lightgreen', alpha=0.6, rstride=1, cstride=1)
            scatter = ax.scatter(X1, X2, y, color='blue', s=20, label='Original Data')
            surface_proxy = mlines.Line2D([0], [0], linestyle="none", color='lightgreen', marker='s', markersize=10)
            ax.legend([surface_proxy, scatter], [f'{model_name} Predictions', 'Original Data'], numpoints=1, loc='upper left')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            ax.set_title(f'{model_name} Predictions (C={C}, Degree={poly_degree})')

            plt.show()
            
### (c) 

kFoldModel("Lasso")

### (e)
kFoldModel("Ridge")

# (ii)

def calulateErrorVsC(model_name):
    mean_error = []
    standard_deviation_error = []

    c_range = [1, 5, 10, 15, 25, 30, 40, 45, 100, 500, 1000]

    for C in c_range:
        model_reg = None
        if(model_name == "Lasso"):
            model_reg = Lasso(alpha=1/(2*C))
        else:
            model_reg = Ridge(alpha=1/(2*C))
            
        mean_square_error_array = []
        k_fold = KFold(n_splits=5)
        for train_index, test_index in k_fold.split(X):
            model_reg.fit(X[train_index], y[train_index])
            y_pred = model_reg.predict(X[test_index])
        
            # Calculate mean square error manually
            y_true = np.array(y[test_index])
            y_pred = np.array(y_pred)
            squared_diff = (y_true - y_pred) ** 2
            mean_square_error = np.mean(squared_diff)
            mean_square_error_array.append(mean_square_error)
    
        mean_error.append(np.mean(mean_square_error_array))
        standard_deviation_error.append(np.std(mean_square_error_array))

    plt.errorbar(c_range, mean_error, yerr=standard_deviation_error, fmt='-o')
    plt.xlabel('C')
    plt.ylabel(f'K-fold CV Mean Error for {model_name} Regression')
    plt.xlim((0, 55))
    plt.title(f'Error vs. C for {model_name} Regression')
    plt.show()


### (a)
calulateErrorVsC("Lasso")

### (c)
calulateErrorVsC("Ridge")


