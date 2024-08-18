import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Get data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
# Prepare data
y = df['logS']
x = df.drop('logS', axis=1)
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
# Train linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)
# Make prediction on linear regression model
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
# Evaluate linear regression model performance
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear refression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)
# Train random forest model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)
#  Make prediction on random forest model
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)
# Evaluate random forest model performance
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(rf_results)
# Combine results
df_models = pd.concat([lr_results, rf_results], axis=0)
df_models = df_models.set_index('Method')
print(df_models)
# Visualise data
plt.scatter(x=y_train, y=y_rf_train_pred, c='#7CAE00', alpha=0.3)