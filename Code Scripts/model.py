#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capital Raised Analysis: Gamma GLM vs OLS Robustness
Author: Clare Zureich
Date: 2025-07-23
"""

# ===============================
# 1. Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gamma
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2

sns.set_style("whitegrid")

# ===============================
# 2. Load and Prepare Data
# ===============================
df = pd.read_csv("uk_sib_projects_full_final.csv")

# Drop rows with missing key fields (Capital Raised, Service Users, Num Investors)
model_data = df.dropna(subset=['Capital Raised', 'Service Users', 'Num Investors']).copy()

print("Initial dataset size:", df.shape[0], "projects")
print("Final dataset size after removing missing values:", model_data.shape[0], "projects")

#Check for zero values variables to be log-transformed  
zero_service_users = (model_data['Service Users'] == 0).sum()
zero_investors = (model_data['Num Investors'] == 0).sum()
zero_capital = (model_data['Capital Raised'] == 0).sum()

print(zero_service_users, zero_investors, zero_capital)

# Add log-transformed predictors
model_data['log_Service_Users'] = np.log(model_data['Service Users'])
model_data['log_Num_Investors'] = np.log(model_data['Num Investors'])

# One-hot encode Policy Sector (drop_first=True to avoid dummy trap), remove spaces from sector names
sector_dummies = pd.get_dummies(model_data['Policy Sector'], prefix='sector', drop_first=True)
sector_dummies.columns = [col.replace(" ", "_") for col in sector_dummies.columns]

# Append dummies to model_data (but keep original Policy Sector column)
model_data = pd.concat([model_data, sector_dummies], axis=1)

# Define predictors for modeling
predictors = ['log_Service_Users', 'log_Num_Investors'] + list(sector_dummies.columns)

# Prepare X and y for modeling
X_clean = model_data[predictors].astype(float)
y_clean = model_data['Capital Raised']
X_const = sm.add_constant(X_clean)

print("Predictors for modeling:", predictors)


# ===============================
# 3. Descriptive Statistics and Exploratory Plots
# ===============================
print("\n--- Descriptive Statistics for Numeric Variables ---")
print(model_data[['Capital Raised', 'Service Users', 'Num Investors']].describe().round(2))

# ===============================
# Distribution by Policy Sector
# ===============================
print("\n--- Frequency by Policy Sector ---")
sector_counts = model_data['Policy Sector'].value_counts()
print(sector_counts)

# Percentage
sector_percent = (sector_counts / sector_counts.sum() * 100).round(2)
print("\nPolicy Sector (%):")
print(sector_percent)

# ===============================
# Histograms
# ===============================
plt.figure(figsize=(15, 4))

# Capital Raised
plt.subplot(1, 3, 1)
sns.histplot(model_data['Capital Raised'], bins=30, kde=True)
plt.title('Distribution of Capital Raised')
plt.xlabel('Capital Raised (£)')

# Service Users
plt.subplot(1, 3, 2)
sns.histplot(model_data['Service Users'], bins=30, kde=True, color='orange')
plt.title('Distribution of Service Users')

# Num Investors
plt.subplot(1, 3, 3)
sns.histplot(model_data['Num Investors'], bins=15, kde=False, color='green')
plt.title('Distribution of Number of Investors')

plt.tight_layout()
plt.show()

# ===============================
# Boxplots for Outlier Detection
# ===============================
plt.figure(figsize=(12, 5))
sns.boxplot(data=model_data[['Capital Raised', 'Service Users', 'Num Investors']])
plt.title('Boxplots of Key Numeric Variables')
plt.show()

# ===============================
# Sector-Level Capital Raised Summary
# ===============================
sector_capital = model_data.groupby('Policy Sector')['Capital Raised'].agg(['mean', 'median', 'min', 'max']).round(2)
print("\n--- Capital Raised by Policy Sector ---")
print(sector_capital)



plt.figure(figsize=(12, 5))

# Capital Raised Original
plt.subplot(1, 2, 1)
sns.histplot(y_clean, bins=30, kde=True)
plt.title("Distribution of Capital Raised")
plt.xlabel("Capital Raised (£)")

# Capital Raised Log
plt.subplot(1, 2, 2)
sns.histplot(np.log(y_clean), bins=30, kde=True, color='orange')
plt.title("Log-Transformed Capital Raised")
plt.xlabel("Log(1 + Capital Raised)")

plt.tight_layout()
plt.show()

# Service Users before/after log
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(model_data['Service Users'], bins=30, kde=True)
plt.title("Original Service Users")
plt.subplot(1, 2, 2)
sns.histplot(np.log(model_data['Service Users']), bins=30, kde=True, color='orange')
plt.title("Log-Transformed Service Users")
plt.tight_layout()
plt.show()

# ===============================
# 4. Model Estimation
# ===============================
# Gamma GLM with log link
glm_model = sm.GLM(y_clean, X_const, family=Gamma(link=sm.genmod.families.links.Log())).fit()
print(glm_model.summary())
# OLS on log-transformed Y
y_log = np.log(y_clean)
ols_model = sm.OLS(y_log, X_const).fit()

# Breusch-Pagan test for heteroskedasticity
bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)

{
    "zero_service_users": zero_service_users,
    "zero_investors": zero_investors,
    "breusch_pagan": {
        "LM stat": bp_test[0],
        "LM p-value": bp_test[1],
        "F stat": bp_test[2],
        "F p-value": bp_test[3]
    }
}


# Gamma GLM with log link (and NO criminal justice sector)
# Copy the data excluding Criminal Justice
model_data_no_cj = model_data[model_data['Policy Sector'] != 'Criminal justice'].copy()
if 'sector_Criminal_justice' in model_data_no_cj.columns:
    model_data_no_cj = model_data_no_cj.drop(columns='sector_Criminal_justice')

# Rebuild predictors (log vars + remaining sector dummies)
predictors_no_cj = ['log_Service_Users', 'log_Num_Investors'] + [col for col in model_data_no_cj.columns if col.startswith('sector_')]
X_no_cj = model_data_no_cj[predictors_no_cj].astype(float)
X_no_cj_const = sm.add_constant(X_no_cj)

# Response variable
y_no_cj = model_data_no_cj['Capital Raised']

# Fit the Gamma GLM
gamma_model_no_cj = sm.GLM(y_no_cj, X_no_cj_const, family=Gamma(link=sm.genmod.families.links.Log())).fit()

# Display summary
print(gamma_model_no_cj.summary())
# ===============================
# 5. Coefficient Tables
# ===============================
# Gamma table
gamma_params = glm_model.params
gamma_conf = glm_model.conf_int()
gamma_exp = np.exp(gamma_params)
gamma_conf_exp = np.exp(gamma_conf)

gamma_table = pd.DataFrame({
    'Gamma_Coef': gamma_params,
    'Exp(Coeff)': gamma_exp,
    'p-value': glm_model.pvalues
})
gamma_table[['2.5%', '97.5%']] = gamma_conf_exp

# OLS table
ols_params = ols_model.params
ols_conf = ols_model.conf_int()
ols_table = pd.DataFrame({
    'OLS_Coef': ols_params,
    'p-value': ols_model.pvalues
})
ols_table[['2.5%', '97.5%']] = ols_conf

# Merge for side-by-side
combined_table = pd.concat([gamma_table, ols_table], axis=1).round(4)
print("\n--- Combined Coefficient Table ---")
print(combined_table)

# ===============================
# 6. VIF Table
# ===============================
vif_df = pd.DataFrame()
vif_df["feature"] = X_clean.columns
vif_df["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]
print("\n--- VIF Table ---")
print(vif_df.sort_values(by="VIF", ascending=False))

# ===============================
# 7. Model Fit Metrics
# ===============================
fit_metrics = pd.DataFrame({
    'Model': ['Gamma GLM', 'OLS (log Y)'],
    'AIC': [glm_model.aic, 'N/A'],
    'Pseudo R²': [1 - glm_model.deviance / glm_model.null_deviance, 'N/A'],
    'R²': ['N/A', ols_model.rsquared],
    'Adj R²': ['N/A', ols_model.rsquared_adj]
})
print("\n--- Model Fit Metrics ---")
print(fit_metrics)

# ===============================
# 8. Predictions and Diagnostics
# ===============================
y_pred_glm = glm_model.predict(X_const)
y_pred_ols = np.expm1(ols_model.predict(X_const))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_clean, y=y_pred_glm, alpha=0.7)
plt.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'k--')
plt.title('Gamma GLM: Actual vs Predicted')
plt.xlabel('Actual Capital Raised'); plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_clean, y=y_pred_ols, alpha=0.7, color='orange')
plt.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'k--')
plt.title('OLS: Actual vs Predicted')
plt.xlabel('Actual Capital Raised'); plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# Residual plots
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_glm, y=glm_model.resid_deviance)
plt.axhline(0, color='red', linestyle='--')
plt.title('Gamma GLM: Deviance Residuals')

plt.subplot(1, 2, 2)
sns.scatterplot(x=ols_model.fittedvalues, y=ols_model.resid, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.title('OLS: Residuals')
plt.tight_layout()
plt.show()

# ===============================
# 9. Sector Effect Visualization
# ===============================
sector_effects = gamma_table.loc[[i for i in gamma_table.index if i.startswith('sector_')], 'Exp(Coeff)']
plt.figure(figsize=(8, 5))
sector_effects.sort_values().plot(kind='barh')
plt.title('Policy Sector Effects on Capital Raised (Gamma, Exp(Coeff))')
plt.xlabel('Multiplicative Effect')
plt.show()

# ===============================
# 10. Likelihood Ration Test for Sector Dummies 
# ===============================
# Fit reduced model (without sector variables, structural predictors only)
reduced_X = X_clean[['log_Service_Users', 'log_Num_Investors']]
reduced_X_const = sm.add_constant(reduced_X)
reduced_model = sm.GLM(y_clean, reduced_X_const, family=Gamma(link=sm.genmod.families.links.Log())).fit()

# Likelihood Ratio Test
lr_stat = 2 * (glm_model.llf - reduced_model.llf)
df_diff = glm_model.df_model - reduced_model.df_model
p_val_lr = chi2.sf(lr_stat, df_diff)

print("Likelihood Ratio Test for Sector Block:")
print(f"Chi-squared = {lr_stat:.2f}, df = {df_diff}, p = {p_val_lr:.4f}")

