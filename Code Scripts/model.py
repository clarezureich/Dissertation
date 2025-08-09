#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capital Raised Analysis: Gamma GLM vs OLS Robustness
Author: Clare Zureich
Date: 2025-07-23
"""

# ===============================
# 1. Libraries and File Paths 
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import statsmodels.api as sm
import os
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gamma
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2


# Times New Roman everywhere
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size']   = 14        # default text / tick size

# Thicker axes & grid lines
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.color']     = '#d0d0d0'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha']     = 0.7

# Consistent colour palette (muted blues / greys)
sns.set_palette(['#1f77b4', '#6baed6', '#08306b'])
sns.set_style("whitegrid")

# ===============================
# 2. Load and Prepare Data
# ===============================
df = pd.read_csv("uk_sib_projects_full_final.csv")

# Drop rows with missing key fields (Capital Raised, Service Users, Num Investors)
model_data = df.dropna(subset=['Capital Raised', 'Service Users', 'Num Investors']).copy()


##Identifying dropped projects 

key_cols = ['Capital Raised', 'Service Users', 'Num Investors']
mask_dropped = df[key_cols].isna().any(axis=1)

# 3. Slice them into a separate DataFrame
dropped = df[mask_dropped].copy()          # ≈ 24 rows
retained = df[~mask_dropped].copy()        # your modelling sample, n = 76

# 4. Quick sanity-checks
print(len(dropped))                        # should show 24
print(dropped['Policy Sector'].value_counts())    # dropped-by-sector frequency

sector_counts = dropped['Policy Sector'].value_counts()

plt.figure(figsize=(8, 5))
sector_counts.sort_values().plot(kind='barh')
plt.title("Dropped Projects by Policy Sector")
plt.xlabel("Number of Projects")
plt.ylabel("Policy Sector")
plt.tight_layout()
plt.show()

#Print dropped projects to csv 
dropped.to_csv("appendixA_dropped_projects.csv", index=False)


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

# Skewness calculations
print("\n--- Skewness ---")
for col in ['Capital Raised', 'Service Users', 'Num Investors']:
    skew_val = stats.skew(model_data[col], bias=False)
    kurt_val = stats.kurtosis(model_data[col], bias=False)
    print(f"{col}: Skewness = {skew_val:.3f}, Kurtosis = {kurt_val:.3f}")
    
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
# Original Distribution and Log-trasnformed Histograms
# ===============================

# Set up figure structure
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# --- First Histogram: Original distributions ---
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

sns.histplot(model_data['Capital Raised'], bins=30, kde=True, color='blue', ax=axs[0])
axs[0].set_title('Distribution of Capital Raised')
axs[0].set_xlabel('Capital Raised (£)')

sns.histplot(model_data['Service Users'], bins=30, kde=True, color='orange', ax=axs[1])
axs[1].set_title('Distribution of Service Users')
axs[1].set_xlabel('Service Users')

sns.histplot(model_data['Num Investors'], bins=15, kde=True, color='green', ax=axs[2])
axs[2].set_title('Distribution of Number of Investors')
axs[2].set_xlabel('Num Investors')

fig.tight_layout()
plt.show()

# --- Second figure: Log-transformed distributions (stacked) ---
fig2, axs2 = plt.subplots(3, 1, figsize=(8, 12))

sns.histplot(np.log(model_data['Capital Raised']), bins=30, kde=True, color='blue', ax=axs2[0])
axs2[0].set_title("Log-Transformed Capital Raised")
axs2[0].set_xlabel("Log(Capital Raised)")

sns.histplot(np.log(model_data['Service Users']), bins=30, kde=True, color='orange', ax=axs2[1])
axs2[1].set_title("Log-Transformed Service Users")
axs2[1].set_xlabel("Log(Service Users)")

sns.histplot(np.log(model_data['Num Investors']), bins=30, kde=True, color='green', ax=axs2[2])
axs2[2].set_title("Log-Transformed Number of Investors")
axs2[2].set_xlabel("Log(Num Investors)")

fig2.tight_layout()
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

# Plot of Project Count by Sector
plt.figure(figsize=(10, 6))
sns.barplot(x=sector_counts.values, y=sector_counts.index, palette="viridis")
plt.xlabel("Number of Projects", fontsize=14)
plt.ylabel("Policy Sector", fontsize=14)
plt.title("Number of SIB Projects per Policy Sector", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
# ===============================
# 4. Model Estimation
# ===============================
# Gamma GLM with log link
glm_model = sm.GLM(y_clean, X_const, family=Gamma(link=sm.genmod.families.links.Log())).fit()
print(glm_model.summary())
glm_model.null_deviance

# OLS on log-transformed Y
y_log = np.log(y_clean)
ols_model = sm.OLS(y_log, X_const).fit()
print(ols_model.summary())

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
plt.title('Gamma GLM: Actual vs Predicted', fontsize=16)
plt.xlabel('Actual capital raised', fontsize=16)
plt.ylabel('Predicted', fontsize=16)

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_clean, y=y_pred_ols, alpha=0.7, color='orange')
plt.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 'k--')
plt.title('OLS: Actual vs Predicted', fontsize=16)
plt.xlabel('Actual capital raised', fontsize=16)
plt.ylabel('Predicted', fontsize=16)

plt.tight_layout()
plt.savefig("diag_actual_vs_predicted.png", dpi=300)
plt.show()

# Residual plots
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_glm, y=glm_model.resid_deviance)
plt.axhline(0, color='red', ls='--')
plt.title('Gamma GLM: Deviance Residuals', fontsize=16)
plt.xlabel('Fitted', fontsize=16)
plt.ylabel('Deviance residual', fontsize=16)

plt.subplot(1, 2, 2)
sns.scatterplot(x=ols_model.fittedvalues, y=ols_model.resid, color='orange')
plt.axhline(0, color='red', ls='--')
plt.title('OLS: Residuals', fontsize=16)
plt.xlabel('Fitted', fontsize=16)
plt.ylabel('OLS residual', fontsize=16)
plt.tight_layout()
plt.show()

# Q–Q plot of deviance residuals
plt.figure(figsize=(6, 5))
stats.probplot(glm_model.resid_deviance, dist="norm", plot=plt)
plt.title("Gamma GLM: Q–Q plot of deviance residuals", fontsize=16, fontweight='bold')
plt.xlabel("Theoretical quantiles", fontsize=16)
plt.ylabel("Ordered values", fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("diag_qq_deviance.png", dpi=300)
plt.show()

# Cook’s D influence
influence  = glm_model.get_influence()
cooks_d    = influence.cooks_distance[0]
threshold  = 4 / len(cooks_d)

plt.figure(figsize=(7, 5))
markerline, stemlines, baseline = plt.stem(cooks_d, basefmt=" ")
plt.setp(markerline, marker=',', color='#1f77b4')
plt.setp(stemlines, linewidth=1, color='#1f77b4')
plt.axhline(threshold, color='#737373', ls='--', label=f"4/n = {threshold:.3f}")
plt.title("Gamma GLM: Cook’s D by observation", fontsize=16, fontweight='bold')
plt.xlabel("Project index", fontsize=16); plt.ylabel("Cook’s D", fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig("diag_cooks_distance.png", dpi=300)
plt.show()

# Over-dispersion ratio bar
pearson_chi2 = glm_model.pearson_chi2
df_resid     = glm_model.df_resid
disp_ratio   = pearson_chi2 / df_resid
print(f"Over-dispersion ratio (Pearson χ² / df): {disp_ratio:.3f}")

plt.figure(figsize=(4, 3.5))
plt.bar(['Ratio'], [disp_ratio], color='#6baed6')
plt.axhline(1, color='#737373', ls='--')
plt.title("Over-dispersion ratio", fontsize=16, fontweight='bold')
plt.ylabel("Value", fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("diag_overdispersion.png", dpi=300)
plt.show()

#Pearson residuals vs fitted
plt.figure(figsize=(6, 5))
sns.scatterplot(x=glm_model.fittedvalues, y=influence.resid_studentized)
plt.axhline(0, color='red', ls='--')
plt.title('Pearson Residuals vs Fitted Values')
plt.xlabel('Fitted')
plt.ylabel('Studentised Pearson Residual')
plt.tight_layout()
plt.savefig("diag_pearson_vs_fitted.png", dpi=300)
plt.show()


# ===============================
# 9. Sector Effect Visualization
# ===============================
sector_effects = gamma_table.loc[[i for i in gamma_table.index if i.startswith('sector_')], 'Exp(Coeff)']
sector_effects = sector_effects.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=sector_effects.values, y=sector_effects.index, palette="viridis")
plt.title('Policy Sector Effects on Capital Raised (Gamma, Exp(Coeff))', fontsize=16)
plt.xlabel('Multiplicative Effect', fontsize=16)
plt.ylabel('Policy Sector', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=16)
plt.tight_layout()
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

