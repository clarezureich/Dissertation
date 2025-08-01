#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:02:42 2025

@author: clarezureich
"""

import pandas as pd

# Load scraped and supplementary data
scraped_df = pd.read_csv("uk_sib_projects_scraped.csv")
supp_sheets = pd.read_excel('/Users/clarezureich/Documents/Applied Social Data Science/Dissertation/supplementary_data.xlsx', sheet_name=None)

# Initialize merged dataframe
merged_df = scraped_df.copy()

# Initialize log list
log_updates = []

# ---- Logging Helper ----
def log_change(sheet_name, project_id, column, old_val, new_val):
    log_updates.append({
        "Sheet": sheet_name,
        "Project ID": project_id,
        "Column": column,
        "Old Value": old_val,
        "New Value": new_val
    })

# ---- 1. Fill Missing Capital Raised ----
if "Missing Capital Raised" in supp_sheets:
    cap_df = supp_sheets["Missing Capital Raised"][["Project ID", "Capital Raised"]]
    merged_df = merged_df.merge(cap_df, on="Project ID", how="left", suffixes=("", "_supp_cap"))
    for idx, row in merged_df.iterrows():
        if pd.isna(row["Capital Raised"]) and not pd.isna(row["Capital Raised_supp_cap"]):
            log_change("Missing Capital Raised", row["Project ID"], "Capital Raised", row["Capital Raised"], row["Capital Raised_supp_cap"])
    merged_df["Capital Raised"] = merged_df["Capital Raised"].fillna(merged_df["Capital Raised_supp_cap"])
    merged_df.drop(columns=["Capital Raised_supp_cap"], inplace=True)

# ---- 2. Fill Missing Max Outcome Payment ----
if "Missing Max Outcome" in supp_sheets:
    max_df = supp_sheets["Missing Max Outcome"][["Project ID", "Max Outcome Payment"]]
    merged_df = merged_df.merge(max_df, on="Project ID", how="left", suffixes=("", "_supp_max"))
    for idx, row in merged_df.iterrows():
        if pd.isna(row["Max Outcome Payment"]) and not pd.isna(row["Max Outcome Payment_supp_max"]):
            log_change("Missing Max Outcome", row["Project ID"], "Max Outcome Payment", row["Max Outcome Payment"], row["Max Outcome Payment_supp_max"])
    merged_df["Max Outcome Payment"] = merged_df["Max Outcome Payment"].fillna(merged_df["Max Outcome Payment_supp_max"])
    merged_df.drop(columns=["Max Outcome Payment_supp_max"], inplace=True)

# ---- 3. Fill Missing Service Users ----
if "Missing Service Users" in supp_sheets:
    user_df = supp_sheets["Missing Service Users"][["Project ID", "Service Users"]]
    merged_df = merged_df.merge(user_df, on="Project ID", how="left", suffixes=("", "_supp_users"))
    for idx, row in merged_df.iterrows():
        if pd.isna(row["Service Users"]) and not pd.isna(row["Service Users_supp_users"]):
            log_change("Missing Service Users", row["Project ID"], "Service Users", row["Service Users"], row["Service Users_supp_users"])
    merged_df["Service Users"] = merged_df["Service Users"].fillna(merged_df["Service Users_supp_users"])
    merged_df.drop(columns=["Service Users_supp_users"], inplace=True)

# ---- 4. Update Num Investors if currently 0 ----
if "0 Investors" in supp_sheets:
    inv_df = supp_sheets["0 Investors"][["Project ID", "Num Investors"]]
    merged_df = merged_df.merge(inv_df, on="Project ID", how="left", suffixes=("", "_supp_inv"))
    for idx, row in merged_df.iterrows():
        if row["Num Investors"] == 0 and not pd.isna(row["Num Investors_supp_inv"]):
            log_change("0 Investors", row["Project ID"], "Num Investors", row["Num Investors"], row["Num Investors_supp_inv"])
    merged_df.loc[(merged_df["Num Investors"] == 0) & (~merged_df["Num Investors_supp_inv"].isna()), "Num Investors"] = \
        merged_df["Num Investors_supp_inv"]
    merged_df.drop(columns=["Num Investors_supp_inv"], inplace=True)

# ---- 5. Update Num Service Providers if currently 0 ----
if "0 Service Providers" in supp_sheets:
    serv_df = supp_sheets["0 Service Providers"][["Project ID", "Num Service Providers"]]
    merged_df = merged_df.merge(serv_df, on="Project ID", how="left", suffixes=("", "_supp_serv"))
    for idx, row in merged_df.iterrows():
        if row["Num Service Providers"] == 0 and not pd.isna(row["Num Service Providers_supp_serv"]):
            log_change("0 Service Providers", row["Project ID"], "Num Service Providers", row["Num Service Providers"], row["Num Service Providers_supp_serv"])
    merged_df.loc[(merged_df["Num Service Providers"] == 0) & (~merged_df["Num Service Providers_supp_serv"].isna()), "Num Service Providers"] = \
        merged_df["Num Service Providers_supp_serv"]
    merged_df.drop(columns=["Num Service Providers_supp_serv"], inplace=True)

# ---- Save merged dataset and log ----
merged_df.to_csv("uk_sib_projects_full_final.csv", index=False)
log_df = pd.DataFrame(log_updates)
log_df.to_csv("update_log.csv", index=False)

print("Merge complete.")
print("Updated dataset saved as: uk_sib_projects_full_final.csv")
print("Log of changes saved as: update_log.csv")
print(f"Total changes logged: {len(log_df)}")
print("\nSample log entries:")
print(log_df.head(10))