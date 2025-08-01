There are 3 code scripts in this folder. Run in the following order: 
1. api_code.py
2. add_supplementary.py
3. model.py

The api_code.py scrapes the INDIGO Impact Bond Database webiste (https://golab.bsg.ox.ac.uk/knowledge-bank/indigo/impact-bond-dataset-v2/), filters UK, and saves all of the fields available for all projects. The result of this code is saved in the uk_sib_projects_scraped.csv file found in the Data Sources folder. As of August 1, 2025 there are SIB 100 projects in the UK in the INDIGO Database
The add_supplementary.py appends the supplementary data to the scraped database. Projects with missing "Captial Raised", "Number of Investors", or "Number of Service Users" were idenfitied. Exteneral research was conducted to supplement the missing values. The update_log.csv shows the changes made when appending the supplementary data to the uk_sib_projects_scraped.csv. The resulting file from this script is the uk_sib_projects_full_final.csv, which is used for the final model analysis.
The model.py script runs the uk_sib_projects_full_final.csv through the Gamma GLM log-link model and OLS robustness test, as well as through several descriptive statstics and heteroskedasticty tests. There are 24 SIB projects with missing values that we dropped from the final csv, so a total of 76 were used. 
