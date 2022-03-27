# Ad26-Dose-Down
Data and code used to draw conclusions for "Defining the determinants of protection against SARS-CoV2 infection and viral control in a dose down Ad26.CoV2.S vaccine study in non-human primates".

## Creating Python Environment
Install Anaconda from the following link: https://docs.anaconda.com/anaconda/install/

In terminal, 

```
conda create --name <environment_name>
conda activate <environment name>
```
where <environment_name> is replaced by the desired environment name. 

The required packages can also be downloaded using the following command. Python/pip should already be installed on the system if conda is working, but if not, the latest version of Python can be found and downloaded here: https://www.python.org/downloads/:
```
conda install <package_name>==<package_version>
```
See "Required Packages and Versions" for a list of names and versions. 


## Downloading R
Download R from this link: https://cran.r-project.org/bin/windows/base/, and optionally RStudio from: https://www.rstudio.com/products/rstudio/download/. Open the R application and use install.packages("package_name", version="package_version") to download requisite packages followed by library(package_name) to load packages. Then open the R files provided in this repository within the R application and run. 



## Required Packages and Versions

Python
------
python (3.8.5) \
matplotlib (3.3.2) \
numpy (1.19.2) \
pandas (1.2.1) \
seaborn (0.11.1) \
statannot (0.2.3) \
plotly (4.14.3) \
scipy (1.7.3) \
scikit-learn (0.24.2) \
 
R
-
R (4.1.2) \
plspm (0.4.9) \
devtools (2.4.3) \


The sequence of terminal commands required to generate all data and figures from code: 
In terminal, navigate to the folder "Systems_Serology.py" was downloaded to (or stored post-download). Create a subdirectory to store initial data files; in the process of running code, "Processed_Data" subdirectory will additionally be created and automatically populated. 

## General Preprocessing (enter this command first)
```
python Systems_Serology.py --datapath (path to Ad26_dosedown.csv) --task Preprocessing --var IgG1_Empty --var IgG2_Empty --var IgG3_Empty --var IgG4_Empty --var IgA_Empty --var IgM_Empty --var FcgR2A-1_Empty --var FcgR2A-2_Empty --var FcgR2A-3_Empty
```

## Figures 3A and 4A (wild-type Spike feature flowers), 4C (volcano plot), 4D (Lasso PLS-DA), Suppl. fig. 1 (wild-type Spike correlation maps)
```
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --task Remove_Columns --key Florian --key Erica
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --task Keep_Columns --key WT --key NK --key Cov-2 --key score --key Sample --key Outcome --key ProtectionBALorSwab --key Neutralization_Group --key Group --key NAbs_Week_6 --key IFNg
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Keep_Columns --key Spike --key NK --key score --key Sample --key Outcome --key ProtectionBALorSwab --key Neutralization_Group --key Group --key NAbs_Week_6 --key IFNg
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Columns --key Erica --key Lake
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Samples --key sham --labels Group
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task Flowers --group Group --key Ig --key Fcg --key NAb --key cell
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task Flowers --group ProtectionBALorSwab --key Ig --key Fcg --key NAb --key cell
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task PCA --group ProtectionBALorSwab --draw_ellipses True --num_loadings 15 --fontsize 20
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task Most_Predictive --group ProtectionBALorSwab --string Holm
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task Feature_Selection --alpha 0.01 --group Outcome
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyFeatureSelected.csv --task PLS --group Outcome --var Outcome --draw_ellipses True --fontsize 16

## NOTE: for correlation heatmaps, change 'key' to I, II, III, IV to generate the heatmap corresponding to that group. ##

python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task Subset_Data --group Group --key IV
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySubset.csv --task Correlations --title Dose_Group_IV
```
For supplementary figure 2, all relevant data is found in "Enrichment_scores.csv". 


## Figure 3C and 3D (PCA), 4B (PCA)
```
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --task Remove_Columns --key Florian --key Erica
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Keep_Columns --key WT --key NK --key Cov-2 --key score --key Sample --key ProtectionBALorSwab --key Outcome --key Neutralization_Group --key Group --key NAbs_Week_6 --key IFNg
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Columns --key Lake
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Samples --key sham --labels Group
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task PCA --group Group --draw_ellipses True --num_loadings 15 --fontsize 20
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task PCA --group Outcome --draw_ellipses True --num_loadings 15 --fontsize 16
```

## Figure 5B (FcgR profile PLS-DA)
```
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --task Keep_Columns --key RBD --key NTD --key S2 --key Outcome --key Group
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Columns --key B.1 --key E484K --key AD
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Samples --key sham --labels Group
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --task Remove_Columns --key Ig

## NOTE: the day 10 AUC BAL column should be manually transferred from Ad26_BAL_results.csv to SystemsSerologyManualFeatureSelection.csv. (replace spaces in column name with underscores) ##

python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task PLS --var Day_10_AUC_BAL --num_loadings 15 --group Outcome --draw_ellipses True
```  

## Figure 6 (Preparation for PLS-PM)
```
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --task Remove_Columns --key Florian --key Erica
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --task Keep_Columns --key WT --key NK --key Cov-2 --key score --key Sample --key Outcome --key ProtectionBALorSwab --key Neutralization_Group --key Group --key NAbs_Week_6 --key IFNg
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Keep_Columns --key Spike --key NK --key score --key Sample --key Outcome --key ProtectionBALorSwab --key Neutralization_Group --key Group --key NAbs_Week_6 --key IFNg
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Columns --key Erica --key Lake
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task Remove_Samples --key sham --labels Group
python Systems_Serology.py --datapath Processed_Data\SystemsSerologySamplesRemoved.csv --datapath2 (path to Ad26_BAL_results.csv) --task Preprocessing
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --datapath2 Processed_Data\SystemsSerologySecondDataset.csv --task Move_Columns --key Day_10_AUC_BAL
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --datapath2 (path to Ad26_nasal_swab_results.csv) --task Preprocessing
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --datapath2 Processed_Data\SystemsSerologySecondDataset.csv --task Move_Columns --key Day_10_AUC_NS

## NOTE: at this point, run Ad26_PLSPM_Preprocessing.py ##

python Systems_Serology.py --datapath Processed_Data\SystemsSerologyZscored.csv --datapath2 Processed_Data\SystemsSerology.csv --task Move_Columns --key Outcome
```
Note that at this point, the rest of figure 6 can be generated using the .Rmd script. 


## Supplementary figure 3B-3D (variant enrichment and PCA)
All relevant data can be found in "Enrichment_scores.csv". To generate the PCA in supplementary figure 3D, 
```
python Systems_Serology.py --datapath Analysis\Enrichment_scores.csv --task Preprocessin
python Systems_Serology.py --datapath Processed_Data\SystemsSerology.csv --task Keep_Columns --key Variant --key Outcome
python Systems_Serology.py --datapath Processed_Data\SystemsSerologyManualFeatureSelection.csv --task PCA --group Outcome --draw_ellipses True --fontsize 16
```
