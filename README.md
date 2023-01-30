# Benchmarking-Analysis-of-PLP-vs-MLR3 
This repo contains the model scripts and the cohort definitions used for a comparison of PLP and mlr3 R-based package using the SynPUF 5% dataset.


# Database configuration
"DatabaseSettings.txt" 

# Cohort defination
Cohort defination was performed using the [ATLAS](https://atlas-demo.ohdsi.org/#/home) platform. The data was imported into a locally running ATLAS and cohort defination option was used. "Ischemia with Atrial Fib.json" file includes the target population defination used in the study. In addition "Death.json" file describes our target cohort within the target population. The tagret population and the target cohort predicted by the machine learning models is shown in "prediction.json". 

# R and RStudio
In this study RStudio version 2022.07.2 Build 576 with R version 4.2.2 was used. 

# ml3 package
In this study we utilized MLR3 v.0.14.1.

# PLP package 
In this study we utilized PLP version v6.0.8.






