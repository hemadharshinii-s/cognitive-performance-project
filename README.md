# **Cognitive Performance and Environmental Exposure: A Multi-Level Data Analysis Using NHANES, EPA, and BRFSS Data**
CS 210: Data Management for Data Science\
Hemadharshinii Sendhilvel (hs1244)

**NOTE:** Please reference the project report for a complete and in-depth start-to-end overview of the project (including project focus, datasets used, methodology and pipelines, analyses, limitations, conclusions, and future work).

## **Project Overview**
This project investigates how cognitive performance is influenced by demographic, behavioral, health, and environmental factors.

It combines multiple large-scale public datasets to:

- Predict cognitive performance (DSST scores) using machine learning
- Analyze relationships between cognition and health behaviors
- Study environmental impacts (air quality) at the population level
- Validate findings using an external dataset (BRFSS)

The project follows a multi-level analysis approach:

- Individual-level prediction (NHANES)
- Population-level trends (NHANES + EPA AQI)
- External validation (BRFSS + EPA AQI)

### **Datasets Used**

#### **NHANES 2011-2012 & 2013–2014**
- Cognitive performance (DSST score)
- Demographics: age, gender, education
- Health: BMI, smoking status, physical activity
#### **EPA Air Quality Data (2011–2014 & 2022)**
- Daily AQI measurements
- Aggregated to yearly and state-level averages
#### **BRFSS 2022**
- Self-reported cognitive impairment
- Health and demographic indicators
- State-level aggregation

## **How to Run the Project**
This project was developed and tested in Codebench using Jupyter notebooks.\
All notebook files uploaded on GitHub contain cell outputs and can be fully viewed as-is.

Alternatively, notebooks can be run on Codebench.\
Upload all data files and notebooks (retaining folder structures) to Codebench.\
Then, run the notebooks sequentially in this order: 

1. **01_predicting_cognitive_performance.ipynb**
- Data cleaning (NHANES datasets)
- Feature engineering
- SQL-based merging
- EDA + visualization
- Machine learning models (regression)
2. **02_exploring_cognitive_performance_population_trends.ipynb**
- NHANES multi-cycle aggregation
- EPA AQI processing (2011–2014)
- Population-level comparison plots
3. **03_validating_cognitive_performance_external.ipynb**
- BRFSS preprocessing
- AQI merging at the state level
- Classification models
- Feature importance analysis

**Note:** Each notebook is independent but builds on the same thematic pipeline
