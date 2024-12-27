# LungSense.AI
# Author: Adrian Simon
# DATASET GENERATOR FOR LUNG CANCER PREDICTION

# This script generated the dataset for the model.
# It creates a CSV file with 3000 rows and 16 columns, 15 of which are input features and the last column is the target column.
# The generated file may not be a 100% accurate representation of real-world data, but it is the closest approximation
# that I can work with, simply due to the fact that accurate datasets concerning this topic in this format
# are nearly impossible to find on kaggle. Most of the datasets on kaggle in this format seem to be purely randomly generated.
# I have generated a dataset using this algorithm and published it on kaggle with the proper disclaimers for others to use. 
# The dataset is available at: INSERT LINK
# Feel free to use this script to generate your own dataset or modify it to suit your needs.

import pandas as pd
import random

# 15 columns of input features, 1 column of target variable
columns = ["GENDER", "AGE", "SMOKING", "EXERCISE", "FAMILY_HISTORY", "HEALTHY_DIET", 
           "CHRONIC_DISEASE", "FATIGUE", "SECONDHAND", "WHEEZING", "AIR_POLLUTION", 
           "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN", 
           "LUNG_CANCER"]

# Risk calculation function that returns either 1 or 0 based on the input features
# 1 = Risk of lung cancer is present
# 0 = Risk of lung cancer is not present
def calculateRisk(inputs):
    # Weights are assigned to each feature based on their importance in determining the risk of lung cancer
    # I determined these weights based on research and reading from the following sources
    # https://www.cdc.gov/lung-cancer/risk-factors/index.html
    # https://www.cancer.org/cancer/types/lung-cancer/causes-risks-prevention/risk-factors.html
    # https://www.mayoclinic.org/diseases-conditions/lung-cancer/symptoms-causes/syc-20374620
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC3445438/
    weights = {
        "GENDER": 0.2,
        "SMOKING": 1,
        "EXERCISE": -0.5,
        "FAMILY_HISTORY": 0.5,
        "HEALTHY_DIET": -0.4,
        "CHRONIC_DISEASE": 0.7,
        "FATIGUE": 0.3,
        "SECONDHAND": 0.6,
        "WHEEZING": 0.6,
        "AIR_POLLUTION": 0.7,
        "COUGHING": 0.7,
        "SHORTNESS_OF_BREATH": 0.8,
        "SWALLOWING_DIFFICULTY": 0.5,
        "CHEST_PAIN": 0.7,
    }
    totalScore = 0
    for col in weights:
        totalScore += weights[col] * inputs[col]
    totalScore += ageScore(inputs["AGE"])
    totalScore += symptomBonus(inputs)
    if totalScore > 4:
        return 1
    else:
        return 0


# Risk of lung cancer varies severely between age groups, this function accurately represents
# the risk score for the data generation based on age.
# Before around the age of 44, the risk of lung cancer solely based on age is relatively low,
# but after that age, the risk score can be represented by a quadratic function.
# Domain: [44, 100]. Range: [0.22933, 1]
# I calculated this function based on the data from this graph:
# https://www.researchgate.net/figure/Lung-cancer-incidence-number-and-rates-per-100-000-by-age-and-sex-North-West_fig7_265574773
# The function's visualization can be found here: https://www.desmos.com/calculator/qsw8lh1jqx
def ageScore(age):
    score = 0.0
    if age < 44:
        score = 0.2
    else:
        score = (((age-78)**2 / -150) + 10) / 10
    return score

# This function adds to the total score if 2 or more symptoms are present, giving additional score for each present symptom
def symptomBonus(inputs):
    score = 0.0
    if inputs["COUGHING"] == 1:
        score += 0.5
    if inputs["SHORTNESS_OF_BREATH"] == 1:
        score += 0.5
    if inputs["CHEST_PAIN"] == 1:
        score += 0.5
    if inputs["WHEEZING"] == 1:
        score += 0.5
    if score >= 1:
        return score / 2
    else:
        return 0

rows = []

# Population of dataset with random numbers as input features
# The target variable is calculated based on the random input features using calculateRisk()
for _ in range(3000):
    row = {}
    for col in columns:
        if col == "AGE":
            row[col] = random.randint(13, 100)
        else:
            row[col] = random.randint(0, 1)
    row["LUNG_CANCER"] = calculateRisk(row)
    rows.append(row)

# Creating a DataFrame from the rows and columns
dataFrame = pd.DataFrame(rows, columns=columns)

# Saving the DataFrame to a CSV file
dataFrame.to_csv('newDataset.csv', index=False)