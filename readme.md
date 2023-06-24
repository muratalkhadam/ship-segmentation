# Project: Semantic Segmentation Solution for Airbus Challenge Task (UNet Model Implementation)

This project is a semantic segmentation solution for the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview). The goal of the challenge is to develop an algorithm that can accurately identify ships in satellite images.

## Purpose
The purpose of this project is to provide an implementation of the UNet model for semantic segmentation. By utilizing the UNet architecture, the project aims to accurately segment and identify ships in satellite images provided by the Airbus Ship Detection Challenge.

## Getting Started
To get started with this project on your local computer, follow the steps below:

1. Clone the repository:
```
git clone <repository-url>
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Project Structure
The project is organized into the following folders:
```
├───data: Contains the test and train datasets provided by the Airbus Ship Detection Challenge.
│   ├───test_v2
│   |───train_v2
|   |───sample_submission_v2.csv
|   └───train_ship_segmentations_v2.csv
├───eda: Includes Jupyter Notebook files for exploratory data analysis.
│   └───dataset-analysis.ipynb
├───log: Stores logs and model checkpoints.
└───src: Contains the main source code for the project.
    ├───models: Includes saved UNet models.
    └───unet: Contains the code specific to the UNet architecture.
```
## Summary

## Additional Information