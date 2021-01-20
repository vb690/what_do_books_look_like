import json

from data_preprocessing import run_data_preprocessing
from model_fitting import run_model_fitting
from dimensionality_reduction import run_dimensionality_reduction


if __name__ == '__main__':
    PROJECT_NAME = input('Provide project name: ')
    PROJECT_THEMES = open(f'data\\jsons\\{PROJECT_NAME}.json')
    PROJECT_THEMES = json.load(PROJECT_THEMES)
    run_data_preprocessing(
        PROJECT_NAME=PROJECT_NAME,
        PROJECT_THEMES=PROJECT_THEMES
    )
    run_model_fitting(PROJECT_NAME=PROJECT_NAME)
    run_dimensionality_reduction(PROJECT_NAME=PROJECT_NAME)
