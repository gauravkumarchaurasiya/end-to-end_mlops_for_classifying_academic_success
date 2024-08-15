End-to-End MLOps for Classifying Academic Success
==============================

This project demonstrates the implementation of a robust MLOps pipeline to classify academic success using machine learning techniques. By leveraging an academic success dataset, we explore data preprocessing, model training, hyperparameter tuning, and deployment within a streamlined MLOps framework. The project showcases end-to-end automation, ensuring efficient model development, continuous integration, continuous deployment, and monitoring to maintain high model performance in predicting student success outcomes.

## Project Workflow : 
![image](https://github.com/gauravkumarchaurasiya/end-to-end_mlops_for_classifying_academic_success/blob/main/images/project%20workflow.png)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── app.py             <- Application script for deployment
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── dvc.yaml           <- DVC pipeline file
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── params.yaml        <- Parameters file for the project
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── static             <- Static files for the application
    │
    ├── .github
    │   └── workflows      <- GitHub Actions workflows for CI/CD
    │
    ├── Dockerfile         <- Dockerfile for building the container image
    │
    ├── docker-compose.yml <- Docker Compose file for orchestrating multi-container Docker applications
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Project Screenshots : 
Docker Image running :   

![image](https://github.com/gauravkumarchaurasiya/end-to-end_mlops_for_classifying_academic_success/blob/main/images/docker%20run%20image.png)


## Prediction :   
![image](https://github.com/gauravkumarchaurasiya/end-to-end_mlops_for_classifying_academic_success/blob/main/images/output.png)
