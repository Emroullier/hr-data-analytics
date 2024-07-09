# Basic libraries
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

# Import from .py files
from hr_data_analytics_package_folder.api_functions.predict import predict_hiring, predict_leaving
from hr_data_analytics_package_folder.params import *
from hr_data_analytics_package_folder.api_functions.plot_api import plot_violin,\
                                                cross_tab_count_feat,\
                                                freq_feat

from fastapi import FastAPI,UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()

@app.get('/')
def index():
    return {'HR data analytics project': 'This is the first app of our project !!!'}

@app.post("/upload_predict_hiring")
def create_upload_files(upload_file: UploadFile = File(...)):
    json_data = json.load(upload_file.file)
    X_pred = pd.DataFrame(json_data)
    response = predict_hiring(X_pred)
    return {'Ranking' : response}

@app.post("/upload_predict_leaving")
def create_upload_files(upload_file: UploadFile = File(...)):
    json_data = json.load(upload_file.file)
    X_pred = pd.DataFrame(json_data)
    response = predict_leaving(X_pred)
    return {'Ranking' : response}

@app.post("/plot_violin")
def plot_violin_api(input : str):
    return FileResponse(plot_violin(input),
                        media_type="image/png")

@app.post("/plot_crosstab")
def cross_tab_count_feat_api(input : str):
    return FileResponse(cross_tab_count_feat(input),
                        media_type="image/png")

@app.post("/plot_freq")
def freq_feat_api(input : str):
    return FileResponse(freq_feat(input),
                        media_type="image/png")
