from django.urls import path
from .views import (
    FileUploadView, ProfileDataView, TrainModelView, GetTrainingStatusView, PredictView, TestView, 
    TaskStatusView, DataPreviewView, GetMetadataView, UpdateMetadataView, 
    GetStoredMetadataView, ColumnListView
)

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('profile/', ProfileDataView.as_view(), name='profile-data'),
    path('train/', TrainModelView.as_view(), name='train-model'),
    path('get_training_status/<uuid:file_id>/', GetTrainingStatusView.as_view(), name='get_training_status'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('test/', TestView.as_view(), name='test-api'),
    path('task_status/<str:task_id>/', TaskStatusView.as_view(), name='task-status'),
    path('data_preview/', DataPreviewView.as_view(), name='data-preview'),
    path('get_metadata/', GetMetadataView.as_view(), name='get-metadata'),
    path('update_metadata/', UpdateMetadataView.as_view(), name='update-metadata'),
    path('get_stored_metadata/', GetStoredMetadataView.as_view(), name='get-stored-metadata'),
    path('get_columns/', ColumnListView.as_view(), name='get-columns')
]

"""
URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/

Routes:
    - **upload/**:
        - **Description**: Uploads a file and returns metadata.
        - **View**: :class:`~api.views.FileUploadView`
        - **Method**: POST
        - **Request**: multipart/form-data
        - **Response**: JSON

    - **profile/**:
        - **Description**: Generates a profiling report for the uploaded data.
        - **View**: :class:`~api.views.ProfileDataView`
        - **Method**: POST
        - **Request**: JSON
        - **Response**: JSON

    - **train/**:
        - **Description**: Trains a model based on the uploaded data.
        - **View**: :class:`~api.views.TrainModelView`
        - **Method**: POST
        - **Request**: JSON
        - **Response**: JSON

    - **get_training_status/<uuid:file_id>/**:
        - **Description**: Retrieves the training status of the model.
        - **View**: :class:`~api.views.GetTrainingStatusView`
        - **Method**: GET
        - **Response**: JSON

    - **predict/**:
        - **Description**: Makes predictions based on the trained model.
        - **View**: :class:`~api.views.PredictView`
        - **Method**: POST
        - **Request**: multipart/form-data or JSON
        - **Response**: HTML

    - **test/**:
        - **Description**: A simple test endpoint to ensure the API is working.
        - **View**: :class:`~api.views.TestView`
        - **Method**: POST
        - **Response**: JSON

    - **task_status/<str:task_id>/**:
        - **Description**: Checks the status of a specific task.
        - **View**: :class:`~api.views.TaskStatusView`
        - **Method**: GET
        - **Response**: JSON

    - **data_preview/**:
        - **Description**: Provides a preview of the uploaded data.
        - **View**: :class:`~api.views.DataPreviewView`
        - **Method**: POST
        - **Request**: JSON
        - **Response**: JSON

    - **get_metadata/**:
        - **Description**: Fetches metadata of the uploaded data.
        - **View**: :class:`~api.views.GetMetadataView`
        - **Method**: POST
        - **Request**: JSON
        - **Response**: JSON

    - **update_metadata/**:
        - **Description**: Updates the metadata of the uploaded data.
        - **View**: :class:`~api.views.UpdateMetadataView`
        - **Method**: POST
        - **Request**: JSON
        - **Response**: JSON

    - **get_stored_metadata/**:
        - **Description**: Retrieves stored metadata of the uploaded data.
        - **View**: :class:`~api.views.GetStoredMetadataView`
        - **Method**: GET
        - **Response**: JSON

    - **get_columns/**:
        - **Description**: Retrieves the column names of the uploaded data.
        - **View**: :class:`~api.views.ColumnListView`
        - **Method**: POST
        - **Request**: JSON
        - **Response**: JSON
"""
