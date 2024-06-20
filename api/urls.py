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
