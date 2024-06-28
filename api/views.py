from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.cache import cache
from .models import UploadedFile
from .utils.delimiter import detect_separator, load_data_dynamic, detect_separator_content
from .utils.trainingUtil import get_meta, convert_sets_to_lists, convert_lists_to_sets, train_with_parameter_tuning_test, make_prediction
import os
import io
import simplejson as json
import pandas as pd
from ydata_profiling import ProfileReport
import joblib
import numpy as np
from django.conf import settings
















class FileUploadView(APIView):
    def post(self, request):
        """
        Handles the upload of a file, detects the delimiter, and caches the dataframe.

        Args:
            request (Request): The HTTP request object containing the uploaded file.

        Returns:
            Response: A response object containing metadata about the uploaded file.
        """
        file = request.FILES['file']
        original_name = file.name
        uploaded_file = UploadedFile(file=file, original_name=original_name)
        uploaded_file.save()

        # Detect the delimiter
        file_path = uploaded_file.file.path
        detected_delimiter = detect_separator(file_path)
        if not detected_delimiter:
            return Response({'error': 'Failed to detect delimiter'}, status=status.HTTP_400_BAD_REQUEST)

        # Load the data using the detected delimiter
        df = load_data_dynamic(file_path, detected_delimiter)
        if df is None:
            return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the dataframe to cache for further use
        cache_key = f'df_{uploaded_file.id}'
        cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour

        uploaded_file.delimiter = detected_delimiter
        uploaded_file.selected_delimiter = detected_delimiter
        uploaded_file.save()

        metadata = {
            'file_id': str(uploaded_file.id),
            'delimiter': detected_delimiter,
            'columns': df.columns.tolist()
        }

        return Response(metadata, status=status.HTTP_201_CREATED)

class DataPreviewView(APIView):
    def post(self, request):
        """
        Provides a preview of the first 10 rows of the uploaded data.

        Args:
            request (Request): The HTTP request object containing the file ID and delimiter.

        Returns:
            Response: A response object containing the data preview.
        """
        file_id = request.data['file_id']
        delimiter = request.data['delimiter']
        uploaded_file = UploadedFile.objects.get(id=file_id)

        try:
            df = load_data_dynamic(uploaded_file.file.path, delimiter)
            if df is None:
                return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)

            # Replace non-JSON-compliant values with None
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(value=np.nan, inplace=True)
            df = df.where(pd.notnull(df), None)

            # Provide a preview of the first 10 rows
            preview = df.head(10).to_dict(orient='records')

            # Update selected delimiter in the database
            uploaded_file.selected_delimiter = delimiter
            uploaded_file.save()

            return Response({'preview': preview}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ProfileDataView(APIView):
    def post(self, request):
        """
        Generates and returns a profiling report of the uploaded data.

        Args:
            request (Request): The HTTP request object containing the file ID.

        Returns:
            Response: A response object containing the path to the profiling report.
        """
        file_id = request.data['file_id']
        uploaded_file = UploadedFile.objects.get(id=file_id)

        if uploaded_file.profile_generated:
            return Response({'profile_path': uploaded_file.profile_report_path}, status=status.HTTP_200_OK)

        # Check if DataFrame is in cache
        cache_key = f'df_{file_id}'
        df = cache.get(cache_key)
        if df is None:
            df = load_data_dynamic(uploaded_file.file.path, uploaded_file.delimiter)
            if df is None:
                return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)
            cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour

        profile = ProfileReport(df, title="Pandas Profiling Report")
        profile_path = os.path.join(os.path.dirname(uploaded_file.file.path), f'profile_report_{file_id}.html')
        profile.to_file(profile_path)
        uploaded_file.profile_report_path = profile_path
        uploaded_file.profile_generated = True
        uploaded_file.save()
        return Response({'profile_path': profile_path}, status=status.HTTP_200_OK)


class GetMetadataView(APIView):
    def post(self, request):
        """
        Retrieves metadata of the uploaded data.

        Args:
            request (Request): The HTTP request object containing the file ID and target column.

        Returns:
            Response: A response object containing the metadata.
        """
        file_id = request.data['file_id']
        target_column = request.data['target_column']
        uploaded_file = UploadedFile.objects.get(id=file_id)
        
        # Check if metadata already exists
        if uploaded_file.model_metadata_path:
            uploaded_file.target_column = target_column
            uploaded_file.save()
            with open(uploaded_file.model_metadata_path, 'r') as f:
                metadata = json.load(f)
            modified_metadata = None
            if uploaded_file.modified_metadata_path:
                with open(uploaded_file.modified_metadata_path, 'r') as f:
                    modified_metadata = json.load(f)
            training_info = {
                'status': 'Model trained successfully',
                'target_type': uploaded_file.target_type,
                'model_type': uploaded_file.model_type,
                'performance_score': uploaded_file.performance_score,
                'model_path': uploaded_file.model_path,
                'num_trials': uploaded_file.num_trials
            } if uploaded_file.model_trained else None
            return Response({
                'default_metadata': metadata,
                'modified_metadata': modified_metadata or metadata,
                'metadata_path': uploaded_file.model_metadata_path,
                'training_info': training_info
            }, status=status.HTTP_200_OK)
        
        # Check if DataFrame is in cache
        cache_key = f'df_{file_id}'
        df = cache.get(cache_key)
        if df is None:
            df = load_data_dynamic(uploaded_file.file.path, uploaded_file.delimiter)
            if df is None:
                return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)
            cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour

        # Basic metadata fetching using utility function
        data, metadata = get_meta(df, target_column)

        # Save metadata to file
        metadata_path = os.path.join(os.path.dirname(uploaded_file.file.path), f'metadata_{file_id}.json')
        with open(metadata_path, 'w') as f:
            json.dump(convert_sets_to_lists(metadata), f)

        uploaded_file.model_metadata_path = metadata_path
        uploaded_file.target_column = target_column
        uploaded_file.save()

        return Response({
            'default_metadata': metadata,
            'modified_metadata': metadata,
            'metadata_path': metadata_path
        }, status=status.HTTP_200_OK)



class UpdateMetadataView(APIView):
    def post(self, request):
        """
        Updates metadata of the uploaded data.

        Args:
            request (Request): The HTTP request object containing the file ID and modified metadata.

        Returns:
            Response: A response object containing the updated metadata.
        """
        file_id = request.data['file_id']
        modified_metadata = request.data['modified_metadata']
        modified_metadata = convert_lists_to_sets(modified_metadata)

        uploaded_file = UploadedFile.objects.get(id=file_id)

        # Prepare forced types
        forced_types = {column: 'cat' if metadata['is_cat'] else 'non_cat' for column, metadata in modified_metadata.items()}

        # Check if DataFrame is in cache
        cache_key = f'df_{file_id}'
        df = cache.get(cache_key)
        if df is None:
            df = load_data_dynamic(uploaded_file.file.path, uploaded_file.delimiter)
            if df is None:
                return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)
            cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour

        # Basic metadata fetching using utility function
        data, newModifiedMetadata = get_meta(df, uploaded_file.target_column, forced_types)





        uploaded_file = UploadedFile.objects.get(id=file_id)
        modified_metadata_path = os.path.join(os.path.dirname(uploaded_file.file.path), f'modified_metadata_{file_id}.json')
        with open(modified_metadata_path, 'w') as f:
            json.dump(convert_sets_to_lists(newModifiedMetadata), f)
        uploaded_file.modified_metadata_path = modified_metadata_path
        uploaded_file.save()
        return Response({'modified_metadata': convert_sets_to_lists(newModifiedMetadata)}, status=status.HTTP_200_OK)


class GetStoredMetadataView(APIView):
    def get(self, request):
        """
        Retrieves stored metadata for the specified file ID.

        Args:
            request (Request): The HTTP request object containing the file ID.

        Returns:
            Response: A response object containing the stored metadata.
        """
        file_id = request.query_params.get('file_id')
        uploaded_file = UploadedFile.objects.get(id=file_id)
        with open(uploaded_file.model_metadata_path, 'r') as f:
            metadata = json.load(f)
        modified_metadata_path = uploaded_file.modified_metadata_path
        if modified_metadata_path and os.path.exists(modified_metadata_path):
            with open(modified_metadata_path, 'r') as f:
                modified_metadata = json.load(f)
        else:
            modified_metadata = metadata
        return Response({
            'default_metadata': convert_sets_to_lists(metadata),
            'modified_metadata': convert_sets_to_lists(modified_metadata)
        }, status=status.HTTP_200_OK)














class TrainModelView(APIView):
    def post(self, request):
        """
        Trains a model with the uploaded data and specified parameters.

        Args:
            request (Request): The HTTP request object containing the file ID, target column, and number of trials.

        Returns:
            Response: A response object containing the training information.
        """
        file_id = request.data['file_id']
        target_column = request.data['target_column']
        num_trials = int(request.data.get('num_trials', 10))  # Ensure num_trials is an integer
        uploaded_file = UploadedFile.objects.get(id=file_id)

        # Check if DataFrame is in cache
        cache_key = f'df_{file_id}'
        df = cache.get(cache_key)
        if df is None:
            df = load_data_dynamic(uploaded_file.file.path, uploaded_file.delimiter)
            if df is None:
                return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)
            cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour

        # Fetch modified metadata from the session or database
        modified_metadata_path = uploaded_file.modified_metadata_path or uploaded_file.model_metadata_path
        if modified_metadata_path:
            with open(modified_metadata_path, 'r') as f:
                modified_metadata = json.load(f)
        else:
            modified_metadata = uploaded_file.model_metadata_path  # Use stored metadata if no modifications

        # Prepare forced types
        forced_types = {column: 'cat' if metadata['is_cat'] else 'non_cat' for column, metadata in modified_metadata.items()}

        # Clean previous model if it exists
        if uploaded_file.model_path:
            try:
                os.remove(uploaded_file.model_path)
            except Exception as e:
                print(f"Failed to delete model file: {e}")

        # Train model using parameter tuning
        status_msg, target_type, model_type, performance_score, model_path, meta_path = train_with_parameter_tuning_test(
            df, forced_types, target_column, num_trials, uploaded_file.file.path
        )

        # Save training details to the database
        uploaded_file.model_trained = True
        uploaded_file.target_type = target_type
        uploaded_file.model_type = model_type
        uploaded_file.performance_score = performance_score
        uploaded_file.model_path = model_path
        uploaded_file.num_trials = num_trials
        uploaded_file.save()

        # Prepare training info for response
        training_info = {
            'status': status_msg,
            'target_type': target_type,
            'model_type': model_type,
            'performance_score': performance_score,
            'model_path': model_path,
            'num_trials': num_trials
        }

        return Response({
            'training_info': training_info
        }, status=status.HTTP_200_OK)


class GetTrainingStatusView(APIView):
    def get(self, request, file_id):
        """
        Retrieves the training status of the model for the specified file ID.

        Args:
            request (Request): The HTTP request object.
            file_id (str): The ID of the uploaded file.

        Returns:
            Response: A response object containing the training status.
        """
        try:
            uploaded_file = UploadedFile.objects.get(id=file_id)
            if uploaded_file.model_trained:
                return Response({
                    'status': 'Model already trained',
                    'target_type': uploaded_file.target_type,
                    'model_type': uploaded_file.model_type,
                    'performance_score': uploaded_file.performance_score,
                    'model_path': uploaded_file.model_path
                }, status=status.HTTP_200_OK)
            else:
                return Response({'status': 'Model not trained'}, status=status.HTTP_200_OK)
        except UploadedFile.DoesNotExist:
            return Response({'error': 'File not found'}, status=status.HTTP_404_NOT_FOUND)











class ColumnListView(APIView):
    def post(self, request):
        """
        Retrieves the list of columns in the uploaded data.

        Args:
            request (Request): The HTTP request object containing the file ID.

        Returns:
            Response: A response object containing the list of columns.
        """
        file_id = request.data['file_id']
        uploaded_file = UploadedFile.objects.get(id=file_id)

        # Check if DataFrame is in cache
        cache_key = f'df_{file_id}'
        df = cache.get(cache_key)
        if df is None:
            df = load_data_dynamic(uploaded_file.file.path, uploaded_file.delimiter)
            if df is None:
                return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)
            cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour

        columns = df.columns.tolist()
        return Response({'columns': columns}, status=status.HTTP_200_OK)














class PredictView(APIView):
    def post(self, request):
        """
        Makes predictions using the trained model on new data.

        Args:
            request (Request): The HTTP request object containing the file ID and prediction type.

        Returns:
            Response: A response object containing the prediction results.
        """
        file_id = request.data['file_id']
        prediction_type = request.data['prediction_type']
        uploaded_file = UploadedFile.objects.get(id=file_id)

        # Check if DataFrame is in cache
        cache_key = f'df_{file_id}'
        df = cache.get(cache_key)
        if df is None:
            df = load_data_dynamic(uploaded_file.file.path, uploaded_file.delimiter)
            if df is None:
                return Response({'error': 'Failed to read file'}, status=status.HTTP_400_BAD_REQUEST)
            cache.set(cache_key, df, timeout=3600)  # Cache for 1 hour

        # Load the trained model
        if not uploaded_file.model_path:
            return Response({'error': 'Model not found'}, status=status.HTTP_400_BAD_REQUEST)

        # Use modified metadata if available, otherwise default metadata
        meta_path = uploaded_file.modified_metadata_path or uploaded_file.model_metadata_path
        if not meta_path:
            return Response({'error': 'Metadata not found'}, status=status.HTTP_400_BAD_REQUEST)

        if prediction_type == 'single':
            single_data = json.loads(request.data['single_data'])
            input_df = pd.DataFrame([single_data])
            prediction_value = make_prediction(input_df, uploaded_file.model_path, meta_path, uploaded_file.target_column)
            prediction_html = prediction_value.to_html(classes="table table-striped")
            return Response({'prediction_html': prediction_html}, status=status.HTTP_200_OK)

        elif prediction_type == 'group':
            prediction_file = request.FILES['file']
            prediction_file_content = prediction_file.read().decode('utf-8')

            # Detect delimiter
            delimiter = detect_separator_content(prediction_file_content)
            if not delimiter:
                return Response({'error': 'Failed to detect delimiter'}, status=status.HTTP_400_BAD_REQUEST)

            prediction_df = pd.read_csv(io.StringIO(prediction_file_content), delimiter=delimiter)
            prediction_value = make_prediction(prediction_df, uploaded_file.model_path, meta_path, uploaded_file.target_column)
            prediction_html = prediction_value.to_html(classes="table table-striped")
            custom_css = """<style> .table thead th { text-align: left !important; } </style>"""

            return Response({'prediction_html': custom_css + prediction_html}, status=status.HTTP_200_OK)

        else:
            return Response({'error': 'Invalid prediction type'}, status=status.HTTP_400_BAD_REQUEST)






















class TestView(APIView):
    def post(self, request):
        """
        Returns a test message.

        Args:
            request (Request): The HTTP request object.

        Returns:
            Response: A response object containing a test message.
        """
        return Response({'message': 'API is working!'}, status=status.HTTP_200_OK)


class TaskStatusView(APIView):
    def get(self, request, task_id):
        """
        Returns the status of the specified task.

        Args:
            request (Request): The HTTP request object.
            task_id (str): The ID of the task.

        Returns:
            Response: A response object containing the task status.
        """
        # You need to implement proper task status tracking if you have an async task queue
        return Response({'task_id': task_id, 'status': 'completed'}, status=status.HTTP_200_OK)


