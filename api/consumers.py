import json
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
import pandas as pd
import os
from ydata_profiling import ProfileReport
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

class BackgroundTaskConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        task_type = data.get('task_type')

        if task_type == 'process_file':
            file_path = data.get('file_path')
            await self.process_file(file_path)
        elif task_type == 'profile_data':
            file_path = data.get('file_path')
            await self.profile_data(file_path)
        elif task_type == 'train_model':
            file_path = data.get('file_path')
            target_column = data.get('target_column')
            await self.train_model(file_path, target_column)
        elif task_type == 'predict':
            file_path = data.get('file_path')
            model_path = data.get('model_path')
            await self.predict(file_path, model_path)

    async def process_file(self, file_path):
        try:
            await asyncio.sleep(1)  # Simulate processing time
            df = pd.read_csv(file_path)
            metadata = {
                'columns': df.columns.tolist(),
                'delimiter': ',',
            }
            await self.send(text_data=json.dumps({'status': 'completed', 'metadata': metadata}))
        except Exception as e:
            await self.send(text_data=json.dumps({'status': 'error', 'message': str(e)}))

    async def profile_data(self, file_path):
        try:
            await asyncio.sleep(1)  # Simulate processing time
            df = pd.read_csv(file_path)
            profile = ProfileReport(df, title="Pandas Profiling Report")
            profile_path = os.path.join(os.path.dirname(file_path), 'profile_report.html')
            profile.to_file(profile_path)
            await self.send(text_data=json.dumps({'status': 'completed', 'profile_path': profile_path}))
        except Exception as e:
            await self.send(text_data=json.dumps({'status': 'error', 'message': str(e)}))

    async def train_model(self, file_path, target_column):
        try:
            await asyncio.sleep(1)  # Simulate processing time
            df = pd.read_csv(file_path)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            
            metadata = {
                "train_score": model.score(X_train, y_train),
                "test_score": model.score(X_test, y_test)
            }
            
            model_path = os.path.join(os.path.dirname(file_path), 'model.pkl')
            joblib.dump(model, model_path)
            await self.send(text_data=json.dumps({'status': 'completed', 'metadata': metadata, 'model_path': model_path}))
        except Exception as e:
            await self.send(text_data=json.dumps({'status': 'error', 'message': str(e)}))

    async def predict(self, file_path, model_path):
        try:
            await asyncio.sleep(1)  # Simulate processing time
            df = pd.read_csv(file_path)
            model = joblib.load(model_path)
            predictions = model.predict(df)
            df['predictions'] = predictions
            result_path = os.path.join(os.path.dirname(file_path), 'predictions.csv')
            df.to_csv(result_path, index=False)
            await self.send(text_data=json.dumps({'status': 'completed', 'result_path': result_path}))
        except Exception as e:
            await self.send(text_data=json.dumps({'status': 'error', 'message': str(e)}))
