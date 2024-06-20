from django.urls import path
from .consumers import BackgroundTaskConsumer

websocket_urlpatterns = [
    path('ws/tasks/', BackgroundTaskConsumer.as_asgi()),
]
