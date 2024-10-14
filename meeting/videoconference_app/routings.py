from django.urls import re_path
from . import consumers  # Import your WebSocket consumer

websocket_urlpatterns = [
    re_path(r'ws/videocall/(?P<room_name>\w+)/$', consumers.VideoCallConsumer.as_asgi()),
]
