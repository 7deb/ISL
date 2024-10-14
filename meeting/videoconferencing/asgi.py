import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from videoconference_app import routing  # Import your WebSocket routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'videoconferencing.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            routing.websocket_urlpatterns  # Add WebSocket URLs here
        )
    ),
})
