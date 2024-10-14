from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('meeting/', include('meeting.videoconference_app.urls')),  # Corrected path
    path('recognition/', include('recognition.recognition.urls')),
]
