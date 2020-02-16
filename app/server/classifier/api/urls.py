from django.urls import path
from classifier.api.views import *


urlpatterns = [
    path("messages/", MessageListCreateAPIView.as_view()),
    path("messages/<int:pk>/", MessageRetrieveUpdateDestroyAPIView.as_view()),
]
