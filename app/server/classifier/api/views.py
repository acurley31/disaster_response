from rest_framework import generics
from classifier.models import *
from classifier.api.serializers import *


class MessageListCreateAPIView(generics.ListCreateAPIView):
    """Message list create API view"""

    queryset = Message.objects.all()
    serializer_class = MessageSerializer


class MessageRetrieveUpdateDestroyAPIView(
    generics.RetrieveUpdateDestroyAPIView):
    """Message retrieve/update/destroy API view"""

    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    lookup_field = "pk"

   
