import pickle
from rest_framework import serializers
from django.contrib.staticfiles import finders

from classifier.models import *



def load_model(filename):
    """Load in the classifier model"""

    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
    except Exception as error:
        model = None
        print(error)

    return model


# Import the model
MODEL_FILENAME = finders.find("classifier.pkl")
model = load_model(MODEL_FILENAME)


class MessageSerializer(serializers.ModelSerializer):
    """Message serializer class"""

    class Meta:
        model = Message
        fields = "__all__"
        read_only_fields = (
            "related",
            "request",
            "offer",
            "aid_related",
            "medical_help",
            "medical_products",
            "search_and_rescue",
            "security",
            "military",
            "child_alone",
            "water",
            "food",
            "shelter",
            "clothing",
            "money",
            "missing_people",
            "refugees",
            "death",
            "other_aid",
            "infrastructure_related",
            "transport",
            "buildings",
            "electricity",
            "tools",
            "hospitals",
            "shops",
            "aid_centers",
            "other_infrastructure",
            "weather_related",
            "floods",
            "storm",
            "fire",
            "earthquake",
            "cold",
            "other_weather",
            "direct_report",
        )

    def create(self, validated_data):
        """Create a new message object"""

        message = validated_data["message"]
        classes = model.predict([message])[0]
        values = dict(zip(self.Meta.read_only_fields, classes))
        validated_data.update(values)
        
        return super().create(validated_data)



