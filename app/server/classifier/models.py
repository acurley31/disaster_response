from django.db import models


class Message(models.Model):
    message = models.TextField()
    related = models.BooleanField(default=False)
    request = models.BooleanField(default=False)
    offer = models.BooleanField(default=False)
    aid_related = models.BooleanField(default=False)
    medical_help = models.BooleanField(default=False)
    medical_products = models.BooleanField(default=False)
    search_and_rescue = models.BooleanField(default=False)
    security = models.BooleanField(default=False)
    military = models.BooleanField(default=False)
    child_alone = models.BooleanField(default=False)
    water = models.BooleanField(default=False)
    food = models.BooleanField(default=False)
    shelter = models.BooleanField(default=False)
    clothing = models.BooleanField(default=False)
    money = models.BooleanField(default=False)
    missing_people = models.BooleanField(default=False)
    refugees = models.BooleanField(default=False)
    death = models.BooleanField(default=False)
    other_aid = models.BooleanField(default=False)
    infrastructure_related = models.BooleanField(default=False)
    transport = models.BooleanField(default=False)
    buildings = models.BooleanField(default=False)
    electricity = models.BooleanField(default=False)
    tools = models.BooleanField(default=False)
    hospitals = models.BooleanField(default=False)
    shops = models.BooleanField(default=False)
    aid_centers = models.BooleanField(default=False)
    other_infrastructure = models.BooleanField(default=False)
    weather_related = models.BooleanField(default=False)
    floods = models.BooleanField(default=False)
    storm = models.BooleanField(default=False)
    fire = models.BooleanField(default=False)
    earthquake = models.BooleanField(default=False)
    cold = models.BooleanField(default=False)
    other_weather = models.BooleanField(default=False)
    direct_report = models.BooleanField(default=False)

    def __str__(self):
        """Represent the object as a string"""

        return self.message

