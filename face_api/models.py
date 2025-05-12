from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100, unique=True)
    image = models.ImageField(upload_to='faces/')

    def __str__(self):
        return self.name