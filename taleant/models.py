from django.db import models


# Create your models here.

class QueryPatterns(models.Model):
    query = models.CharField(max_length=10000)

    def __str__(self):
        return self.query
