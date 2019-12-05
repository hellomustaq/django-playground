from django.db import models


# Create your models here.
class Plan(models.Model):
    type = (
        ('month', 'Month'),
        ('year', 'Year')
    )
    title = models.CharField(max_length=191)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    type = models.CharField(max_length=20, choices=type, default='month')
    description = models.TextField(null=True, blank=True)
    active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
