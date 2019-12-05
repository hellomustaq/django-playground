from django.conf import settings
from django.db import models


class Company(models.Model):
    name = models.CharField(max_length=191, null=False, blank=False)
    serial_number = models.CharField(max_length=100, unique=True, null=False, blank=False)
    trading_name = models.CharField(max_length=100, null=False, blank=False)
    registration_number = models.CharField(max_length=100, unique=True, null=False, blank=False)
    email = models.EmailField(null=False, blank=False)
    phone = models.CharField(max_length=50, null=True, blank=True)
    fax = models.CharField(max_length=50, null=True, blank=True)
    suburb = models.CharField(max_length=100, null=True, blank=True)
    post_code = models.CharField(max_length=50, null=True, blank=True)
    address_line_one = models.CharField(max_length=191, null=True, blank=True)
    address_line_two = models.CharField(max_length=191, null=True, blank=True)
    active = models.BooleanField(default=True)
    website = models.URLField(null=True, blank=None)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    @property
    def is_active(self):
        return self.active

    class Meta:
        db_table = "companies"


class CompanyUser(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    company = models.ForeignKey(
        Company,
        on_delete=models.CASCADE
    )
