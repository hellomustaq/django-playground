from django.contrib import admin

from company.models import Company, CompanyUser


class CompanyAdmin(admin.ModelAdmin):
    pass


admin.site.register(Company, CompanyAdmin)
admin.site.register(CompanyUser, CompanyAdmin)
