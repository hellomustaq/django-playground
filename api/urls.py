from django.urls import include, path
from rest_framework import routers
from .views import CompanyViewSets, PlanViewSets

router = routers.DefaultRouter()
router.register(r'companies', CompanyViewSets)
router.register(r'plans', PlanViewSets)

urlpatterns = [
    path('', include(router.urls)),
]
