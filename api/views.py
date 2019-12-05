from rest_framework import viewsets, permissions

from company.models import Company
from company.serializers import CompanySerializer
from subscription.models import Plan
from subscription.serializers import PlanSerializer


class CompanyViewSets(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    permission_classes = [
        permissions.AllowAny
    ]
    serializer_class = CompanySerializer


class PlanViewSets(viewsets.ModelViewSet):
    queryset = Plan.objects.all()
    permission_classes = [
        permissions.AllowAny
    ]
    serializer_class = PlanSerializer
