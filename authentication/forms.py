from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django import forms
from django.contrib import messages

from .models import User


class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm):
        model = User
        fields = "__all__"


class CustomUserChangeForm(UserChangeForm):
    class Meta:
        model = User
        fields = "__all__"


class UserCreation(forms.Form):
    model = User
    fields = "__all__"

    def clean_name(self):
        name = self.cleaned_data['name']
        lname = self.cleaned_data['lName']
        if not lname == 'Ami':
            raise forms.ValidationError('This name less then 10 character')
        return self.cleaned_data
