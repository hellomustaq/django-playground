from django.conf.urls import url
from django.urls import path

from authentication import views
from .views import (
    login_view, post_login, registration_view, registration, logout_user,
    password_forget, password_reset, password_reset_view, password_reset_post)

urlpatterns = [
    path('login/', login_view, name='login'),
    path('login/post/', post_login, name='login_post'),
    path('logout/', logout_user, name='logout'),
    path('register/', registration_view, name='register'),
    path('register/post', registration, name='register_post'),
    url(r'^activate/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$',
        views.activate, name='activate'),

    path('password/forget', password_forget, name='password_forget'),
    path('password/reset', password_reset, name='password_reset'),
    url(r'^password/resett/(?P<token>\w+)/$', views.password_reset_view, name='password_r'),
    path('password/reset/post', password_reset_post, name='password_reset_post'),

]
