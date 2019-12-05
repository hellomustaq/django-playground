from django.conf.urls import url
from django.contrib import admin

from home.views import home
from taleant import taleantBotApi
# from teleaus import teleausBotApi

urlpatterns = [
    url(r'^admin/', admin.site.urls),

    url(r'^taleantbot/', taleantBotApi.chat),
    url(r'^taleant/login/', taleantBotApi.login),
    # url(r'^teleausbot/', teleausBotApi.chat),
    # url(r'^teleaus/login/', teleausBotApi.login),

    url('', home, name='home'),
]
