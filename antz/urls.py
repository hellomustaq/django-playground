from django.contrib import admin
from django.urls import path, include

# from api.chatbot import chat

admin.site.site_header = "Antz"
admin.site.site_title = "Antz Admin Portal"
admin.site.index_title = "Welcome to Antz"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('authentication.urls')),
    path('', include('profiles.urls')),
    # url(r'^chatbot/', chat)

    path('api/', include('api.urls'), name='api'),
]
