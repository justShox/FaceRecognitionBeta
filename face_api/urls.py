from django.urls import path
from . import views

urlpatterns = [
    path('cameratest/', views.CameraTestView.as_view(), name='cameratest'),
    path('compare/', views.CompareImageView.as_view(), name='compare'),
]