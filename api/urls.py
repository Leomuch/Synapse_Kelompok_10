from django.urls import path
from .views import predict

urlpatterns = [
    path('predict', predict, name='predict')
    # path('predict-image', predict_image, name='predict_image')
]