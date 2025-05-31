from django.urls import path
import views

urlpatterns = [
    path('',views.index,name='index'),
    path('api/receive/',views.receive_data,name='receive'),
]
