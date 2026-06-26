from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="home"),
			path("index.html", views.index, name="index"),
			path("Admin.html", views.Admin, name="Admin"),	      
			path("AdminLogin", views.AdminLogin, name="AdminLogin"),	
			path("UploadDataset.html", views.UploadDataset, name="UploadDataset"),
			path("UploadDatasetAction", views.UploadDatasetAction, name="UploadDatasetAction"),
			path("ClusterPrediction.html", views.ClusterPrediction, name="ClusterPrediction"),
			path("ClusterPredictionAction", views.ClusterPredictionAction, name="ClusterPredictionAction"),
			path("FuturePrediction.html", views.FuturePrediction, name="FuturePrediction"),
			path("FuturePredictionAction", views.FuturePredictionAction, name="FuturePredictionAction"),
			path("Analysis.html", views.Analysis, name="Analysis"),
			path("AnalysisAction", views.AnalysisAction, name="AnalysisAction"),
]