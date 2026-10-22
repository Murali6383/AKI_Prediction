from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from .core.config import FRONTEND_ORIGINS
app=FastAPI(title="Explainable and Dynamic AKI Risk Prediction",version="1.0.0",description="Research-only prototype; not for clinical decisions.")
app.add_middleware(CORSMiddleware,allow_origins=FRONTEND_ORIGINS,allow_credentials=False,allow_methods=["GET","POST","OPTIONS"],allow_headers=["Content-Type","Authorization"])
app.include_router(router)
@app.get("/")
def root():return {"project":app.title,"docs":"/docs","research_only":True}
