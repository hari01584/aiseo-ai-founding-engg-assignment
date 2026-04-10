from fastapi import FastAPI
from app.api import aeo, fanout

app = FastAPI(title="AEGIS — AI Engineer Assignment")

app.include_router(aeo.router, prefix="/api/aeo", tags=["aeo"])
app.include_router(fanout.router, prefix="/api/fanout", tags=["fanout"])

@app.get("/")
async def root():
    return {"message": "Welcome to the AEGIS AI Engineer Assignment API"}
