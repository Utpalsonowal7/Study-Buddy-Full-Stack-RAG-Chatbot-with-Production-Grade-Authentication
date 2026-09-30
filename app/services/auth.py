from fastapi import Request,Response,HTTPException,BackgroundTasks
from app.core.redis import redis


async def send_otp(email:str, background:BackgroundTasks):
     