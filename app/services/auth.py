from fastapi import Request,Response,HTTPException,BackgroundTasks
from app.core.redis import redis

from app.utils.keys import otp_key

async def send_otp(email:str, background:BackgroundTasks):
     