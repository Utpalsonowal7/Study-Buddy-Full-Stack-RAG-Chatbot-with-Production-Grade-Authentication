from fastapi import Request,Response,HTTPException,BackgroundTasks
from app.core.redis import redis
from app.utils.email_templates import send_otp_email as send_email

from app.utils.keys import otp_key
from app.utils.otp import generate_otp
from app.utils.api_response import success_response

async def send_otp(email:str, background:BackgroundTasks):
     key = otp_key(email)

     is_otp_sent = await redis.get(key)
     if is_otp_sent:
          raise HTTPException(status_code=400, detail="OTP already sent. Please wait before requesting again.")

     otp = generate_otp()
     await redis.set(key, otp, ex=300)  
     background.add_task(send_email, email, otp)  

     return success_response("OTP sent successfully. Please check your email.", {"email": email})
     