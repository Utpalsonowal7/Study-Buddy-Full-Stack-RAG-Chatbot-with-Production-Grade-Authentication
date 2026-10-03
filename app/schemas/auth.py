from fastapi import BaseModel, EmailStr

class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

class registerUserRequest(BaseModel):
    email: EmailStr
    full_name: str