
from fastapi import Request


def get_client_ip(request: Request) -> str:
    
    cf_ip = request.headers.get("CF-Connecting-IP")

    if cf_ip:
        return cf_ip

   
    return request.client.host if request.client else "unknown"
