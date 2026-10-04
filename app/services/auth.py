from datetime import datetime, timedelta, timezone
import secrets
import hashlib
import jwt as pyjwt
from urllib.parse import urlencode

from fastapi import BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import redis
from app.models.auth.session import Session
from app.models.auth.user import User
from app.schemas.auth import OTPRequest, OTPVerifyRequest, registerUserRequest
from app.utils.api_response import success_response
from app.utils.cookie_options import (
    ACCESS_TOKEN_COOKIE_OPTIONS,
    REFRESH_TOKEN_COOKIE_OPTIONS,
)
from app.utils.email_templates import send_otp_email as send_email
from app.utils.jwt import create_auth_tokens, decode_refresh_token
from app.utils.keys import otp_key
from app.utils.otp import generate_otp


from app.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    GOOGLE_AUTH_URI,
    GOOGLE_TOKEN_URI,
    GOOGLE_PROVIDER_URI,
    GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET,
    GITHUB_REDIRECT_URI,
    GITHUB_AUTH_URI,
    GITHUB_TOKEN_URI,
    GITHUB_USER_URI,
    GITHUB_USER_EMAILS_URI,
    FRONT_END_URL,
)


async def _set_auth_cookies(
    res: Response,
    req: Request,
    user_id: int,
    db: AsyncSession,
):
    tokens = create_auth_tokens({"sub": str(user_id)})

    session = Session(
        user_id=user_id,
        refresh_token_hash=hashlib.sha256(tokens["refresh_token"].encode()).hexdigest(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        user_agent=req.headers.get("user-agent"),
        ip_address=req.client.host if req.client else None,
    )

    db.add(session)

    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    await db.refresh(session)

    res.set_cookie(
        key="access_token",
        value=tokens["access_token"],
        **ACCESS_TOKEN_COOKIE_OPTIONS,
    )

    res.set_cookie(
        key="refresh_token",
        value=tokens["refresh_token"],
        **REFRESH_TOKEN_COOKIE_OPTIONS,
    )


async def send_otp(
    email: str,
    background: BackgroundTasks,
):
    key = otp_key(email)

    if await redis.get(key):
        raise HTTPException(
            status_code=400,
            detail="OTP already sent. Please wait before requesting again.",
        )

    otp = generate_otp()

    await redis.set(
        key,
        otp,
        ex=300,
    )

    background.add_task(
        send_email,
        email,
        otp,
    )

    return success_response(
        "OTP sent successfully. Please check your email.",
        {"email": email},
    )


async def verify_otp(
    email: str,
    otp: str,
):
    key = otp_key(email)

    stored_otp = await redis.get(key)

    if not stored_otp:
        raise HTTPException(
            status_code=400,
            detail="OTP expired or not found. Please request a new one.",
        )

    if stored_otp != otp:
        raise HTTPException(
            status_code=400,
            detail="Invalid OTP. Please try again.",
        )

    await redis.delete(key)

    return success_response(
        "OTP verified successfully.",
        {"email": email},
    )


async def register_user(
    request: registerUserRequest,
    db: AsyncSession,
    req: Request,
    res: Response,
):
    # Validate signing configuration before creating a user in the database.
    create_auth_tokens({"sub": "configuration-check"})
    result = await db.execute(select(User).where(User.email == request.email))

    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered. Please use a different email.",
        )

    user = User(
        email=request.email, name=request.full_name, is_email_verified=False
    )

    db.add(user)

    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="An error occurred while registering the user.",
        )

    await db.refresh(user)

    await _set_auth_cookies(
        res,
        req,
        user.id,
        db,
    )

    return success_response(
        "User registered successfully.",
        {
            "email": user.email,
            "full_name": user.name,
        },
    )


async def login_user(
    request: OTPRequest,
    db: AsyncSession,
    background: BackgroundTasks,
):
    result = await db.execute(select(User).where(User.email == request.email))

    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=400,
            detail="User not found. Please register first.",
        )

    return await send_otp(
        user.email,
        background,
    )


async def verify_login_otp(
    request: OTPVerifyRequest,
    db: AsyncSession,
    req: Request,
    res: Response,
):
    result = await db.execute(select(User).where(User.email == request.email))

    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=400,
            detail="User not found. Please register first.",
        )

    await verify_otp(
        request.email,
        request.otp,
    )
    user.is_email_verified = True

    await _set_auth_cookies(
        res,
        req,
        user.id,
        db,
    )

    return success_response(
        "User logged in successfully.",
        {"email": user.email},
    )


async def google_login_redirect() -> RedirectResponse:
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }

    url = f"{GOOGLE_AUTH_URI}?{urlencode(params)}"

    return RedirectResponse(
        url=url,
        status_code=302,
    )


async def google_callback(
    code: str,
    request: Request,
    db: AsyncSession,
) -> RedirectResponse:

    async with httpx.AsyncClient() as client:

        token_resp = await client.post(
            GOOGLE_TOKEN_URI,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )

        if token_resp.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail="Google token exchange failed",
            )

        token_data = token_resp.json()

        google_access_token = token_data.get("access_token")

        if not google_access_token:
            raise HTTPException(
                status_code=400,
                detail="Google access token missing",
            )

        userinfo_resp = await client.get(
            GOOGLE_PROVIDER_URI,
            headers={
                "Authorization": f"Bearer {google_access_token}",
            },
        )

    if userinfo_resp.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail="Failed to get Google user",
        )

    profile = userinfo_resp.json()

    google_id = profile.get("id")
    email = profile.get("email")
    name = profile.get("name")
    avatar = profile.get("picture")
    email_verified = profile.get("verified_email", False)

    if not google_id:
        raise HTTPException(
            status_code=400,
            detail="Google user ID missing",
        )

    if not email or not email_verified:
        raise HTTPException(
            status_code=400,
            detail="Google email is not verified",
        )

    result = await db.execute(select(User).where(User.email == email))

    user = result.scalar_one_or_none()

    if user:

        if user.provider == "EMAIL":
            raise HTTPException(
                status_code=400,
                detail="Account already exists with email/password",
            )

        if user.provider == "GOOGLE" and user.provider_id != google_id:
            raise HTTPException(
                status_code=400,
                detail="Google account does not match",
            )

    else:
        user = User(
            name=name,
            email=email,
            password=None,
            is_email_verified=True,
            provider="GOOGLE",
            provider_id=google_id,
            avatar=avatar,
        )

        db.add(user)

        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=400,
                detail="Account creation failed",
            )

        await db.refresh(user)

    redirect_response = RedirectResponse(
        url=f"{FRONT_END_URL.rstrip('/')}/home",
        status_code=302,
    )

    await _set_auth_cookies(
        redirect_response,
        request,
        user.id,
        db,
    )

    return redirect_response


async def github_login_redirect() -> RedirectResponse:
    state = secrets.token_urlsafe(32)

    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": "read:user user:email",
        "state": state,
    }
    query = urlencode(params)
    url = f"{GITHUB_AUTH_URI}?{query}"

    redirect_response = RedirectResponse(url=url)
    redirect_response.set_cookie(
        key="oauth_state",
        value=state,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=600,
    )
    return redirect_response


async def github_callback(
    code: str,
    state: str,
    request: Request,
    db: AsyncSession,
) -> RedirectResponse:
    cookie_state = request.cookies.get("oauth_state")
    if not cookie_state or cookie_state != state:
        return RedirectResponse(url=f"{FRONT_END_URL}invalid_state")

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URI,
            headers={"Accept": "application/json"},
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI,
            },
        )

    if token_resp.status_code != 200:
        return RedirectResponse(url=f"{FRONT_END_URL}token_exchange_failed")

    token_data = token_resp.json()
    github_access_token = token_data.get("access_token")

    if not github_access_token:
        return RedirectResponse(url=f"{FRONT_END_URL}token_exchange_failed")

    auth_headers = {
        "Authorization": f"Bearer {github_access_token}",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(GITHUB_USER_URI, headers=auth_headers)

    if user_resp.status_code != 200:
        return RedirectResponse(url=f"{FRONT_END_URL}userinfo_failed")

    profile = user_resp.json()
    github_id = str(profile["id"])
    name = profile.get("name") or profile.get("login")
    avatar = profile.get("avatar_url")
    email = profile.get("email")

    if not email:
        async with httpx.AsyncClient() as client:
            emails_resp = await client.get(GITHUB_USER_EMAILS_URI, headers=auth_headers)

        if emails_resp.status_code == 200:
            emails = emails_resp.json()
            primary = next(
                (e for e in emails if e.get("primary") and e.get("verified")), None
            )
            if not primary:
                primary = next((e for e in emails if e.get("verified")), None)
            if primary:
                email = primary.get("email")

    if not email:
        return RedirectResponse(url=f"{FRONT_END_URL}email_not_verified")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user:
        if user.provider == "EMAIL":
            return RedirectResponse(
                url=f"{FRONT_END_URL}account_exists_use_email_login"
            )
    else:
        user = User(
            name=name,
            email=email,
            password=None,
            is_email_verified=True,
            provider="GITHUB",
            provider_id=github_id,
            avatar=avatar,
        )
        db.add(user)

        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            return RedirectResponse(url=f"{FRONT_END_URL}account_conflict")

        await db.refresh(user)

    redirect_response = RedirectResponse(url=f"{FRONT_END_URL}dashboard")
    await _set_auth_cookies(redirect_response, request, user.id, db)

    return redirect_response


async def refresh_access_token(request: Request, response: Response, db: AsyncSession):
    refresh_token = request.cookies.get("refresh_token")

    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token provided")

    try:
        payload = decode_refresh_token(refresh_token)
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401, detail="Refresh token expired, please log in again"
        )
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    result = await db.execute(
        select(Session).where(Session.refresh_token_hash == hashlib.sha256(refresh_token.encode()).hexdigest()).with_for_update()
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=401, detail="Session not found, please log in again"
        )

    if str(session.user_id) != str(user_id):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    if session.expires_at < datetime.now(timezone.utc):
        await db.delete(session)
        await db.commit()
        raise HTTPException(
            status_code=401, detail="Session expired, please log in again"
        )

    tokens = create_auth_tokens({"sub": str(user_id)})

    session.refresh_token_hash = hashlib.sha256(tokens["refresh_token"].encode()).hexdigest()
    session.expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    await db.commit()

    response.set_cookie(
        key="access_token",
        value=tokens["access_token"],
        **ACCESS_TOKEN_COOKIE_OPTIONS,
    )
    response.set_cookie(
        key="refresh_token",
        value=tokens["refresh_token"],
        **REFRESH_TOKEN_COOKIE_OPTIONS,
    )

    return success_response(message="Access token refreshed.")


async def logout_user(request: Request, response: Response, db: AsyncSession):
    refresh_token = request.cookies.get("refresh_token")

    if refresh_token:
        result = await db.execute(
            select(Session).where(Session.refresh_token_hash == hashlib.sha256(refresh_token.encode()).hexdigest())
        )
        session = result.scalar_one_or_none()
        if session:
            await db.delete(session)
            await db.commit()

    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")

    return success_response(message="Logged out successfully.")
