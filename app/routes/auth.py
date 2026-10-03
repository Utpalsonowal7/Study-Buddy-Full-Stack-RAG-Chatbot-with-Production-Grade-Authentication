from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Request,
    Response,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import get_current_user
from app.dependencies.rate_limit import rate_limit_dependency
from app.dependencies.session import get_session as get_db
from app.models.auth.user import User

from app.schemas.auth import (
    OTPRequest,
    OTPVerifyRequest,
    registerUserRequest,
)

from app.services.auth import (
    login_user,
    register_user,
    send_otp,
    verify_login_otp,
    verify_otp,
    google_login_redirect,
    google_callback,
    github_callback,
    github_login_redirect,
    refresh_access_token,
    logout_user,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)



@router.post(
    "/register",
    dependencies=[Depends(rate_limit_dependency(5, 900))],
)
async def register(
    request: registerUserRequest,
    req: Request,
    res: Response,
    db: AsyncSession = Depends(get_db),
):
    return await register_user(
        request=request,
        db=db,
        req=req,
        res=res,
    )




@router.post(
    "/send-otp",
    dependencies=[Depends(rate_limit_dependency(3, 900))],
)
async def send_otp_route(
    request: OTPRequest,
    background: BackgroundTasks,
):
    return await send_otp(
        email=request.email,
        background=background,
    )


@router.post(
    "/verify-otp",
    dependencies=[Depends(rate_limit_dependency(5, 900))],
)
async def verify_otp_route(
    request: OTPVerifyRequest,
):
    return await verify_otp(
        email=request.email,
        otp=request.otp,
    )



@router.post(
    "/login",
    dependencies=[Depends(rate_limit_dependency(5, 900))],
)
async def login(
    request: OTPRequest,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    return await login_user(
        request=request,
        db=db,
        background=background,
    )


@router.post(
    "/login/verify-otp",
    dependencies=[Depends(rate_limit_dependency(5, 900))],
)
async def verify_login(
    request: OTPVerifyRequest,
    req: Request,
    res: Response,
    db: AsyncSession = Depends(get_db),
):
    return await verify_login_otp(
        request=request,
        db=db,
        req=req,
        res=res,
    )




@router.get(
    "/google",
    dependencies=[Depends(rate_limit_dependency(10, 900))],
)
async def google_login():
    return await google_login_redirect()


@router.get(
    "/google/callback",
    dependencies=[Depends(rate_limit_dependency(10, 900))],
)
async def google_auth_callback(
    code: str,
    state: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await google_callback(
        code,
        state,
        request,
        db,
    )



@router.get(
    "/github",
    dependencies=[Depends(rate_limit_dependency(10, 900))],
)
async def github_login():
    return await github_login_redirect()


@router.get(
    "/github/callback",
    dependencies=[Depends(rate_limit_dependency(10, 900))],
)
async def github_auth_callback(
    code: str,
    state: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await github_callback(
        code,
        state,
        request,
        db,
    )



@router.post(
    "/refresh",
    dependencies=[Depends(rate_limit_dependency(20, 900))],
)
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    return await refresh_access_token(
        request,
        response,
        db,
    )



@router.get("/me")
async def get_me(
    current_user: User = Depends(get_current_user),
):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "avatar": (current_user.avatar or "https://placehold.net/avatar-2.svg"),
        "is_verified": current_user.isEmailVerified,
    }




@router.post("/logout")
async def logout(
    response: Response,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    return await logout_user(
        req,
        response,
        db,
    )
