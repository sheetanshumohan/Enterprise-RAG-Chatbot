from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from knowledge_assistant.domain.entities import User
from knowledge_assistant.infrastructure.auth.jwt_auth import create_access_token, hash_password, verify_password
from knowledge_assistant.infrastructure.db.repositories import SqlUserRepository
from knowledge_assistant.interfaces.api.dependencies import get_current_user, get_user_repo
from knowledge_assistant.interfaces.api.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, user_repo: SqlUserRepository = Depends(get_user_repo)):
    existing = await user_repo.get_by_email(payload.email)
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = User(email=payload.email, hashed_password=hash_password(payload.password), full_name=payload.full_name)
    await user_repo.create(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, user_repo: SqlUserRepository = Depends(get_user_repo)):
    user = await user_repo.get_by_email(payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    return UserResponse(id=user.id, email=user.email, full_name=user.full_name)
