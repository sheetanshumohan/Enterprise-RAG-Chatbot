from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from knowledge_assistant.domain.entities import Collection, User
from knowledge_assistant.infrastructure.db.repositories import SqlCollectionRepository
from knowledge_assistant.interfaces.api.dependencies import get_collection_repo, get_current_user
from knowledge_assistant.interfaces.api.schemas import CollectionCreateRequest, CollectionResponse

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    payload: CollectionCreateRequest,
    user: User = Depends(get_current_user),
    repo: SqlCollectionRepository = Depends(get_collection_repo),
):
    collection = Collection(user_id=user.id, name=payload.name, description=payload.description)
    await repo.create(collection)
    return CollectionResponse(**collection.__dict__)


@router.get("", response_model=list[CollectionResponse])
async def list_collections(
    user: User = Depends(get_current_user), repo: SqlCollectionRepository = Depends(get_collection_repo)
):
    collections = await repo.list_for_user(user.id)
    return [CollectionResponse(**c.__dict__) for c in collections]


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: str,
    user: User = Depends(get_current_user),
    repo: SqlCollectionRepository = Depends(get_collection_repo),
):
    existing = await repo.get(collection_id, user.id)
    if not existing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Collection not found")
    await repo.delete(collection_id, user.id)
