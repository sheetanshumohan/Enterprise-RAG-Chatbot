"""Initial database schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-08-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    op.create_table(
        'collections',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_collections_user_id', 'collections', ['user_id'])

    op.create_table(
        'documents',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('collection_id', sa.String(length=36), sa.ForeignKey('collections.id'), nullable=False),
        sa.Column('filename', sa.String(length=512), nullable=False),
        sa.Column('doc_type', sa.String(length=32), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('version', sa.Integer(), default=1),
        sa.Column('tags', sa.JSON(), default=list),
        sa.Column('doc_metadata', sa.JSON(), default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_documents_user_id', 'documents', ['user_id'])
    op.create_index('ix_documents_collection_id', 'documents', ['collection_id'])
    op.create_index('ix_documents_content_hash', 'documents', ['content_hash'])

    op.create_table(
        'chunks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('document_id', sa.String(length=36), sa.ForeignKey('documents.id'), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('collection_id', sa.String(length=36), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('level', sa.String(length=16), nullable=False),
        sa.Column('parent_id', sa.String(length=36), nullable=True),
        sa.Column('position', sa.Integer(), default=0),
        sa.Column('token_count', sa.Integer(), default=0),
        sa.Column('chunk_metadata', sa.JSON(), default=dict),
    )
    op.create_index('ix_chunks_document_id', 'chunks', ['document_id'])
    op.create_index('ix_chunks_user_id', 'chunks', ['user_id'])
    op.create_index('ix_chunks_collection_id', 'chunks', ['collection_id'])
    op.create_index('ix_chunks_parent_id', 'chunks', ['parent_id'])

    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('collection_id', sa.String(length=36), nullable=True),
        sa.Column('title', sa.String(length=255), default='New conversation'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_chat_sessions_user_id', 'chat_sessions', ['user_id'])

    op.create_table(
        'messages',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('session_id', sa.String(length=36), sa.ForeignKey('chat_sessions.id'), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('citations', sa.JSON(), default=list),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('reasoning_summary', sa.Text(), nullable=True),
        sa.Column('suggested_followups', sa.JSON(), default=list),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_messages_session_id', 'messages', ['session_id'])

    op.create_table(
        'retrieval_logs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('rewritten_queries', sa.JSON(), default=list),
        sa.Column('retriever_used', sa.String(length=64), nullable=False),
        sa.Column('iterations', sa.Integer(), default=1),
        sa.Column('chunks_retrieved', sa.Integer(), default=0),
        sa.Column('retrieval_latency_ms', sa.Float(), default=0.0),
        sa.Column('llm_latency_ms', sa.Float(), default=0.0),
        sa.Column('embedding_latency_ms', sa.Float(), default=0.0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_retrieval_logs_user_id', 'retrieval_logs', ['user_id'])

    op.create_table(
        'feedback',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('message_id', sa.String(length=36), sa.ForeignKey('messages.id'), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('comment', sa.Text(), default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_feedback_message_id', 'feedback', ['message_id'])
    op.create_index('ix_feedback_user_id', 'feedback', ['user_id'])

    op.create_table(
        'evaluation_results',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('precision', sa.Float(), default=0.0),
        sa.Column('recall', sa.Float(), default=0.0),
        sa.Column('groundedness', sa.Float(), default=0.0),
        sa.Column('context_precision', sa.Float(), default=0.0),
        sa.Column('answer_relevance', sa.Float(), default=0.0),
        sa.Column('hallucination_rate', sa.Float(), default=0.0),
        sa.Column('latency_ms', sa.Float(), default=0.0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('evaluation_results')
    op.drop_table('feedback')
    op.drop_table('retrieval_logs')
    op.drop_table('messages')
    op.drop_table('chat_sessions')
    op.drop_table('chunks')
    op.drop_table('documents')
    op.drop_table('collections')
    op.drop_table('users')
