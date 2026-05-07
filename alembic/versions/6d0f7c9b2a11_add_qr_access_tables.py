"""add qr access tables

Revision ID: 6d0f7c9b2a11
Revises: 5c951e5afb00
Create Date: 2026-05-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "6d0f7c9b2a11"
down_revision = "5c951e5afb00"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "qr_tokens",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("report_id", sa.String(length=64), nullable=False),
        sa.Column("patient_id", sa.String(length=64), nullable=True),
        sa.Column("token_hash", sa.String(length=16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scan_count", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_qr_tokens")),
    )
    op.create_index(op.f("ix_qr_tokens_expires_at"), "qr_tokens", ["expires_at"], unique=False)
    op.create_index(op.f("ix_qr_tokens_is_active"), "qr_tokens", ["is_active"], unique=False)
    op.create_index(op.f("ix_qr_tokens_patient_id"), "qr_tokens", ["patient_id"], unique=False)
    op.create_index(op.f("ix_qr_tokens_report_id"), "qr_tokens", ["report_id"], unique=False)

    op.create_table(
        "qr_audit_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("token_id", sa.String(length=36), nullable=True),
        sa.Column("report_id", sa.String(length=64), nullable=True),
        sa.Column("patient_id", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_qr_audit_logs")),
    )
    op.create_index(op.f("ix_qr_audit_logs_action"), "qr_audit_logs", ["action"], unique=False)
    op.create_index(op.f("ix_qr_audit_logs_patient_id"), "qr_audit_logs", ["patient_id"], unique=False)
    op.create_index(op.f("ix_qr_audit_logs_report_id"), "qr_audit_logs", ["report_id"], unique=False)
    op.create_index(op.f("ix_qr_audit_logs_timestamp"), "qr_audit_logs", ["timestamp"], unique=False)
    op.create_index(op.f("ix_qr_audit_logs_token_id"), "qr_audit_logs", ["token_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_qr_audit_logs_token_id"), table_name="qr_audit_logs")
    op.drop_index(op.f("ix_qr_audit_logs_timestamp"), table_name="qr_audit_logs")
    op.drop_index(op.f("ix_qr_audit_logs_report_id"), table_name="qr_audit_logs")
    op.drop_index(op.f("ix_qr_audit_logs_patient_id"), table_name="qr_audit_logs")
    op.drop_index(op.f("ix_qr_audit_logs_action"), table_name="qr_audit_logs")
    op.drop_table("qr_audit_logs")

    op.drop_index(op.f("ix_qr_tokens_report_id"), table_name="qr_tokens")
    op.drop_index(op.f("ix_qr_tokens_patient_id"), table_name="qr_tokens")
    op.drop_index(op.f("ix_qr_tokens_is_active"), table_name="qr_tokens")
    op.drop_index(op.f("ix_qr_tokens_expires_at"), table_name="qr_tokens")
    op.drop_table("qr_tokens")
