"""Initial migration

Revision ID: 0a0a0a0a0a0a
Revises: 
Create Date: 2026-04-28 09:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0a0a0a0a0a0a'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enums are automatically created by sa.Enum during table creation.

    # Create Patients table
    op.create_table('patients',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_code', sa.String(length=50), nullable=False),
        sa.Column('age', sa.Integer(), nullable=True),
        sa.Column('gender', sa.String(length=10), nullable=True),
        sa.Column('demographics', sa.JSON(), nullable=True),
        sa.Column('medical_history', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('patient_code', name=op.f('uq_patients_patient_code'))
    )

    # Create Claims table
    op.create_table('claims',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('claim_text', sa.Text(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'VERIFIED', 'REFUTED', 'UNCERTAIN', 'FAILED', name='claimstatus'), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('extracted_entities', sa.JSON(), nullable=True),
        sa.Column('source_citations', sa.JSON(), nullable=True),
        sa.Column('source_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('verdict', sa.String(length=20), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('shap_values', sa.JSON(), nullable=True),
        sa.Column('disclaimer', sa.Text(), nullable=True),
        sa.Column('uncertainty_flag', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('hallucination_detected', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('hallucination_details', sa.JSON(), nullable=True),
        sa.Column('celery_task_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], name=op.f('fk_claims_patient_id_patients')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create Reports table
    op.create_table('reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('report_type', sa.Enum('LAB', 'CLINICAL', 'DISCHARGE', name='reporttype'), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('file_format', sa.String(length=10), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETE', 'FAILED', name='analysisstatus'), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=True),
        sa.Column('parsed_data', sa.JSON(), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=True),
        sa.Column('risk_factors', sa.JSON(), nullable=True),
        sa.Column('shap_values', sa.JSON(), nullable=True),
        sa.Column('anomalies', sa.JSON(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('retrieved_sources', sa.JSON(), nullable=True),
        sa.Column('analysis_result', sa.JSON(), nullable=True),
        sa.Column('source_citations', sa.JSON(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('uncertainty_flag', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('medical_disclaimer', sa.Text(), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], name=op.f('fk_reports_patient_id_patients')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create Image Analyses table
    op.create_table('image_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('image_type', sa.Enum('XRAY', 'CT', 'MRI', 'SKIN', 'PATHOLOGY', name='imagetype'), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=True),
        sa.Column('file_format', sa.String(length=10), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETE', 'FAILED', name='analysisstatus'), nullable=False),
        sa.Column('dicom_metadata', sa.JSON(), nullable=True),
        sa.Column('segmentation_mask_path', sa.String(length=500), nullable=True),
        sa.Column('segmentation_overlay', sa.String(length=500), nullable=True),
        sa.Column('segmentation_confidence', sa.Float(), nullable=True),
        sa.Column('gradcam_path', sa.String(length=500), nullable=True),
        sa.Column('gradcam_regions', sa.JSON(), nullable=True),
        sa.Column('classification', sa.JSON(), nullable=True),
        sa.Column('classification_probs', sa.JSON(), nullable=True),
        sa.Column('roi_metadata', sa.JSON(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('source_citations', sa.JSON(), nullable=True),
        sa.Column('retrieved_sources', sa.JSON(), nullable=True),
        sa.Column('uncertainty_flag', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('anomaly_detected', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('medical_disclaimer', sa.Text(), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], name=op.f('fk_image_analyses_patient_id_patients')),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('image_analyses')
    op.drop_table('reports')
    op.drop_table('claims')
    op.drop_table('patients')
    
    # Drop Enums
    postgresql.ENUM(name='imagetype').drop(op.get_bind())
    postgresql.ENUM(name='analysisstatus').drop(op.get_bind())
    postgresql.ENUM(name='reporttype').drop(op.get_bind())
    postgresql.ENUM(name='claimstatus').drop(op.get_bind())
