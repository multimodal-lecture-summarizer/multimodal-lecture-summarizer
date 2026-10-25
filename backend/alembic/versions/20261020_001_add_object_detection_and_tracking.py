"""Add object detection and tracking columns to video_scenes, summaries, and evaluation_logs

Revision ID: 20261020_001
Revises: 20261013_001
Create Date: 2026-10-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '20261020_001'
down_revision: Union[str, None] = '20261013_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Alter video_scenes
    with op.batch_alter_table('video_scenes', schema=None) as batch_op:
        batch_op.add_column(sa.Column('detected_objects_json', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('object_counts_json', sa.JSON(), nullable=True))

    # 2. Alter summaries
    with op.batch_alter_table('summaries', schema=None) as batch_op:
        batch_op.add_column(sa.Column('objects_summary_json', sa.JSON(), nullable=True))

    # 3. Alter evaluation_logs (or create table if not already existing)
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    if 'evaluation_logs' in tables:
        with op.batch_alter_table('evaluation_logs', schema=None) as batch_op:
            batch_op.add_column(sa.Column('od_precision', sa.Float(), nullable=True))
            batch_op.add_column(sa.Column('od_recall', sa.Float(), nullable=True))
            batch_op.add_column(sa.Column('od_map50', sa.Float(), nullable=True))
            batch_op.add_column(sa.Column('od_map50_95', sa.Float(), nullable=True))
    else:
        op.create_table(
            'evaluation_logs',
            sa.Column('eval_id', UUID(as_uuid=True), primary_key=True),
            sa.Column('video_id', UUID(as_uuid=True), sa.ForeignKey('videos.video_id', ondelete='CASCADE'), nullable=True),
            sa.Column('wer', sa.Float(), nullable=True),
            sa.Column('cer', sa.Float(), nullable=True),
            sa.Column('rouge_1', sa.Float(), nullable=True),
            sa.Column('rouge_2', sa.Float(), nullable=True),
            sa.Column('rouge_l', sa.Float(), nullable=True),
            sa.Column('bertscore_f1', sa.Float(), nullable=True),
            sa.Column('od_precision', sa.Float(), nullable=True),
            sa.Column('od_recall', sa.Float(), nullable=True),
            sa.Column('od_map50', sa.Float(), nullable=True),
            sa.Column('od_map50_95', sa.Float(), nullable=True),
            sa.Column('reference_transcript', sa.Text(), nullable=True),
            sa.Column('reference_summary', sa.Text(), nullable=True),
            sa.Column('details_json', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if 'evaluation_logs' in tables:
        columns = [c['name'] for c in inspector.get_columns('evaluation_logs')]
        with op.batch_alter_table('evaluation_logs', schema=None) as batch_op:
            if 'od_map50_95' in columns:
                batch_op.drop_column('od_map50_95')
            if 'od_map50' in columns:
                batch_op.drop_column('od_map50')
            if 'od_recall' in columns:
                batch_op.drop_column('od_recall')
            if 'od_precision' in columns:
                batch_op.drop_column('od_precision')

    with op.batch_alter_table('summaries', schema=None) as batch_op:
        batch_op.drop_column('objects_summary_json')

    with op.batch_alter_table('video_scenes', schema=None) as batch_op:
        batch_op.drop_column('object_counts_json')
        batch_op.drop_column('detected_objects_json')
