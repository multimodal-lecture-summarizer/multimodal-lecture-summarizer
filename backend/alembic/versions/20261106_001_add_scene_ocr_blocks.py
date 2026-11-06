"""Add OCR text and per-line OCR blocks to video_scenes

Revision ID: 20261106_001
Revises: 20261025_001
Create Date: 2026-11-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20261106_001'
down_revision: Union[str, None] = '20261025_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('video_scenes', schema=None) as batch_op:
        batch_op.add_column(sa.Column('ocr_text', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('ocr_blocks_json', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('video_scenes', schema=None) as batch_op:
        batch_op.drop_column('ocr_blocks_json')
        batch_op.drop_column('ocr_text')
