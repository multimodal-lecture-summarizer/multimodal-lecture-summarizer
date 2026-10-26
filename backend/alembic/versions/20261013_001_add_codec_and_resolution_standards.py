"""Add codec and resolution standards to video_standards and video_metadata

Revision ID: 20261013_001
Revises: 
Create Date: 2026-10-13 22:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20261013_001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Alter video_standards
    with op.batch_alter_table('video_standards', schema=None) as batch_op:
        batch_op.add_column(sa.Column('allowed_codecs', sa.JSON(), server_default='["h264", "hevc"]', nullable=False))
        batch_op.add_column(sa.Column('min_width', sa.Integer(), server_default='640', nullable=False))
        batch_op.add_column(sa.Column('min_height', sa.Integer(), server_default='360', nullable=False))
        batch_op.add_column(sa.Column('max_width', sa.Integer(), server_default='1920', nullable=False))
        batch_op.add_column(sa.Column('max_height', sa.Integer(), server_default='1080', nullable=False))

    # 2. Alter video_metadata
    with op.batch_alter_table('video_metadata', schema=None) as batch_op:
        batch_op.add_column(sa.Column('video_codec', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('audio_codec', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('bitrate', sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('video_metadata', schema=None) as batch_op:
        batch_op.drop_column('bitrate')
        batch_op.drop_column('audio_codec')
        batch_op.drop_column('video_codec')

    with op.batch_alter_table('video_standards', schema=None) as batch_op:
        batch_op.drop_column('max_height')
        batch_op.drop_column('max_width')
        batch_op.drop_column('min_height')
        batch_op.drop_column('min_width')
        batch_op.drop_column('allowed_codecs')
