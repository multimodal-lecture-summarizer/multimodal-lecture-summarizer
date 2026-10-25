"""Add summary_length and summary_variants_json to summaries table

Revision ID: 20261025_001
Revises: 20261020_001
Create Date: 2026-10-25 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20261025_001'
down_revision: Union[str, None] = '20261020_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Alter summaries table using batch mode for SQLite and Postgres compatibility
    with op.batch_alter_table('summaries', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('summary_length', sa.String(length=20), nullable=True, server_default='medium')
        )
        batch_op.add_column(
            sa.Column('summary_variants_json', sa.JSON(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('summaries', schema=None) as batch_op:
        batch_op.drop_column('summary_variants_json')
        batch_op.drop_column('summary_length')
