import sys
sys.path.insert(0, "backend")
from app.core.database import SessionLocal
from sqlalchemy import text
import json

db = SessionLocal()
cols = db.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'jobs'")).fetchall()
print("Jobs columns:", [c[0] for c in cols])
cols_sum = db.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'summaries'")).fetchall()
print("Summaries columns:", [c[0] for c in cols_sum])

jobs = db.execute(text("SELECT * FROM jobs WHERE video_id = 'ccee3926-267d-428c-8237-2a8c92e35c57'")).fetchall()
for j in jobs:
    print("Job for ccee3926:", dict(j._mapping))
