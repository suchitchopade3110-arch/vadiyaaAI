import sqlalchemy as sa
engine = sa.create_engine("postgresql+psycopg2://vaidya:vaidya@localhost/vaidyaai")
with engine.connect() as conn:
    conn.execute(sa.text("ALTER TABLE claims ADD COLUMN IF NOT EXISTS retrieved_sources JSONB DEFAULT '[]';"))
    conn.commit()
print("Column 'retrieved_sources' added successfully.")
