"""Add RAG metadata to existing tables without dropping or replacing data.

Run with: uv run --frozen python -m app.db.migrate_rag
"""
import asyncio

from sqlalchemy import text

from app.db.database import engine


async def migrate(connection):
    await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    if await connection.scalar(text("SELECT to_regclass('docs')")):
        await connection.execute(text('ALTER TABLE docs ADD COLUMN IF NOT EXISTS "embeddingModel" VARCHAR(255) NOT NULL DEFAULT \'legacy_unknown\''))
        await connection.execute(text('ALTER TABLE docs ADD COLUMN IF NOT EXISTS "chunkCount" INTEGER NOT NULL DEFAULT 0'))
    if await connection.scalar(text("SELECT to_regclass('message_sources')")):
        if not await connection.scalar(text("SELECT to_regclass('docs_chunks')")):
            raise RuntimeError("message_sources exists without docs_chunks. Reconcile the existing schema before migrating.")
        missing = await connection.scalar(text('''
            SELECT count(*) FROM message_sources AS source
            WHERE source."chunkId" IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM docs_chunks AS chunk WHERE chunk.id = source."chunkId")
        '''))
        if missing:
            raise RuntimeError("Existing message sources reference missing chunks. Reconcile those references before migrating; no data was removed.")
        await connection.execute(text("ALTER TABLE message_sources ADD COLUMN IF NOT EXISTS snapshot JSON NOT NULL DEFAULT '{}'"))
        await connection.execute(text('ALTER TABLE message_sources ALTER COLUMN "chunkId" DROP NOT NULL'))
        constraints = (await connection.execute(text('''
            SELECT constraint_record.conname, constraint_record.confrelid = 'docs_chunks'::regclass AS correct_target,
                   constraint_record.confdeltype = 'n' AS sets_null
            FROM pg_constraint AS constraint_record
            JOIN pg_attribute AS attribute ON attribute.attrelid = constraint_record.conrelid
                AND attribute.attnum = ANY(constraint_record.conkey)
            WHERE constraint_record.conrelid = 'message_sources'::regclass
                AND constraint_record.contype = 'f' AND attribute.attname = 'chunkId'
        '''))).all()
        if not (len(constraints) == 1 and constraints[0].correct_target and constraints[0].sets_null):
            for constraint in constraints:
                name = connection.dialect.identifier_preparer.quote(constraint.conname)
                await connection.execute(text(f'ALTER TABLE message_sources DROP CONSTRAINT {name}'))
            await connection.execute(text('''
                ALTER TABLE message_sources ADD CONSTRAINT message_sources_chunk_id_fkey
                FOREIGN KEY ("chunkId") REFERENCES docs_chunks(id) ON DELETE SET NULL
            '''))


async def main():
    try:
        async with engine.begin() as connection:
            await migrate(connection)
        print("RAG schema migration completed. Existing tables and records were preserved.")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
