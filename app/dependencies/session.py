from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator
from app.db.database import LocalSession

async def get_session()->AsyncGenerator[AsyncSession,None]:
     async with LocalSession() as se:
          yield se