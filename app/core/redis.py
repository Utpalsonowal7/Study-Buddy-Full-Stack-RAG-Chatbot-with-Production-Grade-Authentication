from upstash_redis import Redis
from app.config import REDIS_TOKEN,REDIS_URL

redis = Redis(url=REDIS_URL, token=REDIS_TOKEN)
redis.set("foo", "bar")
