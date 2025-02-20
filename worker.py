# worker.py

import os
import redis
from rq import Worker, Queue

redis_url = os.getenv('REDISCLOUD_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    # Create a queue *with* the explicit connection
    default_queue = Queue('default', connection=conn)

    # Create a worker *with* the explicit connection
    w = Worker([default_queue], connection=conn)
    w.work()
