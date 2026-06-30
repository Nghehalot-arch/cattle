from __future__ import annotations

__version__ = "3.5.0"


class threadpool_limits:
    def __init__(self, limits=None, user_api=None):
        self.limits = limits
        self.user_api = user_api

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def threadpool_info():
    return []


class ThreadpoolController:
    def limit(self, limits=None, user_api=None):
        return threadpool_limits(limits=limits, user_api=user_api)

    def info(self):
        return []
