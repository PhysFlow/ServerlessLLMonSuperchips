from __future__ import annotations

import asyncio

__all__ = ["InferenceResponseStream", "settle"]


class InferenceResponseStream:
    def __init__(self, source, release, isolate) -> None:
        self._source = source
        self._iterator = source.__aiter__() if hasattr(source, "__aiter__") else iter(source)
        self._release = release
        self._isolate_keeping_admission = isolate
        self._closed = False
        self._lock = asyncio.Lock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        async with self._lock:
            if self._closed:
                raise StopAsyncIteration
            try:
                if hasattr(self._iterator, "__anext__"):
                    return await self._iterator.__anext__()
                task = asyncio.create_task(asyncio.to_thread(_next_item, self._iterator))
                try:
                    present, item = await asyncio.shield(task)
                except asyncio.CancelledError:
                    await settle(task)
                    raise
                if not present:
                    raise StopAsyncIteration
                return item
            except BaseException:
                await self._close()
                raise

    async def aclose(self):
        async with self._lock:
            await self._close()

    async def _close(self):
        if self._closed:
            return
        task = asyncio.create_task(self._close_source())
        cancelled = False
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                await settle(task)
        except BaseException:
            self._isolate_keeping_admission()
            raise
        self._release()
        self._closed = True
        if cancelled:
            raise asyncio.CancelledError

    async def _close_source(self):
        targets = (self._iterator,) if self._iterator is self._source else (self._iterator, self._source)
        for target in targets:
            if hasattr(target, "aclose"):
                await target.aclose()
            elif hasattr(target, "close"):
                await asyncio.to_thread(target.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


def _next_item(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def settle(task):
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
