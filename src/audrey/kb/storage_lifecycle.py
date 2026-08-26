"""Provider-neutral storage reservations over the uploads SQLite index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from audrey.kb.uploads_db import QuotaDecision, QuotaUsage, UploadsDB

ReservationKind = Literal["single_shot", "chunked", "url_fetch"]


@dataclass(frozen=True)
class StorageReservation:
    reservation_id: str
    user: str
    kind: ReservationKind
    reserved_bytes: int


class QuotaExceededError(Exception):
    """A transactional reservation would put one user over their limit."""

    def __init__(self, decision: QuotaDecision) -> None:
        self.usage = decision.usage
        self.requested_bytes = decision.requested_bytes
        self.max_user_bytes = decision.max_user_bytes
        super().__init__(
            "storage quota exceeded: "
            f"{decision.usage.total_bytes} + {decision.requested_bytes} > "
            f"{decision.max_user_bytes}",
        )


class StorageLifecycle:
    """Own quota decisions and reservation-to-storage transitions."""

    def __init__(self, db: UploadsDB) -> None:
        self._db = db

    @staticmethod
    def _require(decision: QuotaDecision) -> None:
        if not decision.accepted:
            raise QuotaExceededError(decision)

    async def usage(self, user: str) -> QuotaUsage:
        return await self._db.quota_usage(user)

    async def reserve_single_upload(
        self,
        *,
        reservation_id: str,
        user: str,
        bytes_: int,
        max_user_bytes: int,
        now: str,
        expired_before: str,
    ) -> StorageReservation:
        decision = await self._db.reserve_single_upload(
            reservation_id=reservation_id,
            user=user,
            bytes_=bytes_,
            max_user_bytes=max_user_bytes,
            now=now,
            expired_before=expired_before,
        )
        self._require(decision)
        return StorageReservation(reservation_id, user, "single_shot", bytes_)

    async def open_chunk_session(
        self,
        *,
        upload_id: str,
        user: str,
        filename: str,
        total_bytes: int,
        part_size: int,
        parts_total: int,
        max_user_bytes: int,
        now: str,
        expired_before: str,
    ) -> StorageReservation:
        decision = await self._db.reserve_upload_session(
            upload_id=upload_id,
            user=user,
            filename=filename,
            total_bytes=total_bytes,
            part_size=part_size,
            parts_total=parts_total,
            max_user_bytes=max_user_bytes,
            now=now,
            expired_before=expired_before,
        )
        self._require(decision)
        return StorageReservation(upload_id, user, "chunked", total_bytes)

    async def record_chunk_part(
        self,
        reservation: StorageReservation,
        *,
        part_no: int,
        bytes_: int,
        now: str,
    ) -> None:
        if reservation.kind != "chunked":
            raise ValueError("only a chunk reservation can accept parts")
        await self._db.record_reserved_part(
            upload_id=reservation.reservation_id,
            user=reservation.user,
            part_no=part_no,
            bytes_=bytes_,
            now=now,
        )

    async def reserve_url_fetch(
        self,
        *,
        file_id: str,
        user: str,
        source_url: str,
        filename: str,
        ceiling_bytes: int,
        max_user_bytes: int,
        now: str,
        expired_before: str,
    ) -> StorageReservation:
        decision = await self._db.reserve_url_fetch(
            file_id=file_id,
            user=user,
            source_url=source_url,
            filename=filename,
            ceiling_bytes=ceiling_bytes,
            max_user_bytes=max_user_bytes,
            uploaded_at=now,
            expired_before=expired_before,
        )
        self._require(decision)
        return StorageReservation(file_id, user, "url_fetch", ceiling_bytes)

    async def reserve_url_refetch(
        self,
        *,
        file_id: str,
        user: str,
        ceiling_bytes: int,
        max_user_bytes: int,
        expired_before: str,
    ) -> StorageReservation:
        decision = await self._db.reserve_url_refetch(
            file_id=file_id,
            user=user,
            ceiling_bytes=ceiling_bytes,
            max_user_bytes=max_user_bytes,
            expired_before=expired_before,
        )
        self._require(decision)
        return StorageReservation(file_id, user, "url_fetch", ceiling_bytes)

    async def restore_pending_url_fetches(self, *, ceiling_bytes: int) -> int:
        return await self._db.restore_url_fetch_reservations(
            ceiling_bytes=ceiling_bytes,
        )

    async def commit_upload(
        self,
        reservation: StorageReservation,
        *,
        file_id: str,
        filename: str,
        mime: str,
        bytes_: int,
        kind: str,
        collection: str,
        chunks: int,
        uploaded_at: str,
        status: str,
        max_user_bytes: int,
    ) -> None:
        if reservation.kind == "url_fetch":
            raise ValueError("URL reservations commit through complete_fetch")
        decision = await self._db.commit_reserved_upload(
            reservation_id=reservation.reservation_id,
            reservation_kind=reservation.kind,
            user=reservation.user,
            file_id=file_id,
            filename=filename,
            mime=mime,
            bytes_=bytes_,
            kind=kind,
            collection=collection,
            chunks=chunks,
            uploaded_at=uploaded_at,
            status=status,
            max_user_bytes=max_user_bytes,
        )
        self._require(decision)

    async def release(self, reservation: StorageReservation) -> None:
        if reservation.kind == "single_shot":
            await self._db.release_storage_reservation(
                reservation.reservation_id,
                user=reservation.user,
            )
        elif reservation.kind == "chunked":
            await self._db.close_session(reservation.reservation_id)
        else:
            await self._db.release_url_reservation(
                reservation.reservation_id,
                user=reservation.user,
            )

    async def expire_chunk_sessions(self, *, older_than: str) -> list[dict]:
        return await self._db.expire_upload_sessions(older_than=older_than)


__all__ = [
    "QuotaExceededError",
    "StorageLifecycle",
    "StorageReservation",
]
