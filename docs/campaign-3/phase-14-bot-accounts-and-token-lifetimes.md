# Campaign 3 Phase 14 — Bot accounts, tokens, and admin refresh

**Status:** Complete and accepted, including immediate token revocation.

## Bot access

The protected Bots group lets administrators publish selected native direct models to automation accounts. Bots remain non-administrators and receive `bots` plus `users` membership. Token scopes still apply; Bot status does not grant administrative routes or a universal model bypass.

Pending accounts may be approved as User, Tester, or Bot. Active roles remain editable in the existing role control. The approval API retains its older tester-boolean compatibility.

Native role-filtered `direct/<exact-tag>` publication and compatibility `audrey_passthrough/<exact-tag>` policy are separate. The public/static `/v1/models` catalog is not a native authorization or token-revocation check.

## Token lifetime

A lifetime of `0` intentionally means no expiration. The API exposes `expires_at: null` and Settings displays **Never**. Values 1–365 retain dated expiry; negative values are invalid.

Every use rechecks the digest, current owner status, scope, and revocation state. The accepted revocation boundary is the same token succeeding at protected `/api/models` before revocation and receiving 401 afterward. Permanent tokens remain revocable.

## Administration freshness

Each Admin Panel opening fetches users, models, and roles with `cache: no-store`, showing newly submitted pending accounts without a whole-page reload.
