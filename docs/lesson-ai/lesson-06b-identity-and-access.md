# Lesson 6b — Identity and access

Every query in Lesson 6a filtered by a `user_id` taken from an authenticated
caller. This lesson is about that caller. It answers two questions: How does
Audrey know who is calling, and what decides what they may do?

Audrey answers them in different places. Proving who someone is happens at the
door, mostly by a service outside Audrey. Deciding what they may do happens
inside, from Audrey's own records, on every request.

## 1. Context

### 1.1 Three kinds of caller

Lesson 4 showed two doors into Audrey, the browser and the API. Three kinds of
caller use them, and each proves who it is a different way:

| Caller | Proves who it is with | Acts as |
|---|---|---|
| A person in the browser | a signed assertion from Cloudflare Access | their own account |
| A script or bot | a personal access token | the token owner's account, within the token's scopes |
| Audrey's own services, such as the tools sidecar | the shared service token | no account; each call names the person it works for |

The first two end the same way, in a **principal**: Audrey's picture of one
account, built fresh for each request. Routes don't read credentials. They
declare which principal they need, and a FastAPI dependency either supplies one
or stops the request with 401 or 403 before the route runs.

### 1.2 Who it is, and what they may do

**Authentication** answers "who is this?" and **authorization** answers "what
may they do?" Audrey keeps the two apart on purpose. Cloudflare Access decides
who can reach the site and proves who they are, but it has no say over what
they can do inside: Status, groups and administrator rights live in Audrey's
database, and signing in can't change them (`sync_role=False` in
[`_resolve_cloudflare_access`](../../src/audrey/auth.py#L92)).

Every request passes the same checks, in this order:

```text
a request
 -> who is it?          an Access assertion or a personal token      401
 -> which account?      the principal: Audrey's ids, status, groups
 -> active?             pending and disabled accounts stop here       403
 -> token scope?        tokens only; browser sessions have none       403
 -> the route's rule    administrators only, browser only, ...        403
 -> whose data?         every query filters by user_id                404
```

The last line is Lesson 6a's ownership rule. This lesson covers the ones above
it.

### 1.3 The files

| Job | File |
|---|---|
| The request dependencies: who is calling, and what they may reach | [`auth.py`](../../src/audrey/auth.py) |
| Checking Cloudflare Access assertions | [`cloudflare_access.py`](../../src/audrey/identity/cloudflare_access.py) |
| The principal and token types | [`models.py`](../../src/audrey/identity/models.py) |
| Accounts, logins, tokens, groups and approval | [`store.py`](../../src/audrey/app_state/store.py) |
| Account settings and tokens | [`me.py`](../../src/audrey/routes/app/me.py) |
| The first-administrator command | [`admin_cli.py`](../../src/audrey/admin_cli.py) |

## 2. Read-along

### 2.1 The browser: a signed assertion

After someone signs in, Cloudflare Access adds a JSON Web Token to each of their
requests, in the `Cf-Access-Jwt-Assertion` header, and nginx passes it on
(Lesson 4). Audrey trusts it only after checking it
([`CloudflareAccessVerifier.verify`](../../src/audrey/identity/cloudflare_access.py#L66)):

- **The signature**, RS256 only, against the signing keys Cloudflare publishes
  for the team. Audrey caches the keys for an hour. A token signed with a key it
  hasn't seen makes it fetch the keys again straight away, since Cloudflare
  rotates them, but at most once every 30 seconds, so a flood of made-up key
  ids can't make Audrey hammer Cloudflare
  ([`_get_keys`](../../src/audrey/identity/cloudflare_access.py#L128)).
- **The claims:** issued by the team's domain, meant for this application (the
  audience tag from `.env`), not expired, allowing 30 seconds of clock skew, and
  carrying a subject and an email.

A token that fails a check gets 401. If Audrey can't fetch Cloudflare's keys at
all, it answers 502 instead: It couldn't check the token, which isn't the same
as the token being bad.

One rule in [`require_user`](../../src/audrey/auth.py#L156) matters more than it
looks. When the assertion header is present, it decides the request: A bad
assertion fails there, and Audrey never falls back to a bearer token that
happens to ride along on the same request.

### 2.2 From a login to an account

A checked assertion says who someone is to Cloudflare: a subject id and an
email. [`resolve_external_identity`](../../src/audrey/app_state/store.py#L254)
turns that into an Audrey account:

```text
this login already linked?           -> its account
else one account has that email      -> link the login to that account
else several accounts have it        -> 409, until someone reconciles them
else                                 -> a new account: fresh ids, status 'pending'
```

Linking by email is how an account an administrator created ahead of time meets
its owner's first sign-in. Once linked, a login's email can't change quietly.
If the same subject arrives with a different email, Audrey refuses with 409,
and the account needs an explicit migration.

Most requests come from a login that's already linked and have nothing new to
record, so they only read. The account's status and groups are still read fresh
each time, without taking the database's write lock
([`store.py:326`](../../src/audrey/app_state/store.py#L326 '# Re-read current status and groups on every request without')).
A request writes only when something changed, or when the last-seen time the
Admin panel shows is five minutes old.

### 2.3 The principal

What comes out is a principal, at
[`models.py:10-35`](../../src/audrey/identity/models.py#L10-L35):

```python
@dataclass(frozen=True, slots=True)
class Principal:
    ...
    user_id: str
    storage_namespace: str
    provider: str
    provider_subject: str
    email: str
    display_name: str
    role: str
    status: str
    auth_method: str
    groups: frozenset[str] = frozenset()
    token_id: str | None = None
    scopes: frozenset[str] = frozenset()

    @property
    def is_admin(self) -> bool:
        return self.role == "admin" or "admins" in self.groups
```

- **Two ids belong to Audrey.** `user_id` (`usr_…`) keys every row in the
  application database. `storage_namespace` keys what lives outside it: saved
  memories, the chat archive and files on disk. New accounts get a random
  `ns_…`. Accounts from before Audrey kept its own accounts kept their original
  email as their namespace, so nothing stored had to move.
- **Everything else is an attribute.** Changing an email or a display name moves
  no data.
- **How the request proved itself** is in `auth_method`. Only a token sets
  `token_id` and `scopes`.
- **It's frozen:** the account as this request found it. The next request
  builds a new one, so an administrator's change applies at once.

Older routes receive an [`AuthedUser`](../../src/audrey/auth.py#L35) instead,
which wraps the principal. Its `email` field holds the storage namespace, not
the email, a name kept so those routes didn't have to change.

### 2.4 Personal access tokens

Scripts and bots can't sign in through a browser, so a person makes them a
token in Settings ([`create_token`](../../src/audrey/routes/app/me.py#L225)). A
token has a name, its scopes and a lifetime: 90 days unless chosen otherwise, at
most 365, or 0 for one that never expires. It's shown once:

```text
aud_pat_0f3c…9a1e.Q2xh…      # aud_, the token's id, a dot, a random secret
```

The database keeps only a SHA-256 digest of the whole token
([`create_personal_token`](../../src/audrey/app_state/store.py#L588)), so a copy
of the database holds no working tokens. The id inside the token lets Audrey
find its row directly, and every use repeats every check
([`_authenticate_personal_token_sync`](../../src/audrey/app_state/store.py#L726)):

```text
aud_pat_<id>.<secret>
 -> the row for <id>
 -> the digest matches?             no  -> 401
 -> revoked or expired?             yes -> 401
 -> the owner's account active?     no  -> 401
 -> a principal with the token's scopes
```

Nothing is cached, so revoking a token, or disabling its owner, takes effect on
the very next request. The only write is the token's last-used time, at most
once every five minutes.

A token holds one or both of two scopes
([`TOKEN_SCOPES`](../../src/audrey/identity/models.py#L7)): `account:read`, to
read the account's own details, and `compat:full`, for everything else a person
can do through the API. Every `/v1/` route requires `compat:full` as part of
authentication ([`_resolve_personal_token`](../../src/audrey/auth.py#L54)), and
native routes declare theirs with
[`require_scope`](../../src/audrey/auth.py#L244). Scopes only narrow tokens: A
browser session has none and needs none.

No token can change account settings, manage tokens or administer Audrey,
whatever its scopes. Those need a browser session
([`require_provider_principal`](../../src/audrey/auth.py#L260)), so a leaked
token can't make more tokens or promote anyone.

### 2.5 Status, groups and administrators

An account's **status** is `pending` from its first sign-in until an
administrator approves it, then `active`, or `disabled` if an administrator
turns it off. Only an active account gets past the checks. A pending account can
still read its own status
([`require_account_user`](../../src/audrey/auth.py#L188)), which is how the
browser can say that approval is pending instead of just failing.

What an active account may use comes from its **groups**:

| Group | Members |
|---|---|
| `users` | every active account |
| `testers` | people who get preview features and models |
| `bots` | automation accounts |
| `admins` | administrators |

An account is an administrator when its role is `admin` or it belongs to
`admins` ([`is_admin`](../../src/audrey/identity/models.py#L34)), and
administrator changes keep the two in step. Administrators approve a pending
account as a user, tester or bot, change groups, and disable accounts
([`_admin_update_user_sync`](../../src/audrey/app_state/store.py#L1239)). That
method refuses changes that would leave an account half-configured or lock
Audrey out: A pending account can't hold groups, an active one always keeps
`users`, administrators can't disable themselves or drop their own admin
access, and the last active administrator can't be removed. Each change goes
into an audit log with the account's state before and after.

Approving an account as a bot puts it in `users` and `bots`. Bots are kept out of
Private work, agents, page watches and automatic skill selection
([`_active_owner`](../../src/audrey/tax_review.py#L113) is one such check).

### 2.6 Which models an account may use

Groups also decide which models an account may pick. Each model the native app
offers has an access policy, set in the Admin panel: whether it's enabled, and
its **audience**, one of the groups above. A model whose audience is `admins` is
restricted to administrators, any other audience opens it to that group, and
administrators may use every model. The policy is checked against the current
principal on every request, so a change to someone's groups, or to a model's
audience, applies to their next message. The next lesson shows the catalog
applying it, both in the picker and again when a run starts.

The `/v1` API has its own, separate list: Audrey's virtual models plus a
passthrough allowlist from configuration, open to every caller whose token has
`compat:full`. The two are independent controls. A model's native audience
doesn't affect `/v1`, and `/v1` callers aren't filtered by group.

### 2.7 Services, and work that outlives its request

The tools sidecar reads saved conversations for the chat-history tool, and media
workers claim video jobs and post their results. Their routes accept only the
shared service token, from `.env`, in the `X-Audrey-Service-Token` header,
compared in constant time ([`require_service`](../../src/audrey/auth.py#L299)).
A person's login doesn't open them: These routes act on any account's data, so
a user's credentials there would let one person claim another's video job.

A service call also has to say whose data it wants, and the service token can't
prove that part. Dispatch does. When a model calls a tool that works on personal
data, Audrey overwrites the tool's user argument with the caller's storage
namespace before the call leaves
([`dispatch.py:293`](../../src/audrey/tools/dispatch.py#L293 '# Bind the authenticated identity exactly as the declared policy says.')),
so the model never chooses whose data a tool reads. The token proves the caller
is Audrey's own service, and the binding proves which person it works for. The
chat-history route needs both
([`read_conversation`](../../src/audrey/routes/tool_history.py#L195)).

Work that runs after its request ends, such as agents, page watches and Private
tasks, has no request to authenticate. It keeps its owner's `user_id`, and
before it runs, it reads the account again
([`get_principal`](../../src/audrey/app_state/store.py#L529)) and checks that
it's still active and may still use the model it needs
([`agent_executor.py:90`](../../src/audrey/agent_executor.py#L90 'principal = await store.get_principal(user_id=user_id)')).
Authority isn't frozen when work is queued: An account disabled overnight stops
its scheduled work too.

### 2.8 The first administrator

Approval needs an administrator, which leaves the very first one with nobody to
approve them. An operator command fills that gap
([`admin_cli.py:33`](../../src/audrey/admin_cli.py#L33 '"grant-admin",')):

```text
audrey-admin grant-admin --email alice@example.com
```

The person signs in once, so that a pending account exists. The command then
activates it, adds it to `users` and `admins`, and writes an audit entry
([`_bootstrap_admin_sync`](../../src/audrey/app_state/store.py#L1512)). If two
accounts share the email, it refuses and asks for the exact `usr_…` id instead.
From then on, administrators manage everyone else in the Admin panel.

## 3. Comprehension questions

**1. alice@example.com signs in through Cloudflare Access for the first time and
is told the account is awaiting approval. What does the database hold for it,
and what has to happen next?**

A pending account with fresh `usr_…` and `ns_…` ids, the login linked to it,
default settings, and no groups (§2.2, §2.5). All it can do is read its own
status. An administrator approves it as a user, tester or bot. If Audrey has no
administrator yet, an operator runs `audrey-admin grant-admin` (§2.8).

**2. A script's token is pasted into a public chat by mistake, and its owner
revokes it a minute later. When does it stop working, and could it have made a
replacement in that minute?**

It stops on the next request that uses it: Every use rechecks revocation, and
nothing is cached (§2.4). It couldn't have made a replacement, because managing
tokens needs a browser session, whatever the token's scopes.

**3. An administrator disables Bob's account. A script holds one of Bob's
tokens, and a page watch of Bob's is due to run tonight. What happens to each?**

The token fails with 401 on its next use, because each use rechecks the owner's
status (§2.4). The watch reads Bob's account again before it runs, finds it
disabled, and stops (§2.7). Neither kept the authority it had when it started.

**4. An administrator gives a script a token with `compat:full`. Can the script
approve pending accounts?**

No. Administration needs a browser session, and no scope changes that (§2.4). A
token can use what its owner can use through the API, but it can't change who
has access.

**5. During a chat, a model calls the chat-history tool with its `user` argument
set to Bob's namespace. Whose conversations come back?**

The caller's own. Dispatch replaces the user argument with the caller's storage
namespace before the call reaches the sidecar (§2.7). The service token only
proves the call comes from Audrey's own sidecar; the binding decides whose data
it reads.

## When you're ready for the next lesson

Lessons 6a and 6b covered what Audrey keeps and who may reach it. The next
lesson returns to the answer pipeline and opens the model layer:
[Lesson 6 — The model layer I: choosing a model](lesson-06c-the-model-layer.md).
It starts where §2.6 stopped, with the catalog that turns each model's audience
into the picker a person sees, then traces what happens when a pipeline node
says "I need a general-purpose model": how the registry ranks candidates, and
how the health tracker steps around failing ones.
