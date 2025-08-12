# Supabase Configuration for chatbot_backend

This document describes how Supabase is integrated into the FastAPI backend for:
- User credentials (users table – app-managed)
- Conversations (per chat thread, keyed by `session_id`)
- Messages (chat history per conversation)

The backend connects directly to the Supabase Postgres database via a standard Postgres connection string using SQLAlchemy. SQLite fallback is removed to ensure all environments use the hosted database consistently.

IMPORTANT: Before running the backend, ensure you have created a Supabase project and have the required environment variables available.

1) Required environment variables

Set these in your deployment environment (or `.env` for local dev). The backend supports container-style `REACT_APP_*` keys with fallback to legacy names:

- REACT_APP_SUPABASE_DB_URL (preferred) or SUPABASE_DB_URL (legacy) — REQUIRED
  Example:
    postgresql://postgres:<YOUR-PASSWORD>@db.<ref>.supabase.co:5432/postgres?sslmode=require

  Notes:
  - This is the standard Postgres connection string (not the pooled URL). You may also use the pooled connection string:
    postgresql://<user>:<password>@aws-0-<ref>.pooler.supabase.com:6543/postgres?sslmode=require
  - The backend connects with SQLAlchemy using this URL and will fail fast if it is not provided.

- REACT_APP_GEMINI_API_KEY (preferred) or GEMINI_API_KEY (legacy) — REQUIRED
  - Used by the Gemini API for title generation and chat responses.

- Optional (for future use if you integrate Supabase Auth/Webhooks):
  - REACT_APP_SUPABASE_URL
  - REACT_APP_SUPABASE_ANON_KEY
  - REACT_APP_SUPABASE_SERVICE_ROLE_KEY

2) Database schema (public schema)

Tables used by the backend:

- users
  Columns:
    id                integer primary key
    username          varchar(50) unique not null
    email             varchar(120) unique not null
    hashed_password   varchar(128) not null

- conversations
  Columns:
    id                bigserial primary key
    session_id        text unique not null
    title             text null
    user_id           integer null references public.users(id) on delete set null
    created_at        timestamptz default now() not null

- messages
  Columns:
    id                bigserial primary key
    conversation_id   bigint not null references public.conversations(id) on delete cascade
    role              text not null check (role in ('user','assistant'))
    content           text not null
    created_at        timestamptz default now() not null

3) Automated setup via Supabase (recommended)

Using Supabase SQL Editor or CLI, ensure the tables exist. If using the SQL Editor, run:

```sql
create table if not exists public.conversations (
  id bigserial primary key,
  session_id text unique not null,
  title text,
  user_id integer references public.users(id) on delete set null,
  created_at timestamptz not null default now()
);

create table if not exists public.messages (
  id bigserial primary key,
  conversation_id bigint not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user','assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.users (
  id serial primary key,
  username varchar(50) unique not null,
  email varchar(120) unique not null,
  hashed_password varchar(128) not null
);
```

Row Level Security (RLS):

```sql
alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.users enable row level security;

/* Example permissive read policies for demos; restrict as needed. */
create policy if not exists "conversations_read_all"
  on public.conversations for select using (true);

create policy if not exists "messages_read_all"
  on public.messages for select using (true);

/* Grant privileges to your backend role if not using superuser. Example:
grant usage on schema public to service_role;
grant select, insert, update, delete on all tables in schema public to service_role;
*/
```

Note: If you connect as the `postgres` superuser (service role), RLS is bypassed. For production, use a dedicated role with least privileges and define strict policies.

4) Backend integration details

- SQLAlchemy connection uses:
  - REACT_APP_SUPABASE_DB_URL (preferred) or SUPABASE_DB_URL (legacy). The backend will raise an error if not set.

- Models:
  - `public.users`, `public.conversations`, and `public.messages` are defined in `src/api/auth_utils.py`
  - Tables are created if missing on startup (`create_all`). For Supabase, you should still manage RLS/policies explicitly.

- Main chat flow:
  - On each POST `/chat`:
    * Ensure a conversation row exists for `session_id` (user_id optional).
    * Hydrate in-memory context from DB messages if not already present.
    * Persist the new user message (role='user').
    * Generate the Gemini answer.
    * Persist the assistant message (role='assistant').
    * Return the last 20 messages from DB as `conversation_history`.

- History retrieval:
  - GET `/chat/history/{session_id}` returns the chronological history for the session (up to a reasonable limit).

- Associate sessions with a user:
  - POST `/chat/session/attach` with body `{ "session_id": "...", "username": "...", "title": "optional" }` will bind the session's conversation to the given user and optionally set a title if one is not present.
  - GET `/conversations/{username}` returns the conversations associated with that user, ordered by most recent.

- User authentication:
  - POST `/register` — create a new user (username and email must be unique; password stored as a secure hash).
  - POST `/login` — verify username/password and return basic user info.

5) Local development

- Create a `.env` file with:
  - REACT_APP_SUPABASE_DB_URL=postgresql://postgres:<YOUR-PASSWORD>@db.<ref>.supabase.co:5432/postgres?sslmode=require
  - REACT_APP_GEMINI_API_KEY=<YOUR-GEMINI-API-KEY>

- Start the backend; it will connect to Supabase and create tables if they do not exist.

6) Troubleshooting

- Missing env var: If the backend fails on startup with a DB URL error, set `REACT_APP_SUPABASE_DB_URL` or `SUPABASE_DB_URL` and restart.
- Connection errors: Ensure your URL includes `sslmode=require` for Supabase.
- RLS blocked operations: If you enforce RLS, connect with a role that has necessary grants or implement appropriate policies.
- Stale connections: The backend enables `pool_pre_ping` and `pool_recycle` to mitigate stale connections typical on managed DB services.

Status

- Backend uses Supabase Postgres exclusively.
- User signup/login supported with secure password hashing.
- Conversations and messages persisted per session, with optional association to a user.
- New endpoints to attach sessions to a user and list user conversations enable restoring context after login.
