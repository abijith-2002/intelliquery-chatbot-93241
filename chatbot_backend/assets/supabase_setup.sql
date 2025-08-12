-- Supabase setup script for IntelliQuery Chatbot (backend)
-- Purpose:
--  - Create required tables: users, conversations, messages
--  - Enable RLS and create permissive read policies (adjust for prod)
--  - Bootstrap a generic RPC function public.run_sql(text) to enable automation tools that depend on it

-- 1) Tables (public schema)

create table if not exists public.users (
  id serial primary key,
  username varchar(50) unique not null,
  email varchar(120) unique not null,
  hashed_password varchar(128) not null
);

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

-- 2) Row Level Security (enable + permissive demo policies)
alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.users enable row level security;

-- NOTE: These policies are permissive for demos. Tighten or replace for production.
create policy if not exists "conversations_read_all"
  on public.conversations for select using (true);

create policy if not exists "messages_read_all"
  on public.messages for select using (true);

-- Example grants to a service role (adjust as needed)
-- grant usage on schema public to service_role;
-- grant select, insert, update, delete on all tables in schema public to service_role;

-- 3) Bootstrap RPC function for automation tools
-- Some tools (including our CI tools) expect a generic RPC endpoint: public.run_sql(query text)
-- This implementation tries to return JSON for SELECT/WITH queries and a simple status JSON for non-SELECT.
-- WARNING: Be cautious with exposing such a function; it can be powerful. Restrict privileges as needed.

create or replace function public.run_sql(query text)
returns jsonb
language plpgsql
security definer
as $$
declare
  res jsonb;
  q text := trim(both from query);
begin
  if q ilike 'select %' or q ilike 'with %' then
    execute format('select coalesce(jsonb_agg(t), ''[]''::jsonb) from (%s) t', query) into res;
    return coalesce(res, '[]'::jsonb);
  else
    execute query;
    return jsonb_build_object('status','ok');
  end if;
exception
  when others then
    return jsonb_build_object('status','error','message',SQLERRM);
end;
$$;

-- Grant execute as appropriate for your environment.
-- For tooling, service_role typically suffices. Add/remove as needed.
do $$
begin
  begin
    grant execute on function public.run_sql(text) to service_role;
  exception when undefined_object then
    -- role might not exist on non-Supabase Postgres
    null;
  end;
  begin
    grant execute on function public.run_sql(text) to authenticated;
  exception when undefined_object then
    null;
  end;
  begin
    grant execute on function public.run_sql(text) to anon;
  exception when undefined_object then
    null;
  end;
end$$;
