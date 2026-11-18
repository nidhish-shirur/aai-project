create extension if not exists pgcrypto;

create type public.app_role as enum (
  'patient',
  'clinician',
  'reviewer',
  'admin'
);

create type public.session_status as enum (
  'active',
  'completed',
  'deleted'
);

create type public.message_role as enum (
  'user',
  'agent'
);

create table public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  role public.app_role not null default 'patient',
  created_at timestamptz not null default now()
);

create table public.patients (
  id uuid primary key default gen_random_uuid(),
  auth_user_id uuid not null unique references auth.users(id) on delete cascade,
  public_code text not null unique,
  created_at timestamptz not null default now(),
  deleted_at timestamptz
);

create table public.sessions (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid not null references public.patients(id) on delete cascade,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  status public.session_status not null default 'active',
  complaint text,
  urgency text check (
    urgency in ('low', 'moderate', 'urgent', 'emergency')
  ),
  state jsonb not null default '{}'::jsonb,
  report jsonb,
  agent_runtime jsonb not null default '{}'::jsonb
);

create table public.messages (
  id bigint generated always as identity primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  role public.message_role not null,
  content text not null,
  created_at timestamptz not null default now()
);

create table public.symptom_events (
  id bigint generated always as identity primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  patient_id uuid not null references public.patients(id) on delete cascade,
  symptom text not null,
  onset_at timestamptz not null,
  recorded_at timestamptz not null default now(),
  resolved_at timestamptz,
  unique (session_id, symptom)
);

create table public.audit_logs (
  id bigint generated always as identity primary key,
  actor_user_id uuid references auth.users(id),
  action text not null,
  entity_type text not null,
  entity_id text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table public.clinician_feedback (
    id bigint generated always as identity primary key,
    session_id uuid not null references public.sessions(id) on delete cascade,
    clinician_user_id uuid not null references auth.users(id) on delete cascade,
    outcome text not null check (
        outcome in ('correct', 'needs_review', 'incorrect_urgency')
    ),
    corrected_urgency text check (
        corrected_urgency in ('low', 'moderate', 'urgent', 'emergency')
    ),
    comment text,
    created_at timestamptz not null default now(),
    unique (session_id, clinician_user_id)
);

create index sessions_patient_idx
  on public.sessions(patient_id, created_at desc);

create index sessions_urgency_idx
  on public.sessions(urgency, updated_at desc);

create index messages_session_idx
  on public.messages(session_id, id);

create index symptom_events_patient_idx
  on public.symptom_events(patient_id, onset_at);

create index audit_logs_actor_idx
  on public.audit_logs(actor_user_id, created_at desc);

create index clinician_feedback_session_idx
on public.clinician_feedback(session_id);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (
    user_id,
    display_name,
    role
  )
  values (
    new.id,
    coalesce(
      new.raw_user_meta_data ->> 'display_name',
      new.email
    ),
    'patient'
  )
  on conflict (user_id) do nothing;

  return new;
end;
$$;

create trigger on_auth_user_created
after insert on auth.users
for each row
execute procedure public.handle_new_user();

create or replace function public.current_role()
returns public.app_role
language sql
stable
security definer
set search_path = public
as $$
  select role
  from public.profiles
  where user_id = auth.uid();
$$;

create or replace function public.is_patient()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select public.current_role() = 'patient';
$$;

create or replace function public.is_clinician()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select public.current_role() in ('clinician', 'reviewer', 'admin');
$$;

create or replace function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select public.current_role() = 'admin';
$$;

create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger sessions_touch_updated_at
before update on public.sessions
for each row
execute procedure public.touch_updated_at();

alter table public.profiles enable row level security;
alter table public.patients enable row level security;
alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.symptom_events enable row level security;
alter table public.audit_logs enable row level security;
alter table public.clinician_feedback enable row level security;

create policy profiles_self_read
on public.profiles
for select
to authenticated
using (
  user_id = auth.uid()
  or public.is_clinician()
);

create policy profiles_admin_update
on public.profiles
for update
to authenticated
using (public.is_admin())
with check (public.is_admin());

create policy patients_self_read
on public.patients
for select
to authenticated
using (auth_user_id = auth.uid());

create policy patients_clinician_read
on public.patients
for select
to authenticated
using (public.is_clinician());

create policy patients_self_insert
on public.patients
for insert
to authenticated
with check (
  auth_user_id = auth.uid()
  and public.is_patient()
);

create policy patients_self_update
on public.patients
for update
to authenticated
using (auth_user_id = auth.uid())
with check (auth_user_id = auth.uid());

create policy patients_admin_delete
on public.patients
for delete
to authenticated
using (public.is_admin());

create policy sessions_patient_read
on public.sessions
for select
to authenticated
using (
  patient_id in (
    select id
    from public.patients
    where auth_user_id = auth.uid()
  )
);

create policy sessions_clinician_read
on public.sessions
for select
to authenticated
using (public.is_clinician());

create policy sessions_patient_insert
on public.sessions
for insert
to authenticated
with check (
  patient_id in (
    select id
    from public.patients
    where auth_user_id = auth.uid()
  )
);

create policy sessions_patient_update
on public.sessions
for update
to authenticated
using (
  patient_id in (
    select id
    from public.patients
    where auth_user_id = auth.uid()
  )
)
with check (
  patient_id in (
    select id
    from public.patients
    where auth_user_id = auth.uid()
  )
);

create policy sessions_clinician_update
on public.sessions
for update
to authenticated
using (public.is_clinician())
with check (public.is_clinician());

create policy messages_patient_read
on public.messages
for select
to authenticated
using (
  session_id in (
    select s.id
    from public.sessions s
    join public.patients p on p.id = s.patient_id
    where p.auth_user_id = auth.uid()
  )
);

create policy messages_clinician_read
on public.messages
for select
to authenticated
using (public.is_clinician());

create policy messages_patient_insert
on public.messages
for insert
to authenticated
with check (
  session_id in (
    select s.id
    from public.sessions s
    join public.patients p on p.id = s.patient_id
    where p.auth_user_id = auth.uid()
  )
);

create policy messages_clinician_insert
on public.messages
for insert
to authenticated
with check (public.is_clinician());

create policy symptom_events_patient_read
on public.symptom_events
for select
to authenticated
using (
  patient_id in (
    select id
    from public.patients
    where auth_user_id = auth.uid()
  )
);

create policy symptom_events_clinician_read
on public.symptom_events
for select
to authenticated
using (public.is_clinician());

create policy symptom_events_patient_insert
on public.symptom_events
for insert
to authenticated
with check (
  patient_id in (
    select id
    from public.patients
    where auth_user_id = auth.uid()
  )
);

create policy symptom_events_clinician_insert
on public.symptom_events
for insert
to authenticated
with check (public.is_clinician());

create policy audit_admin_read
on public.audit_logs
for select
to authenticated
using (public.is_admin());

create policy audit_authenticated_insert
on public.audit_logs
for insert
to authenticated
with check (
  actor_user_id = auth.uid()
);

create policy clinician_feedback_insert
on public.clinician_feedback
for insert
to authenticated
with check (
    clinician_user_id = auth.uid()
    and exists (
        select 1
        from public.user_profiles
        where id = auth.uid()
          and role in ('clinician', 'reviewer', 'admin')
    )
);

create policy clinician_feedback_read
on public.clinician_feedback
for select
to authenticated
using (
    exists (
        select 1
        from public.user_profiles
        where id = auth.uid()
          and role in ('clinician', 'reviewer', 'admin')
    )
);

create policy clinician_feedback_update
on public.clinician_feedback
for update
to authenticated
using (clinician_user_id = auth.uid())
with check (clinician_user_id = auth.uid());