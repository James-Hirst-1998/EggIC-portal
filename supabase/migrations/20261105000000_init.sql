-- EggIC storage and staff test. Applied by pasting into the Supabase SQL editor.
-- What each part is for: README.md, "What gets stored". Never edit once applied;
-- add a new file to change the schema.

-- Private buckets
insert into storage.buckets (id, name, public) values
  ('submissions', 'submissions', false),
  ('test-images', 'test-images', false)
on conflict (id) do nothing;

-- Every photo the public page sends, with the model's answer
create table submissions (
  id            text primary key,          -- e.g. 20260804T153012-a1b2c3d4
  created_at    timestamptz not null default now(),
  image_path    text not null,             -- file in the submissions bucket
  filename      text,                      -- name on the phone
  model_version text not null,             -- the arm that answered, e.g. ensemble
  top           text not null,             -- species key, e.g. ThornbackRay
  confidence    real not null,             -- shown scale
  confident     boolean not null,          -- confidence >= threshold
  threshold     real not null,             -- the accept line when it was served
  ranked        jsonb not null,            -- all nine scores
  by_model      jsonb,                     -- each arm's top + confidence
  latency_ms    int,
  correct       boolean,                   -- tester's Y / N, null until answered
  user_species  text,                      -- what the tester says it is
  feedback_at   timestamptz
);

-- Staff test: the answer key. Staff can't read it.
create table test_images (
  image_path       text primary key,       -- random name in the test-images bucket
  true_species     text not null,
  specimen         text,                   -- same egg, another angle (ST sighting id)
  source           text not null,          -- where the photo came from
  model_top        text,                   -- filled by scripts/score_test_set.py
  model_confidence real
);

-- Staff test: one answer per person per photo
create table test_answers (
  user_id    uuid not null default auth.uid() references auth.users on delete cascade,
  image_path text not null references test_images,
  species    text not null,                -- a species key, or 'not_sure'
  created_at timestamptz not null default now(),
  primary key (user_id, image_path)
);

-- Lock it all down. No policy means only the secret key gets in.
alter table submissions  enable row level security;
alter table test_images  enable row level security;
alter table test_answers enable row level security;

-- Spelled out in case the project doesn't expose new tables to the API by default.
grant all on submissions, test_images, test_answers to service_role;
grant select, insert, update on test_answers to authenticated;

-- A tester is a user made by scripts/add_tester.py, which sets app_metadata.tester.
-- Users can't set app_metadata themselves, so a stray sign-up sees nothing.
create function public.is_tester() returns boolean language sql stable as $$
  select coalesce((auth.jwt() -> 'app_metadata' ->> 'tester')::boolean, false)
$$;
grant execute on function public.is_tester() to authenticated;

-- Testers read and write their own answers only
create policy own_answers on test_answers for all to authenticated
  using (public.is_tester() and user_id = auth.uid())
  with check (public.is_tester() and user_id = auth.uid());

-- Testers can list and view the test photos, nothing else
create policy read_test_images on storage.objects for select to authenticated
  using (bucket_id = 'test-images' and public.is_tester());
