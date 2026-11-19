-- One row per egg case: a photo holds none, one or several, and each gets its own
-- answer, outline and verdict. submissions keeps the photo and the run.
create table cases (
  submission_id text not null references submissions on delete cascade,
  n             int  not null check (n >= 1),  -- 1, 2, … top to bottom
  box           jsonb,                     -- [x0, y0, x1, y1], fractions of the photo; null = whole photo
  outline       jsonb,                     -- [[[x, y], …], …], fractions of the photo; null = whole photo
  mask_path     text,                      -- PNG in the submissions bucket; null = whole photo
  top           text not null,             -- species key, e.g. ThornbackRay
  confidence    real not null,             -- shown scale
  confident     boolean not null,          -- confidence >= the photo's threshold
  ranked        jsonb not null,            -- every class's score
  by_model      jsonb,                     -- each arm's top + confidence
  correct       boolean,                   -- tester's Y / N, null until answered
  user_species  text,                      -- what the tester says it is
  feedback_at   timestamptz,
  primary key (submission_id, n)
);
alter table cases enable row level security;
grant all on cases to service_role;

alter table submissions
  add column segment_status text,          -- ok | no_egg_case | too_many | unsure | skipped
  add column case_count     int;           -- rows in cases for this photo

-- The answer moves to cases. Rows before this keep theirs here as well.
alter table submissions
  alter column top drop not null,
  alter column confidence drop not null,
  alter column confident drop not null,
  alter column ranked drop not null;

-- Every earlier photo was named whole: one case, unless the model was sure it held none.
insert into cases (submission_id, n, top, confidence, confident, ranked, by_model,
                   correct, user_species, feedback_at)
select id, 1, top, confidence, confident, ranked, by_model, correct, user_species, feedback_at
from submissions
where not (confident and top = 'NoEggCase');

update submissions s set case_count = (select count(*) from cases c where c.submission_id = s.id);
