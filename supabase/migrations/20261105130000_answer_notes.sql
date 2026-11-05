-- A tester's optional note on an answer, saved with the species they chose.
-- The own_answers policy and the table grants already cover a new column.
alter table test_answers add column note text check (char_length(note) <= 1000);
