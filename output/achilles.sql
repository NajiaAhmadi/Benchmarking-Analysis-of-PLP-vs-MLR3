-- 0	cdm name, version of Achilles and date when pre-computations were executed
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_0
AS
SELECT
0 as analysis_id,  CAST('' AS VARCHAR(255)) as stratum_1, CAST('1.7' AS VARCHAR(255)) as stratum_2, 
CAST(CURRENT_DATE AS VARCHAR(255)) as stratum_3,
cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value
FROM
omop.cmd.person;
ANALYZE s_tmpach_0
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_0
AS
SELECT
0 as analysis_id, CAST('' AS VARCHAR(255)) as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value, 
  cast(null as NUMERIC) as min_value,
	cast(null as NUMERIC) as max_value,
	cast(null as NUMERIC) as avg_value,
	cast(null as NUMERIC) as stdev_value,
	cast(null as NUMERIC) as median_value,
	cast(null as NUMERIC) as p10_value,
	cast(null as NUMERIC) as p25_value,
	cast(null as NUMERIC) as p75_value,
	cast(null as NUMERIC) as p90_value
FROM
omop.cmd.person;
ANALYZE s_tmpach_dist_0
;


-- 1	Number of persons
CREATE TEMP TABLE s_tmpach_1
AS
SELECT
1 as analysis_id,  
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value
FROM
omop.cmd.person;
ANALYZE s_tmpach_1
;


-- 2	Number of persons by gender
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2
AS
SELECT
2 as analysis_id, 
CAST(gender_concept_id AS VARCHAR(255)) as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value
FROM
omop.cmd.person
group by GENDER_CONCEPT_ID;
ANALYZE s_tmpach_2
;


-- 3	Number of persons by year of birth
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_3
AS
SELECT
3 as analysis_id,  CAST(year_of_birth AS VARCHAR(255)) as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value
FROM
omop.cmd.person
group by YEAR_OF_BIRTH;
ANALYZE s_tmpach_3
;


-- 4	Number of persons by race
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_4
AS
SELECT
4 as analysis_id,  CAST(RACE_CONCEPT_ID AS VARCHAR(255)) as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value
FROM
omop.cmd.person
group by RACE_CONCEPT_ID;
ANALYZE s_tmpach_4
;


-- 5	Number of persons by ethnicity
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_5
AS
SELECT
5 as analysis_id,  CAST(ETHNICITY_CONCEPT_ID AS VARCHAR(255)) as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value
FROM
omop.cmd.person
group by ETHNICITY_CONCEPT_ID;
ANALYZE s_tmpach_5
;


-- 7	Number of persons with invalid provider_id
CREATE TEMP TABLE s_tmpach_7
AS
SELECT
7 as analysis_id,  
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(p1.person_id) as count_value
FROM
omop.cmd.person p1
	left join omop.cmd.provider pr1
	on p1.provider_id = pr1.provider_id
where p1.provider_id is not null
	and pr1.provider_id is null
;
ANALYZE s_tmpach_7
;


-- 8	Number of persons with invalid location_id
CREATE TEMP TABLE s_tmpach_8
AS
SELECT
8 as analysis_id,  
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(p1.person_id) as count_value
FROM
omop.cmd.person p1
	left join omop.cmd.location l1
	on p1.location_id = l1.location_id
where p1.location_id is not null
	and l1.location_id is null
;
ANALYZE s_tmpach_8
;


-- 9	Number of persons with invalid care_site_id
CREATE TEMP TABLE s_tmpach_9
AS
SELECT
9 as analysis_id,  
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(p1.person_id) as count_value
FROM
omop.cmd.person p1
	left join omop.cmd.care_site cs1
	on p1.care_site_id = cs1.care_site_id
where p1.care_site_id is not null
	and cs1.care_site_id is null
;
ANALYZE s_tmpach_9
;


-- 10	Number of all persons by year of birth and by gender
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_10
AS
SELECT
10 as analysis_id,  CAST(year_of_birth AS VARCHAR(255)) as stratum_1,
  CAST(gender_concept_id AS VARCHAR(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
  COUNT(distinct person_id) as count_value
FROM
omop.cmd.person
group by YEAR_OF_BIRTH, gender_concept_id;
ANALYZE s_tmpach_10
;


-- 11	Number of non-deceased persons by year of birth and by gender
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_11
AS
SELECT
11 as analysis_id,  CAST(P.year_of_birth AS VARCHAR(255)) as stratum_1,
  CAST(P.gender_concept_id AS VARCHAR(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
  COUNT(distinct P.person_id) as count_value
FROM
omop.cmd.person P
where not exists
(
  select 1
  from omop.cmd.death D
  where P.person_id = D.person_id
)
group by P.YEAR_OF_BIRTH, P.gender_concept_id;
ANALYZE s_tmpach_11
;


-- 12	Number of persons by race and ethnicity
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_12
AS
SELECT
12 as analysis_id, CAST(RACE_CONCEPT_ID AS VARCHAR(255)) as stratum_1, CAST(ETHNICITY_CONCEPT_ID AS VARCHAR(255)) as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct person_id) as count_value
FROM
omop.cmd.person
group by RACE_CONCEPT_ID,ETHNICITY_CONCEPT_ID;
ANALYZE s_tmpach_12
;


-- 101	Number of persons by age, with age at first observation period
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_101
AS
WITH rawData  AS (
  select
    EXTRACT(YEAR FROM op1.index_date) - p1.YEAR_OF_BIRTH as stratum_1,
    COUNT(p1.person_id) as count_value
  from omop.cmd.person p1
    inner join (select person_id, MIN(observation_period_start_date) as index_date from omop.cmd.observation_period group by PERSON_ID) op1
    on p1.PERSON_ID = op1.PERSON_ID
  group by EXTRACT(YEAR FROM op1.index_date) - p1.YEAR_OF_BIRTH
)
 SELECT
101 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(null as varchar(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_101
;


-- 102	Number of persons by gender by age, with age at first observation period
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_102
AS
WITH rawData  AS (
  select
    p1.gender_concept_id as stratum_1,
    EXTRACT(YEAR FROM op1.index_date) - p1.YEAR_OF_BIRTH as stratum_2,
    COUNT(p1.person_id) as count_value
  from omop.cmd.person p1
    inner join (select person_id, MIN(observation_period_start_date) as index_date from omop.cmd.observation_period group by PERSON_ID) op1
    on p1.PERSON_ID = op1.PERSON_ID
  group by p1.gender_concept_id, EXTRACT(YEAR FROM op1.index_date) - p1.YEAR_OF_BIRTH)
 SELECT
102 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(stratum_2 as varchar(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_102
;


-- 103	Distribution of age at first observation period
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_103
AS
WITH rawData (person_id, age_value)  AS (
select p.person_id, 
  MIN(EXTRACT(YEAR FROM observation_period_start_date)) - P.YEAR_OF_BIRTH as age_value
  from omop.cmd.person p
  JOIN omop.cmd.observation_period op on p.person_id = op.person_id
  group by p.person_id, p.year_of_birth
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * age_value) AS NUMERIC) as avg_value,
  CAST(STDDEV(age_value) AS NUMERIC) as stdev_value,
  min(age_value) as min_value,
  max(age_value) as max_value,
  COUNT(*) as total
  FROM rawData
),
ageStats (age_value, total, rn) as
(
  select age_value, COUNT(*) as total, row_number() over (order by age_value) as rn
  from rawData
  group by age_value
),
ageStatsPrior (age_value, total, accumulated) as
(
  select s.age_value, s.total, sum(p.total) as accumulated
  from ageStats s
  join ageStats p on p.rn <= s.rn
  group by s.age_value, s.total, s.rn
),
tempResults as
(
  select 103 as analysis_id,
    o.total as count_value,
  	o.min_value,
  	o.max_value,
  	o.avg_value,
  	o.stdev_value,
  	MIN(case when p.accumulated >= .50 * o.total then age_value end) as median_value,
  	MIN(case when p.accumulated >= .10 * o.total then age_value end) as p10_value,
  	MIN(case when p.accumulated >= .25 * o.total then age_value end) as p25_value,
  	MIN(case when p.accumulated >= .75 * o.total then age_value end) as p75_value,
  	MIN(case when p.accumulated >= .90 * o.total then age_value end) as p90_value
  --INTO #tempResults
  from ageStatsPrior p
  CROSS JOIN overallStats o
  GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
)
 SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5, 
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults
;
ANALYZE s_tmpach_dist_103
;


-- 104	Distribution of age at first observation period by gender
--HINT DISTRIBUTE_ON_KEY(stratum_1) 
CREATE TEMP TABLE tempResults_104
AS
WITH rawData (gender_concept_id, age_value)  AS (
  select p.gender_concept_id, MIN(EXTRACT(YEAR FROM observation_period_start_date)) - P.YEAR_OF_BIRTH as age_value
	from omop.cmd.person p
	JOIN omop.cmd.observation_period op on p.person_id = op.person_id
	group by p.person_id,p.gender_concept_id, p.year_of_birth
),
overallStats (gender_concept_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select gender_concept_id,
  CAST(avg(1.0 * age_value) AS NUMERIC) as avg_value,
  CAST(STDDEV(age_value) AS NUMERIC) as stdev_value,
  min(age_value) as min_value,
  max(age_value) as max_value,
  COUNT(*) as total
  FROM rawData
  group by gender_concept_id
),
ageStats (gender_concept_id, age_value, total, rn) as
(
  select gender_concept_id, age_value, COUNT(*) as total, row_number() over (order by age_value) as rn
  FROM rawData
  group by gender_concept_id, age_value
),
ageStatsPrior (gender_concept_id, age_value, total, accumulated) as
(
  select s.gender_concept_id, s.age_value, s.total, sum(p.total) as accumulated
  from ageStats s
  join ageStats p on s.gender_concept_id = p.gender_concept_id and p.rn <= s.rn
  group by s.gender_concept_id, s.age_value, s.total, s.rn
)
 SELECT
104 as analysis_id,
  CAST(o.gender_concept_id AS VARCHAR(255)) as stratum_1,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then age_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then age_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then age_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then age_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then age_value end) as p90_value
FROM
ageStatsPrior p
join overallStats o on p.gender_concept_id = o.gender_concept_id
GROUP BY o.gender_concept_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_104
;
--HINT DISTRIBUTE_ON_KEY(stratum_1) 
CREATE TEMP TABLE s_tmpach_dist_104
AS
SELECT
analysis_id, stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_104
;
ANALYZE s_tmpach_dist_104
;
truncate table tempResults_104;
drop table tempResults_104;


-- 105	Length of observation (days) of first observation period
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempObs_105
AS
SELECT
count_value, rn 
FROM
(
  select (CAST(op.observation_period_end_date AS DATE) - CAST(op.observation_period_start_date AS DATE)) as count_value,
	  ROW_NUMBER() over (PARTITION by op.person_id order by op.observation_period_start_date asc) as rn
  from omop.cmd.observation_period op
) A
where rn = 1;
ANALYZE tempObs_105
;
CREATE TEMP TABLE statsView_105
AS
SELECT
count_value, COUNT(*) as total, row_number() over (order by count_value) as rn
FROM
tempObs_105
group by count_value;
ANALYZE statsView_105
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_105
AS
WITH overallStats (avg_value, stdev_value, min_value, max_value, total)  AS (
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
  CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
  min(count_value) as min_value,
  max(count_value) as max_value,
  COUNT(*) as total
  from tempObs_105
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView_105 s
  join statsView_105 p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
105 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_105
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_105
AS
SELECT
analysis_id,
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5, count_value,
min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_105
;
ANALYZE s_tmpach_dist_105
;
truncate table tempObs_105;
drop table tempObs_105;
truncate table statsView_105;
drop table statsView_105;
truncate table tempResults_105;
drop table tempResults_105;


-- 106	Length of observation (days) of first observation period by gender
--HINT DISTRIBUTE_ON_KEY(gender_concept_id)
CREATE TEMP TABLE rawData_106
AS
SELECT
p.gender_concept_id, op.count_value
FROM
(
  select person_id, (CAST(op.observation_period_end_date AS DATE) - CAST(op.observation_period_start_date AS DATE)) as count_value,
    ROW_NUMBER() over (PARTITION by op.person_id order by op.observation_period_start_date asc) as rn
  from omop.cmd.observation_period op
) op
JOIN omop.cmd.person p on op.person_id = p.person_id
where op.rn = 1
;
ANALYZE rawData_106
;
--HINT DISTRIBUTE_ON_KEY(gender_concept_id)
CREATE TEMP TABLE tempResults_106
AS
WITH overallStats (gender_concept_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select gender_concept_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_106
  group by gender_concept_id
),
statsView (gender_concept_id, count_value, total, rn) as
(
  select gender_concept_id, count_value, COUNT(*) as total, row_number() over (order by count_value) as rn
  FROM rawData_106
  group by gender_concept_id, count_value
),
priorStats (gender_concept_id,count_value, total, accumulated) as
(
  select s.gender_concept_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.gender_concept_id = p.gender_concept_id and p.rn <= s.rn
  group by s.gender_concept_id, s.count_value, s.total, s.rn
)
 SELECT
106 as analysis_id,
  CAST(o.gender_concept_id AS VARCHAR(255)) as gender_concept_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value end) as p90_value
FROM
priorStats p
join overallStats o on p.gender_concept_id = o.gender_concept_id
GROUP BY o.gender_concept_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_106
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_106
AS
SELECT
analysis_id, gender_concept_id as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_106
;
ANALYZE s_tmpach_dist_106
;
truncate table rawData_106;
drop table rawData_106;
truncate table tempResults_106;
drop table tempResults_106;


-- 107	Length of observation (days) of first observation period by age decile
--HINT DISTRIBUTE_ON_KEY(age_decile)
CREATE TEMP TABLE tempResults_107
AS
WITH rawData (age_decile, count_value)  AS (
  select floor((EXTRACT(YEAR FROM op.OBSERVATION_PERIOD_START_DATE) - p.YEAR_OF_BIRTH)/10) as age_decile,
    (CAST(op.observation_period_end_date AS DATE) - CAST(op.observation_period_start_date AS DATE)) as count_value
  FROM
  (
    select person_id, 
  		op.observation_period_start_date,
  		op.observation_period_end_date,
      ROW_NUMBER() over (PARTITION by op.person_id order by op.observation_period_start_date asc) as rn
    from omop.cmd.observation_period op
  ) op
  JOIN omop.cmd.person p on op.person_id = p.person_id
  where op.rn = 1
),
overallStats (age_decile, avg_value, stdev_value, min_value, max_value, total) as
(
  select age_decile,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
  group by age_decile
),
statsView (age_decile, count_value, total, rn) as
(
  select age_decile,
    count_value, 
		COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by age_decile, count_value
),
priorStats (age_decile,count_value, total, accumulated) as
(
  select s.age_decile, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.age_decile = p.age_decile and p.rn <= s.rn
  group by s.age_decile, s.count_value, s.total, s.rn
)
 SELECT
107 as analysis_id,
  CAST(o.age_decile AS VARCHAR(255)) as age_decile,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.age_decile = o.age_decile
GROUP BY o.age_decile, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_107
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_107
AS
SELECT
analysis_id, age_decile as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_107
;
ANALYZE s_tmpach_dist_107
;
truncate table tempResults_107;
drop table tempResults_107;


-- 108	Number of persons by length of observation period, in 30d increments
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_108
AS
WITH rawData  AS (
  select
    floor((CAST(op1.observation_period_end_date AS DATE) - CAST(op1.observation_period_start_date AS DATE))/30) as stratum_1,
    COUNT(distinct p1.person_id) as count_value
  from omop.cmd.person p1
    inner join
    (select person_id,
      OBSERVATION_PERIOD_START_DATE,
      OBSERVATION_PERIOD_END_DATE,
      ROW_NUMBER() over (PARTITION by person_id order by observation_period_start_date asc) as rn1
       from omop.cmd.observation_period
    ) op1
    on p1.PERSON_ID = op1.PERSON_ID
    where op1.rn1 = 1
  group by floor((CAST(op1.observation_period_end_date AS DATE) - CAST(op1.observation_period_start_date AS DATE))/30)
)
 SELECT
108 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(null as varchar(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_108
;


-- 109	Number of persons with continuous observation in each year
--HINT DISTRIBUTE_ON_KEY(stratum_1)
-- generating date key sequences in a cross-dialect compatible fashion
CREATE TEMP TABLE s_tmpach_109
AS
WITH century   AS (SELECT  CAST('19' as TEXT) num union select '20' num), 
tens as (select '0' num union select '1' num union select '2' num union select '3' num union select '4' num union select '5' num union select '6' num union select '7' num union select '8' num union select '9' num),
ones as (select '0' num union select '1' num union select '2' num union select '3' num union select '4' num union select '5' num union select '6' num union select '7' num union select '8' num union select '9' num),
months as (select '01' as num union select '02' num union select '03' num union select '04' num union select '05' num union select '06' num union select '07' num union select '08' num union select '09' num union select '10' num union select '11' num union select '12' num),
date_keys as (select concat(century.num, tens.num, ones.num,months.num)  obs_month from century cross join tens cross join ones cross join months),
-- From date_keys, we just need each year and the first and last day of each year
ymd as (
select cast(left(obs_month,4) as integer)               as obs_year,
       min(cast(right(left(obs_month,6),2) as integer)) as month_start,
       1                                                as day_start,
       max(cast(right(left(obs_month,6),2) as integer)) as month_end,
       31                                               as day_end
  from date_keys
 where right(left(obs_month,6),2) in ('01','12')
 group by left(obs_month,4)
),
-- This gives us each year and the first and last day of the year 
year_ranges as (
select obs_year,
       TO_DATE(TO_CHAR(obs_year,'0000')||'-'||TO_CHAR(month_start,'00')||'-'||TO_CHAR(day_start,'00'), 'YYYY-MM-DD') obs_year_start,
       TO_DATE(TO_CHAR(obs_year,'0000')||'-'||TO_CHAR(month_end,'00')||'-'||TO_CHAR(day_end,'00'), 'YYYY-MM-DD') obs_year_end
  from ymd
 where obs_year >= (select min(EXTRACT(YEAR FROM observation_period_start_date)) from omop.cmd.observation_period)
   and obs_year <= (select max(EXTRACT(YEAR FROM observation_period_start_date)) from omop.cmd.observation_period)
) 
 SELECT
109                               AS analysis_id,  
	CAST(yr.obs_year AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255))        AS stratum_2, 
	CAST(NULL AS VARCHAR(255))        AS stratum_3, 
	CAST(NULL AS VARCHAR(255))        AS stratum_4, 
	CAST(NULL AS VARCHAR(255))        AS stratum_5,
	COUNT(DISTINCT op.person_id)  AS count_value
FROM
omop.cmd.observation_period op
CROSS JOIN 
	year_ranges yr
WHERE
	op.observation_period_start_date <= yr.obs_year_start
AND
	op.observation_period_end_date   >= yr.obs_year_end
GROUP BY 
	yr.obs_year
;
ANALYZE s_tmpach_109
;


-- 110	Number of persons with continuous observation in each month
-- Note: using temp table instead of nested query because this gives vastly improved performance in Oracle
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_110
AS
SELECT
110 as analysis_id,  
	CAST(t1.obs_month AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct op1.PERSON_ID) as count_value
FROM
omop.cmd.observation_period op1
join 
(
  SELECT DISTINCT 
    EXTRACT(YEAR FROM observation_period_start_date)*100 + EXTRACT(MONTH FROM observation_period_start_date) AS obs_month,
    TO_DATE(TO_CHAR(EXTRACT(YEAR FROM observation_period_start_date),'0000')||'-'||TO_CHAR(EXTRACT(MONTH FROM observation_period_start_date),'00')||'-'||TO_CHAR(1,'00'), 'YYYY-MM-DD')
    AS obs_month_start,
    (DATE_TRUNC('MONTH', observation_period_start_date) + INTERVAL '1 MONTH - 1 day')::DATE AS obs_month_end
  FROM omop.cmd.observation_period
) t1 on	op1.observation_period_start_date <= t1.obs_month_start
	and	op1.observation_period_end_date >= t1.obs_month_end
group by t1.obs_month;
ANALYZE s_tmpach_110
;


-- 111	Number of persons by observation period start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_111
AS
WITH rawData  AS (
  select
    EXTRACT(YEAR FROM observation_period_start_date)*100 + EXTRACT(MONTH FROM OBSERVATION_PERIOD_START_DATE) as stratum_1,
    COUNT(distinct op1.PERSON_ID) as count_value
  from
    omop.cmd.observation_period op1
  group by EXTRACT(YEAR FROM observation_period_start_date)*100 + EXTRACT(MONTH FROM OBSERVATION_PERIOD_START_DATE)
)
 SELECT
111 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(null as varchar(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_111
;


-- 112	Number of persons by observation period end month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_112
AS
WITH rawData  AS (
  select
    EXTRACT(YEAR FROM observation_period_end_date)*100 + EXTRACT(MONTH FROM observation_period_end_date) as stratum_1,
    COUNT(distinct op1.PERSON_ID) as count_value
  from
    omop.cmd.observation_period op1
  group by EXTRACT(YEAR FROM observation_period_end_date)*100 + EXTRACT(MONTH FROM observation_period_end_date)
)
 SELECT
112 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(null as varchar(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_112
;


-- 113	Number of persons by number of observation periods
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_113
AS
SELECT
113 as analysis_id,  
	CAST(op1.num_periods AS VARCHAR(255)) as stratum_1, 
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct op1.PERSON_ID) as count_value
FROM
(select person_id, COUNT(OBSERVATION_period_start_date) as num_periods from omop.cmd.observation_period group by PERSON_ID) op1
group by op1.num_periods
;
ANALYZE s_tmpach_113
;


-- 114	Number of persons with observation period before year-of-birth
CREATE TEMP TABLE s_tmpach_114
AS
SELECT
114 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.PERSON_ID) as count_value
FROM
omop.cmd.person p1
	inner join (select person_id, MIN(EXTRACT(YEAR FROM OBSERVATION_period_start_date)) as first_obs_year from omop.cmd.observation_period group by PERSON_ID) op1
	on p1.person_id = op1.person_id
where p1.year_of_birth > op1.first_obs_year
;
ANALYZE s_tmpach_114
;


-- 115	Number of persons with observation period end < start
CREATE TEMP TABLE s_tmpach_115
AS
SELECT
115 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(op1.PERSON_ID) as count_value
FROM
omop.cmd.observation_period op1
where op1.observation_period_end_date < op1.observation_period_start_date
;
ANALYZE s_tmpach_115
;


-- 116	Number of persons with at least one day of observation in each year by gender and age decile
-- Note: using temp table instead of nested query because this gives vastly improved performance in Oracle
CREATE TEMP TABLE temp_dates_116
AS
SELECT
distinct 
  EXTRACT(YEAR FROM observation_period_start_date) as obs_year 
FROM
omop.cmd.observation_period
;
ANALYZE temp_dates_116
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_116
AS
WITH rawData  AS (
  select
    t1.obs_year as stratum_1,
    p1.gender_concept_id as stratum_2,
    floor((t1.obs_year - p1.year_of_birth)/10) as stratum_3,
    COUNT(distinct p1.PERSON_ID) as count_value
  from
    omop.cmd.person p1
    inner join
    omop.cmd.observation_period op1
    on p1.person_id = op1.person_id
    ,
    temp_dates_116 t1
  where EXTRACT(YEAR FROM op1.OBSERVATION_PERIOD_START_DATE) <= t1.obs_year
    and EXTRACT(YEAR FROM op1.OBSERVATION_PERIOD_END_DATE) >= t1.obs_year
  group by t1.obs_year,
    p1.gender_concept_id,
    floor((t1.obs_year - p1.year_of_birth)/10)
)
 SELECT
116 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(stratum_2 as varchar(255)) as stratum_2,
  cast(stratum_3 as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_116
;
TRUNCATE TABLE temp_dates_116;
DROP TABLE temp_dates_116;


-- 117	Number of persons with at least one day of observation in each month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
-- generating date key sequences in a cross-dialect compatible fashion
CREATE TEMP TABLE s_tmpach_117	
AS
WITH century   AS (SELECT  CAST('19' as TEXT) num union select '20' num), 
tens as (select '0' num union select '1' num union select '2' num union select '3' num union select '4' num union select '5' num union select '6' num union select '7' num union select '8' num union select '9' num),
ones as (select '0' num union select '1' num union select '2' num union select '3' num union select '4' num union select '5' num union select '6' num union select '7' num union select '8' num union select '9' num),
months as (select '01' as num union select '02' num union select '03' num union select '04' num union select '05' num union select '06' num union select '07' num union select '08' num union select '09' num union select '10' num union select '11' num union select '12' num),
date_keys as (select cast(concat(century.num, tens.num, ones.num,months.num) as int) obs_month from century cross join tens cross join ones cross join months)
 SELECT
117 as analysis_id,  
	CAST(t1.obs_month AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COALESCE(COUNT(distinct op1.PERSON_ID),0) as count_value
FROM
date_keys t1
left join
  (select t2.obs_month, op2.*
    from omop.cmd.observation_period op2, date_keys t2
    where EXTRACT(YEAR FROM op2.observation_period_start_date)*100 + EXTRACT(MONTH FROM op2.observation_period_start_date) <= t2.obs_month
    and EXTRACT(YEAR FROM op2.observation_period_end_date)*100 + EXTRACT(MONTH FROM op2.observation_period_end_date) >= t2.obs_month
  ) op1 on op1.obs_month = t1.obs_month
group by t1.obs_month
having COALESCE(COUNT(distinct op1.PERSON_ID),0) > 0;
ANALYZE s_tmpach_117	
;


-- 118  Number of observation period records with invalid person_id
CREATE TEMP TABLE s_tmpach_118
AS
SELECT
118 as analysis_id,
  cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
  COUNT(op1.PERSON_ID) as count_value
FROM
omop.cmd.observation_period op1
  left join omop.cmd.person p1
  on p1.person_id = op1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_118
;


-- 119  Number of observation period records by period_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_119
AS
SELECT
119 as analysis_id,
  CAST(op1.period_type_concept_id AS VARCHAR(255)) as stratum_1,
  cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
  COUNT(*) as count_value
FROM
omop.cmd.observation_period op1
group by op1.period_type_concept_id
;
ANALYZE s_tmpach_119
;


-- 200	Number of persons with at least one visit occurrence, by visit_concept_id
-- restricted to visits overlapping with observation period
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_200
AS
SELECT
200 AS analysis_id,
	CAST(vo.visit_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT vo.person_id) AS count_value
FROM
omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	vo.visit_concept_id;
ANALYZE s_tmpach_200
;


-- 201	Number of visit occurrence records, by visit_concept_id
-- restricted to visits overlapping with observation period
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_201
AS
SELECT
201 AS analysis_id,
	CAST(vo.visit_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(vo.person_id) AS count_value
FROM
omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	vo.visit_concept_id;
ANALYZE s_tmpach_201
;


-- 202	Number of persons by visit occurrence start month, by visit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_202
AS
WITH rawData  AS (
SELECT 
	vo.visit_concept_id AS stratum_1,
	EXTRACT(YEAR FROM vo.visit_start_date) * 100 + EXTRACT(MONTH FROM vo.visit_start_date) AS stratum_2,
	COUNT(DISTINCT vo.person_id) AS count_value
FROM 
	omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	vo.visit_concept_id,
	EXTRACT(YEAR FROM vo.visit_start_date) * 100 + EXTRACT(MONTH FROM vo.visit_start_date)
)
 SELECT
202 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_202
;


-- 203	Number of distinct visit occurrence concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_203
AS
WITH rawData(person_id, count_value)  AS (
SELECT 
	vo.person_id,
	COUNT(DISTINCT vo.visit_concept_id) AS count_value
FROM 
	omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	vo.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
203 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_203
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_203
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_203
;
ANALYZE s_tmpach_dist_203
;
truncate table tempResults_203;
drop table tempResults_203;


-- 204	Number of persons with at least one visit occurrence, by visit_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_204
AS
WITH rawData  AS (
SELECT 
	vo.visit_concept_id AS stratum_1,
	EXTRACT(YEAR FROM vo.visit_start_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM vo.visit_start_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.visit_occurrence vo 
ON 
	p.person_id = vo.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	vo.visit_concept_id,
	EXTRACT(YEAR FROM vo.visit_start_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM vo.visit_start_date) - p.year_of_birth) / 10)
)
 SELECT
204 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 as varchar(255)) AS stratum_2,
	CAST(stratum_3 as varchar(255)) AS stratum_3,
	CAST(stratum_4 as varchar(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_204
;


-- 206	Distribution of age by visit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_206
AS
WITH rawData (stratum1_id, stratum2_id, count_value)  AS (
SELECT 
	v.visit_concept_id AS stratum1_id,
	p.gender_concept_id AS stratum2_id,
	v.visit_start_year - p.year_of_birth AS count_value
FROM 
	omop.cmd.person p
JOIN (
	SELECT 
		vo.person_id,
		vo.visit_concept_id,
		MIN(EXTRACT(YEAR FROM vo.visit_start_date)) AS visit_start_year
	FROM 
		omop.cmd.visit_occurrence vo
	JOIN 
		omop.cmd.observation_period op 
	ON 
		vo.person_id = op.person_id
	AND 
		vo.visit_start_date >= op.observation_period_start_date
	AND 
		vo.visit_start_date <= op.observation_period_end_date
	GROUP BY 
		vo.person_id,
		vo.visit_concept_id
	) v
ON 
	p.person_id = v.person_id
),
overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum1_id,
    stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData
	group by stratum1_id, stratum2_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select stratum1_id, stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by stratum1_id, stratum2_id order by count_value) as rn
  FROM rawData
  group by stratum1_id, stratum2_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
206 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_206
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_206
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_206
;
ANALYZE s_tmpach_dist_206
;
truncate table tempResults_206;
drop table tempResults_206;


--207	Number of visit records with invalid person_id
CREATE TEMP TABLE s_tmpach_207
AS
SELECT
207 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(vo1.PERSON_ID) as count_value
FROM
omop.cmd.visit_occurrence vo1
	left join omop.cmd.person p1
	on p1.person_id = vo1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_207
;


-- 209	Number of visit records with invalid care_site_id
CREATE TEMP TABLE s_tmpach_209
AS
SELECT
209 as analysis_id,
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(vo1.PERSON_ID) as count_value
FROM
omop.cmd.visit_occurrence vo1
	left join omop.cmd.care_site cs1
	on vo1.care_site_id = cs1.care_site_id
where vo1.care_site_id is not null
	and cs1.care_site_id is null
;
ANALYZE s_tmpach_209
;


-- 210 Number of visit_occurrence records outside a valid observation period
CREATE TEMP TABLE s_tmpach_210
AS
SELECT
210 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.visit_occurrence vo
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
WHERE 
	op.person_id IS NULL
;
ANALYZE s_tmpach_210
;


-- 211	Number of visit records with end date < start date
CREATE TEMP TABLE s_tmpach_211
AS
SELECT
211 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(vo1.PERSON_ID) as count_value
FROM
omop.cmd.visit_occurrence vo1
where visit_end_date < visit_start_date
;
ANALYZE s_tmpach_211
;


-- 212	Number of persons with at least one visit occurrence by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_212
AS
WITH rawData  AS (
  select
    EXTRACT(YEAR FROM visit_start_date) as stratum_1,
    p1.gender_concept_id as stratum_2,
    floor((EXTRACT(YEAR FROM visit_start_date) - p1.year_of_birth)/10) as stratum_3,
    COUNT(distinct p1.PERSON_ID) as count_value
  from omop.cmd.person p1
  inner join
  omop.cmd.visit_occurrence vo1
  on p1.person_id = vo1.person_id
  group by
    EXTRACT(YEAR FROM visit_start_date),
    p1.gender_concept_id,
    floor((EXTRACT(YEAR FROM visit_start_date) - p1.year_of_birth)/10)
)
 SELECT
212 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(stratum_2 as varchar(255)) as stratum_2,
  cast(stratum_3 as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_212
;


-- 213	Distribution of length of stay by visit_concept_id
-- restrict to visits inside observation period
--HINT DISTRIBUTE_ON_KEY(stratum_id) 
CREATE TEMP TABLE tempResults_213
AS
WITH rawData(stratum_id, count_value)  AS (
  select visit_concept_id AS stratum_id, (CAST(visit_end_date AS DATE) - CAST(visit_start_date AS DATE)) as count_value
  from omop.cmd.visit_occurrence vo inner join 
  omop.cmd.observation_period op on vo.person_id = op.person_id
  -- only include events that occur during observation period
  where vo.visit_start_date >= op.observation_period_start_date and
  COALESCE(vo.visit_end_date,vo.visit_start_date) <= op.observation_period_end_date
),
overallStats (stratum_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData
  group by stratum_id
),
statsView (stratum_id, count_value, total, rn) as
(
  select stratum_id, count_value, COUNT(*) as total, row_number() over (order by count_value) as rn
  FROM rawData
  group by stratum_id, count_value
),
priorStats (stratum_id, count_value, total, accumulated) as
(
  select s.stratum_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum_id = p.stratum_id and p.rn <= s.rn
  group by s.stratum_id, s.count_value, s.total, s.rn
)
 SELECT
213 as analysis_id,
  CAST(o.stratum_id AS VARCHAR(255)) AS stratum_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum_id = o.stratum_id
GROUP BY o.stratum_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_213
;
--HINT DISTRIBUTE_ON_KEY(stratum_1) 
CREATE TEMP TABLE s_tmpach_dist_213
AS
SELECT
analysis_id, stratum_id as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_213
;
ANALYZE s_tmpach_dist_213
;
truncate table tempResults_213;
drop table tempResults_213;


-- 220	Number of visit occurrence records by condition occurrence start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_220
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM vo.visit_start_date) * 100 + EXTRACT(MONTH FROM vo.visit_start_date) AS stratum_1,
	COUNT(vo.person_id) AS count_value
FROM 
	omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM vo.visit_start_date) * 100 + EXTRACT(MONTH FROM vo.visit_start_date)
)
 SELECT
220 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_220
;


-- 221	Number of persons by visit start year 
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_221
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM vo.visit_start_date) AS stratum_1,
	COUNT(DISTINCT vo.person_id) AS count_value
FROM 
	omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM vo.visit_start_date)
)
 SELECT
221 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_221
;


-- 225	Number of visit_occurrence records, by visit_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_225
AS
SELECT
225 AS analysis_id,
	CAST(vo.visit_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
GROUP BY 
	visit_source_concept_id;
ANALYZE s_tmpach_225
;


-- 226	Number of records by domain by visit concept id
CREATE TEMP TABLE s_tmpach_226
AS
SELECT
226 as analysis_id, 
	CAST(v.visit_concept_id AS VARCHAR(255)) as stratum_1,
	v.cdm_table as stratum_2,
	cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	v.record_count as count_value
FROM
(
  select 'drug_exposure' cdm_table, coalesce(visit_concept_id,0) visit_concept_id, count(*) record_count
  from omop.cmd.drug_exposure t
  left join omop.cmd.visit_occurrence v on t.visit_occurrence_id = v.visit_occurrence_id
  group by visit_concept_id
  union
  select 'condition_occurrence' cdm_table, coalesce(visit_concept_id,0) visit_concept_id, count(*) record_count
  from omop.cmd.condition_occurrence t
  left join omop.cmd.visit_occurrence v on t.visit_occurrence_id = v.visit_occurrence_id
  group by visit_concept_id
  union
  select 'device_exposure' cdm_table, coalesce(visit_concept_id,0) visit_concept_id, count(*) record_count
  from omop.cmd.device_exposure t
  left join omop.cmd.visit_occurrence v on t.visit_occurrence_id = v.visit_occurrence_id
  group by visit_concept_id
  union
  select 'procedure_occurrence' cdm_table, coalesce(visit_concept_id,0) visit_concept_id, count(*) record_count
  from omop.cmd.procedure_occurrence t
  left join omop.cmd.visit_occurrence v on t.visit_occurrence_id = v.visit_occurrence_id
  group by visit_concept_id
  union
  select 'measurement' cdm_table, coalesce(visit_concept_id,0) visit_concept_id, count(*) record_count
  from omop.cmd.measurement t
  left join omop.cmd.visit_occurrence v on t.visit_occurrence_id = v.visit_occurrence_id
  group by visit_concept_id
  union
  select 'observation' cdm_table, coalesce(visit_concept_id,0) visit_concept_id, count(*) record_count
  from omop.cmd.observation t
  left join omop.cmd.visit_occurrence v on t.visit_occurrence_id = v.visit_occurrence_id
  group by visit_concept_id
) v
;
ANALYZE s_tmpach_226
;


-- 230 Number of visit_occurrence records inside a valid observation period
CREATE TEMP TABLE s_tmpach_230
AS
SELECT
230 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.visit_occurrence vo
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_230
;


-- 231	Proportion of people with at least one visit_occurrence record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in visit_occurrence (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_231
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT vo.person_id) AS person_count
FROM 
	omop.cmd.visit_occurrence vo
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), vo_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.visit_occurrence
)
 SELECT
231 AS analysis_id,
	CASE WHEN vo.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/vo.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(vo.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	vo_total vo
;
ANALYZE s_tmpach_231
;


-- 232	Proportion of visit_occurrence records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of visit_occurrence records outside a valid observation period (numerator)
-- stratum_3:   Number of visit_occurrence records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_232
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.visit_occurrence vo
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), vo_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.visit_occurrence
)
 SELECT
232 AS analysis_id,
	CASE WHEN vo.record_count != 0 THEN
		CAST(CAST(1.0*op.record_count/vo.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
	    CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(vo.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	vo_total vo
;
ANALYZE s_tmpach_232
;


-- 300	Number of providers
CREATE TEMP TABLE s_tmpach_300
AS
SELECT
300 as analysis_id,  
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct provider_id) as count_value
FROM
omop.cmd.provider;
ANALYZE s_tmpach_300
;


-- 301	Number of providers by specialty concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_301
AS
SELECT
301 as analysis_id,
CAST(specialty_concept_id AS VARCHAR(255)) as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(distinct provider_id) as count_value
FROM
omop.cmd.provider
group by specialty_CONCEPT_ID;
ANALYZE s_tmpach_301
;


-- 303	Number of provider records, by specialty_concept_id, visit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_303 
AS
SELECT
303 as analysis_id,
       cast(p.specialty_concept_id AS varchar(255)) AS stratum_1,
       cast(vo.visit_concept_id    AS varchar(255)) AS stratum_2,
       cast(null as varchar(255)) as stratum_3,
       cast(null as varchar(255)) as stratum_4,
       cast(null as varchar(255)) as stratum_5, 
       COUNT(*) AS count_value
FROM
omop.cmd.provider p
  join omop.cmd.visit_occurrence vo
    on vo.provider_id = p.provider_id
 group by p.specialty_concept_id, visit_concept_id;
ANALYZE s_tmpach_303 
  ;


-- 325	Number of provider records, by specialty_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_325 
AS
SELECT
325 as analysis_id,
       cast(specialty_source_concept_id AS varchar(255)) AS stratum_1,
       cast(null AS varchar(255)) AS stratum_2,
       cast(null as varchar(255)) as stratum_3,
       cast(null as varchar(255)) as stratum_4,
       cast(null as varchar(255)) as stratum_5, 
       COUNT(*) AS count_value
FROM
omop.cmd.provider
 group by specialty_source_concept_id;
ANALYZE s_tmpach_325 
  ;


-- 400	Number of persons with at least one condition occurrence, by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_400
AS
SELECT
400 AS analysis_id,
	CAST(co.condition_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT co.person_id) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_concept_id;
ANALYZE s_tmpach_400
;


-- 401	Number of condition occurrence records, by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_401
AS
SELECT
401 AS analysis_id,
	CAST(co.condition_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(co.person_id) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_concept_id;
ANALYZE s_tmpach_401
;


-- 402	Number of persons by condition occurrence start month, by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_402
AS
WITH rawData  AS (
SELECT 
	co.condition_concept_id AS stratum_1,
	EXTRACT(YEAR FROM co.condition_start_date) * 100 + EXTRACT(MONTH FROM co.condition_start_date) AS stratum_2,
	COUNT(DISTINCT co.person_id) AS count_value
FROM 
	omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_concept_id,
	EXTRACT(YEAR FROM co.condition_start_date) * 100 + EXTRACT(MONTH FROM co.condition_start_date)
)
 SELECT
402 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_402
;


-- 403	Number of distinct condition occurrence concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_403
AS
WITH rawData(person_id, count_value)  AS (
SELECT 
	co.person_id,
	COUNT(DISTINCT co.condition_concept_id) AS count_value
FROM 
	omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
403 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_403
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_403
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_403
;
ANALYZE s_tmpach_dist_403
;
truncate table tempResults_403;
drop table tempResults_403;


-- 404	Number of persons with at least one condition occurrence, by condition_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_404
AS
WITH rawData  AS (
SELECT 
	co.condition_concept_id AS stratum_1,
	EXTRACT(YEAR FROM co.condition_start_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM co.condition_start_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.condition_occurrence co 
ON 
	p.person_id = co.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_concept_id,
	EXTRACT(YEAR FROM co.condition_start_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM co.condition_start_date) - p.year_of_birth) / 10)
)
 SELECT
404 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_404
;


-- 405	Number of condition occurrence records, by condition_concept_id by condition_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_405
AS
SELECT
405 AS analysis_id,
	CAST(co.condition_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(co.condition_type_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(co.person_id) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_CONCEPT_ID,
	co.condition_type_concept_id;
ANALYZE s_tmpach_405
;


-- 406	Distribution of age by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_406
AS
SELECT
c.condition_concept_id AS subject_id,
	p.gender_concept_id,
	(c.condition_start_year - p.year_of_birth) AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		co.person_id,
		co.condition_concept_id,
		MIN(EXTRACT(YEAR FROM co.condition_start_date)) AS condition_start_year
	FROM 
		omop.cmd.condition_occurrence co
	JOIN 
		omop.cmd.observation_period op 
	ON 
		co.person_id = op.person_id
	AND 
		co.condition_start_date >= op.observation_period_start_date
	AND 
		co.condition_start_date <= op.observation_period_end_date
	GROUP BY 
		co.person_id,
		co.condition_concept_id
	) c 
ON 
	p.person_id = c.person_id;
ANALYZE rawData_406
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_406
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_406
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_406
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
406 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_406
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_406
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_406
;
ANALYZE s_tmpach_dist_406
;
truncate table tempResults_406;
drop table tempResults_406;
truncate Table rawData_406;
drop table rawData_406;


-- 409	Number of condition occurrence records with invalid person_id
CREATE TEMP TABLE s_tmpach_409
AS
SELECT
409 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(co1.PERSON_ID) as count_value
FROM
omop.cmd.condition_occurrence co1
	left join omop.cmd.person p1
	on p1.person_id = co1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_409
;


-- 410	Number of condition occurrence records outside valid observation period
CREATE TEMP TABLE s_tmpach_410
AS
SELECT
410 as analysis_id,
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(co1.PERSON_ID) as count_value
FROM
omop.cmd.condition_occurrence co1
	left join omop.cmd.observation_period op1
	on op1.person_id = co1.person_id
	and co1.condition_start_date >= op1.observation_period_start_date
	and co1.condition_start_date <= op1.observation_period_end_date
where op1.person_id is null
;
ANALYZE s_tmpach_410
;


-- 411	Number of condition occurrence records with end date < start date
CREATE TEMP TABLE s_tmpach_411
AS
SELECT
411 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(co1.PERSON_ID) as count_value
FROM
omop.cmd.condition_occurrence co1
where co1.condition_end_date < co1.condition_start_date
;
ANALYZE s_tmpach_411
;


-- 412	Number of condition occurrence records with invalid provider_id
CREATE TEMP TABLE s_tmpach_412
AS
SELECT
412 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(co1.PERSON_ID) as count_value
FROM
omop.cmd.condition_occurrence co1
	left join omop.cmd.provider p1
	on p1.provider_id = co1.provider_id
where co1.provider_id is not null
	and p1.provider_id is null
;
ANALYZE s_tmpach_412
;


-- 413	Number of condition occurrence records with invalid visit_id
CREATE TEMP TABLE s_tmpach_413
AS
SELECT
413 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(co1.PERSON_ID) as count_value
FROM
omop.cmd.condition_occurrence co1
	left join omop.cmd.visit_occurrence vo1
	on co1.visit_occurrence_id = vo1.visit_occurrence_id
where co1.visit_occurrence_id is not null
	and vo1.visit_occurrence_id is null
;
ANALYZE s_tmpach_413
;


-- 414	Number of condition occurrence records, by condition_status_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_414
AS
SELECT
414 AS analysis_id,
	CAST(co.condition_status_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_status_concept_id;
ANALYZE s_tmpach_414
;


-- 415	Number of condition occurrence records, by condition_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_415
AS
SELECT
415 AS analysis_id,
	CAST(co.condition_type_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_type_concept_id;
ANALYZE s_tmpach_415
;


-- 416	Number of condition occurrence records, by condition_status_concept_id, condition_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_416
AS
SELECT
416 AS analysis_id,
	CAST(co.condition_status_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(co.condition_type_concept_id AS VARCHAR(255))   AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_status_concept_id, co.condition_type_concept_id;
ANALYZE s_tmpach_416
;


-- 420	Number of condition occurrence records by condition occurrence start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_420
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM co.condition_start_date) * 100 + EXTRACT(MONTH FROM co.condition_start_date) AS stratum_1,
	COUNT(co.person_id) AS count_value
FROM 
	omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM co.condition_start_date) * 100 + EXTRACT(MONTH FROM co.condition_start_date)
)
 SELECT
420 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_420
;


-- 425	Number of condition_occurrence records, by condition_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_425
AS
SELECT
425 AS analysis_id,
	CAST(co.condition_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
GROUP BY 
	co.condition_source_concept_id;
ANALYZE s_tmpach_425
;


-- 430	Number of condition occurrence records inside a valid observation period
CREATE TEMP TABLE s_tmpach_430
AS
SELECT
430 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.condition_occurrence co
JOIN 
	omop.cmd.observation_period op 
ON 
	op.person_id = co.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_430
;


-- 431	Proportion of people with at least one condition_occurrence record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in condition_occurrence (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_431
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT co.person_id) AS person_count
FROM 
	omop.cmd.condition_occurrence co
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), co_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.condition_occurrence
)
 SELECT
431 AS analysis_id,
	CASE WHEN co.person_count != 0 THEN
		CAST(CAST(1.0*op.person_count/co.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(co.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	co_total co
;
ANALYZE s_tmpach_431
;


-- 432	Proportion of condition_occurrence records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of condition_occurrence records outside a valid observation period (numerator)
-- stratum_3:   Number of condition_occurrence records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_432
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.condition_occurrence co
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	co.person_id = op.person_id
AND 
	co.condition_start_date >= op.observation_period_start_date
AND 
	co.condition_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), co_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.condition_occurrence
)
 SELECT
432 AS analysis_id,
	CASE WHEN co.record_count != 0 THEN 
		CAST(CAST(1.0*op.record_count/co.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE
		CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(co.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	co_total co
;
ANALYZE s_tmpach_432
;


-- 500	Number of persons with death, by cause_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_500
AS
SELECT
500 AS analysis_id,
	CAST(d.cause_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT d.person_id) AS count_value
FROM
omop.cmd.death d
JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date	
GROUP BY 
	d.cause_concept_id;
ANALYZE s_tmpach_500
;


-- 501	Number of records of death, by cause_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_501
AS
SELECT
501 AS analysis_id,
	CAST(d.cause_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(d.person_id) AS count_value
FROM
omop.cmd.death d
JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date	
GROUP BY 
	d.cause_concept_id;
ANALYZE s_tmpach_501
;


-- 502	Number of persons by death month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_502
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM d.death_date) * 100 + EXTRACT(MONTH FROM d.death_date) AS stratum_1,
	COUNT(DISTINCT d.person_id) AS count_value
FROM 
	omop.cmd.death d
JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date	
GROUP BY 
	EXTRACT(YEAR FROM d.death_date) * 100 + EXTRACT(MONTH FROM d.death_date)
)
 SELECT
502 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(NULL AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_502
;


-- 504	Number of persons with a death, by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_504
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM d.death_date) AS stratum_1,
	p.gender_concept_id AS stratum_2,
	FLOOR((EXTRACT(YEAR FROM d.death_date) - p.year_of_birth) / 10) AS stratum_3,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.death d 
ON 
	p.person_id = d.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date	
GROUP BY 
	EXTRACT(YEAR FROM d.death_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM d.death_date) - p.year_of_birth) / 10)
)
 SELECT
504 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_504
;


-- 505	Number of death records, by death_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_505
AS
SELECT
505 AS analysis_id,
	CAST(d.death_type_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(d.person_id) AS count_value
FROM
omop.cmd.death d
JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date	
GROUP BY 
	d.death_type_concept_id;
ANALYZE s_tmpach_505
;


-- 506	Distribution of age at death by gender
--HINT DISTRIBUTE_ON_KEY(stratum_id)
CREATE TEMP TABLE tempResults_506
AS
WITH rawData(stratum_id, count_value)  AS (
SELECT 
	p.gender_concept_id AS stratum_id,
	d.death_year - p.year_of_birth AS count_value
FROM 
	omop.cmd.person p
JOIN (
	SELECT 
		d.person_id,
		MIN(EXTRACT(YEAR FROM d.death_date)) AS death_year
	FROM 
		omop.cmd.death d
	JOIN 
		omop.cmd.observation_period op 
	ON 
		d.person_id = op.person_id
	AND 
		d.death_date >= op.observation_period_start_date
	AND 
		d.death_date <= op.observation_period_end_date	
	GROUP BY 
		d.person_id
  ) d
ON 
	p.person_id = d.person_id
),
overallStats (stratum_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData
  group by stratum_id
),
statsView (stratum_id, count_value, total, rn) as
(
  select stratum_id, count_value, COUNT(*) as total, row_number() over (order by count_value) as rn
  FROM rawData
  group by stratum_id, count_value
),
priorStats (stratum_id, count_value, total, accumulated) as
(
  select s.stratum_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum_id = p.stratum_id and p.rn <= s.rn
  group by s.stratum_id, s.count_value, s.total, s.rn
)
 SELECT
506 as analysis_id,
  CAST(o.stratum_id AS VARCHAR(255)) AS stratum_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum_id = o.stratum_id
GROUP BY o.stratum_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_506
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_506
AS
SELECT
analysis_id, stratum_id as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_506
;
ANALYZE s_tmpach_dist_506
;
truncate table tempResults_506;
drop table tempResults_506;


-- 509	Number of death records with invalid person_id
CREATE TEMP TABLE s_tmpach_509
AS
SELECT
509 as analysis_id, 
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(d1.PERSON_ID) as count_value
FROM
omop.cmd.death d1
		left join omop.cmd.person p1
		on d1.person_id = p1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_509
;


-- 510	Number of death records outside valid observation period
CREATE TEMP TABLE s_tmpach_510
AS
SELECT
510 as analysis_id, 
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(d1.PERSON_ID) as count_value
FROM
omop.cmd.death d1
		left join omop.cmd.observation_period op1
		on d1.person_id = op1.person_id
		and d1.death_date >= op1.observation_period_start_date
		and d1.death_date <= op1.observation_period_end_date
where op1.person_id is null
;
ANALYZE s_tmpach_510
;


-- 511	Distribution of time from death to last condition
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_511
AS
SELECT
511 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(count_value) AS count_value,
	MIN(count_value) AS min_value,
	MAX(count_value) AS max_value,
	CAST(AVG(1.0 * count_value) AS NUMERIC) AS avg_value,
	CAST(STDDEV(count_value) AS NUMERIC) AS stdev_value,
	MAX(CASE WHEN p1 <= 0.50 THEN count_value ELSE - 9999 END) AS median_value,
	MAX(CASE WHEN p1 <= 0.10 THEN count_value ELSE - 9999 END) AS p10_value,
	MAX(CASE WHEN p1 <= 0.25 THEN count_value ELSE - 9999 END) AS p25_value,
	MAX(CASE WHEN p1 <= 0.75 THEN count_value ELSE - 9999 END) AS p75_value,
	MAX(CASE WHEN p1 <= 0.90 THEN count_value ELSE - 9999 END) AS p90_value
FROM
(
SELECT 
	(CAST(co.max_date AS DATE) - CAST(d.death_date AS DATE)) AS count_value,
	1.0 * (ROW_NUMBER() OVER (ORDER BY (CAST(co.max_date AS DATE) - CAST(d.death_date AS DATE)))) / (COUNT(*) OVER () + 1) AS p1
FROM 
	omop.cmd.death d
JOIN (
	SELECT 
		co.person_id,
		MAX(co.condition_start_date) AS max_date
	FROM 
		omop.cmd.condition_occurrence co
	JOIN 
		omop.cmd.observation_period op 
	ON 
		co.person_id = op.person_id
	AND 
		co.condition_start_date >= op.observation_period_start_date
	AND 
		co.condition_start_date <= op.observation_period_end_date	
	GROUP BY 
		co.person_id
	) co 
ON d.person_id = co.person_id
	) t1;
ANALYZE s_tmpach_dist_511
;


-- 512	Distribution of time from death to last drug
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_512
AS
WITH rawData(count_value)  AS (
SELECT 
	(CAST(de.max_date AS DATE) - CAST(d.death_date AS DATE)) AS count_value
FROM 
	omop.cmd.death d
JOIN (
	SELECT 
		de.person_id,
		MAX(de.drug_exposure_start_date) AS max_date
	FROM 
		omop.cmd.drug_exposure de
	JOIN 
		omop.cmd.observation_period op 
	ON 
		de.person_id = op.person_id
	AND 
		de.drug_exposure_start_date >= op.observation_period_start_date
	AND 
		de.drug_exposure_start_date <= op.observation_period_end_date	
	GROUP BY 
		de.person_id
	) de
ON 
	d.person_id = de.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
512 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_512
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_512
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_512
;
ANALYZE s_tmpach_dist_512
;
truncate table tempResults_512;
drop table tempResults_512;


-- 513	Distribution of time from death to last visit
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_513
AS
WITH rawData(count_value)  AS (
SELECT 
	(CAST(vo.max_date AS DATE) - CAST(d.death_date AS DATE)) AS count_value
FROM 
	omop.cmd.death d
JOIN (
	SELECT 
		vo.person_id,
		MAX(vo.visit_start_date) AS max_date
	FROM 
		omop.cmd.visit_occurrence vo
	JOIN 
		omop.cmd.observation_period op 
	ON 
		vo.person_id = op.person_id
	AND 
		vo.visit_start_date >= op.observation_period_start_date
	AND 
		vo.visit_start_date <= op.observation_period_end_date	
	GROUP BY 
		vo.person_id
	) vo
ON 
	d.person_id = vo.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
513 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_513
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_513
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_513
;
ANALYZE s_tmpach_dist_513
;
truncate table tempResults_513;
drop table tempResults_513;


-- 514	Distribution of time from death to last procedure
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_514
AS
WITH rawData(count_value)  AS (
SELECT 
	(CAST(po.max_date AS DATE) - CAST(d.death_date AS DATE)) AS count_value
FROM 
	omop.cmd.death d
JOIN (
	SELECT 
		po.person_id,
		MAX(po.procedure_date) AS max_date
	FROM 
		omop.cmd.procedure_occurrence po
	JOIN 
		omop.cmd.observation_period op 
	ON 
		po.person_id = op.person_id
	AND 
		po.procedure_date >= op.observation_period_start_date
	AND 
		po.procedure_date <= op.observation_period_end_date	
	GROUP BY 
		po.person_id
	) po
ON 
	d.person_id = po.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
514 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_514
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_514
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_514
;
ANALYZE s_tmpach_dist_514
;
truncate table tempResults_514;
drop table tempResults_514;


-- 515	Distribution of time from death to last observation
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_515
AS
WITH rawData(count_value)  AS (
SELECT 
	(CAST(o.max_date AS DATE) - CAST(d.death_date AS DATE)) AS count_value
FROM 
	omop.cmd.death d
JOIN (
	SELECT 
		o.person_id,
		MAX(o.observation_date) AS max_date
	FROM 
		omop.cmd.observation o
	JOIN 
		omop.cmd.observation_period op 
	ON 
		o.person_id = op.person_id
	AND 
		o.observation_date >= op.observation_period_start_date
	AND 
		o.observation_date <= op.observation_period_end_date	
	GROUP BY 
		o.person_id
	) o
ON 
	d.person_id = o.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
515 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_515
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_515
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_515
;
ANALYZE s_tmpach_dist_515
;
truncate table tempResults_515;
drop table tempResults_515;


-- 525	Number of death records, by cause_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_525 
AS
SELECT
525 as analysis_id,
       cast(cause_source_concept_id AS varchar(255)) AS stratum_1,
       cast(null AS varchar(255)) AS stratum_2,
       cast(null as varchar(255)) as stratum_3,
       cast(null as varchar(255)) as stratum_4,
       cast(null as varchar(255)) as stratum_5,
       COUNT(*) AS count_value
FROM
omop.cmd.death
 group by cause_source_concept_id;
ANALYZE s_tmpach_525 
  ;


-- 530	Number of death records inside a valid observation period
CREATE TEMP TABLE s_tmpach_530
AS
SELECT
530 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.death d
JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_530
;


-- 531	Proportion of people with at least one death record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in death (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_531
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT d.person_id) AS person_count
FROM 
	omop.cmd.death d
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), death_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.death
)
 SELECT
531 AS analysis_id,
	CASE WHEN dt.person_count != 0 THEN
		CAST(CAST(1.0*op.person_count/dt.person_count AS NUMERIC) AS VARCHAR(255))
	ELSE 
		CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(dt.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	death_total dt
;
ANALYZE s_tmpach_531
;


-- 532	Proportion of death records outside a valid observation period
--
-- stratum_1:   Proportion to 6 decimal places
-- stratum_2:   Number of death records outside a valid observation period (numerator)
-- stratum_3:   Number of death records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such rceords exist
--
CREATE TEMP TABLE s_tmpach_532
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.death d
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	d.person_id = op.person_id
AND 
	d.death_date >= op.observation_period_start_date
AND 
	d.death_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), death_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.death
)
 SELECT
532 AS analysis_id,
	CASE WHEN dt.record_count != 0 THEN
		CAST(CAST(1.0*op.record_count/dt.record_count AS NUMERIC(7,6)) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(dt.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	death_total dt
;
ANALYZE s_tmpach_532
;


-- 600	Number of persons with at least one procedure occurrence, by procedure_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_600
AS
SELECT
600 AS analysis_id,
	CAST(po.procedure_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT po.person_id) AS count_value
FROM
omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	po.procedure_concept_id;
ANALYZE s_tmpach_600
;


-- 601	Number of procedure occurrence records, by procedure_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_601
AS
SELECT
601 AS analysis_id,
	CAST(po.procedure_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(po.person_id) AS count_value
FROM
omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	po.procedure_concept_id;
ANALYZE s_tmpach_601
;


-- 602	Number of persons by procedure occurrence start month, by procedure_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_602
AS
WITH rawData  AS (
SELECT 
	po.procedure_concept_id AS stratum_1,
	EXTRACT(YEAR FROM po.procedure_date) * 100 + EXTRACT(MONTH FROM po.procedure_date) AS stratum_2,
	COUNT(DISTINCT po.person_id) AS count_value
FROM 
	omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	po.procedure_concept_id,
	EXTRACT(YEAR FROM po.procedure_date) * 100 + EXTRACT(MONTH FROM po.procedure_date)
)
 SELECT
602 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS varchar(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_602
;


-- 603	Number of distinct procedure occurrence concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_603
AS
WITH rawData(count_value)  AS (
SELECT 
	COUNT(DISTINCT po.procedure_concept_id) AS count_value
FROM 
	omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	po.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
603 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_603
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_603
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_603
;
ANALYZE s_tmpach_dist_603
;
truncate table tempResults_603;
drop table tempResults_603;


-- 604	Number of persons with at least one procedure occurrence, by procedure_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_604
AS
WITH rawData  AS (
SELECT 
	po.procedure_concept_id AS stratum_1,
	EXTRACT(YEAR FROM po.procedure_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM po.procedure_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.procedure_occurrence po 
ON 
	p.person_id = po.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	po.procedure_concept_id,
	EXTRACT(YEAR FROM po.procedure_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM po.procedure_date) - p.year_of_birth) / 10)
)
 SELECT
604 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_604
;


-- 605	Number of procedure occurrence records, by procedure_concept_id by procedure_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_605
AS
SELECT
605 AS analysis_id,
	CAST(po.procedure_CONCEPT_ID AS VARCHAR(255)) AS stratum_1,
	CAST(po.procedure_type_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(po.person_id) AS count_value
FROM
omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	po.procedure_CONCEPT_ID,
	po.procedure_type_concept_id;
ANALYZE s_tmpach_605
;


-- 606	Distribution of age by procedure_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_606
AS
SELECT
po.procedure_concept_id AS subject_id,
	p.gender_concept_id,
	po.procedure_start_year - p.year_of_birth AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		po.person_id,
		po.procedure_concept_id,
		MIN(EXTRACT(YEAR FROM po.procedure_date)) AS procedure_start_year
	FROM 
		omop.cmd.procedure_occurrence po
	JOIN 
		omop.cmd.observation_period op 
	ON 
		po.person_id = op.person_id
	AND 
		po.procedure_date >= op.observation_period_start_date
	AND 
		po.procedure_date <= op.observation_period_end_date
	GROUP BY 
		po.person_id,
		po.procedure_concept_id
	) po 
ON 
	p.person_id = po.person_id
;
ANALYZE rawData_606
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_606
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_606
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_606
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
606 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_606
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_606
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_606
;
ANALYZE s_tmpach_dist_606
;
truncate table tempResults_606;
drop table tempResults_606;
truncate table rawData_606;
drop table rawData_606;


-- 609	Number of procedure occurrence records with invalid person_id
CREATE TEMP TABLE s_tmpach_609
AS
SELECT
609 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(po1.PERSON_ID) as count_value
FROM
omop.cmd.procedure_occurrence po1
	left join omop.cmd.person p1
	on p1.person_id = po1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_609
;


-- 610	Number of procedure occurrence records outside valid observation period
CREATE TEMP TABLE s_tmpach_610
AS
SELECT
610 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(po1.PERSON_ID) as count_value
FROM
omop.cmd.procedure_occurrence po1
	left join omop.cmd.observation_period op1
	on op1.person_id = po1.person_id
	and po1.procedure_date >= op1.observation_period_start_date
	and po1.procedure_date <= op1.observation_period_end_date
where op1.person_id is null
;
ANALYZE s_tmpach_610
;


-- 612	Number of procedure occurrence records with invalid provider_id
CREATE TEMP TABLE s_tmpach_612
AS
SELECT
612 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(po1.PERSON_ID) as count_value
FROM
omop.cmd.procedure_occurrence po1
	left join omop.cmd.provider p1
	on p1.provider_id = po1.provider_id
where po1.provider_id is not null
	and p1.provider_id is null
;
ANALYZE s_tmpach_612
;


-- 613	Number of procedure occurrence records with invalid visit_id
CREATE TEMP TABLE s_tmpach_613
AS
SELECT
613 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(po1.PERSON_ID) as count_value
FROM
omop.cmd.procedure_occurrence po1
	left join omop.cmd.visit_occurrence vo1
	on po1.visit_occurrence_id = vo1.visit_occurrence_id
where po1.visit_occurrence_id is not null
	and vo1.visit_occurrence_id is null
;
ANALYZE s_tmpach_613
;


-- 620	Number of procedure occurrence records by condition occurrence start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_620
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM po.procedure_date) * 100 + EXTRACT(MONTH FROM po.procedure_date) AS stratum_1,
	COUNT(po.person_id) AS count_value
FROM
	omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM po.procedure_date)*100 + EXTRACT(MONTH FROM po.procedure_date)
)
 SELECT
620 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_620
;


-- 625	Number of procedure_occurrence records, by procedure_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_625
AS
SELECT
625 AS analysis_id,
	CAST(po.procedure_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
GROUP BY 
	po.procedure_source_concept_id;
ANALYZE s_tmpach_625
;


-- 630	Number of procedure_occurrence records inside a valid observation period
CREATE TEMP TABLE s_tmpach_630
AS
SELECT
630 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.procedure_occurrence po
JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_630
;


-- 631	Proportion of people with at least one procedure_occurrence record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in procedure_occurrence (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_631
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT po.person_id) AS person_count
FROM 
	omop.cmd.procedure_occurrence po
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), po_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.procedure_occurrence
)
 SELECT
631 AS analysis_id,
	CASE WHEN po.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/po.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(po.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	po_total po
;
ANALYZE s_tmpach_631
;


-- 632	Proportion of procedure_occurrence records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of procedure_occurrence records outside a valid observation period (numerator)
-- stratum_3:   Number of procedure_occurrence records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_632
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.procedure_occurrence po
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	po.person_id = op.person_id
AND 
	po.procedure_date >= op.observation_period_start_date
AND 
	po.procedure_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), po_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.procedure_occurrence
)
 SELECT
632 AS analysis_id,
	CASE WHEN po.record_count != 0 THEN 
		CAST(CAST(1.0*op.record_count/po.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(po.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	po_total po
;
ANALYZE s_tmpach_632
;


-- 691	Number of persons that have at least x procedures
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_691
AS
SELECT
691 AS analysis_id,
	CAST(po.procedure_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(po.prc_cnt AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SUM(COUNT(po.person_id)) OVER (PARTITION BY po.procedure_concept_id ORDER BY po.prc_cnt DESC) AS count_value
FROM
(
	SELECT 
		po.procedure_concept_id,
		COUNT(po.procedure_occurrence_id) AS prc_cnt,
		po.person_id
	FROM 
		omop.cmd.procedure_occurrence po
	JOIN 
		omop.cmd.observation_period op 
	ON 
		po.person_id = op.person_id
	AND 
		po.procedure_date >= op.observation_period_start_date
	AND 
		po.procedure_date <= op.observation_period_end_date
	GROUP BY 
		po.person_id,
		po.procedure_concept_id
	) po
GROUP BY 
	po.procedure_concept_id,
	po.prc_cnt;
ANALYZE s_tmpach_691
;


-- 700	Number of persons with at least one drug occurrence, by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_700
AS
SELECT
700 AS analysis_id,
	CAST(de.drug_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT de.person_id) AS count_value
FROM
omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id;
ANALYZE s_tmpach_700
;


-- 701	Number of drug occurrence records, by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_701
AS
SELECT
701 AS analysis_id,
	CAST(de.drug_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(de.person_id) AS count_value
FROM
omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id;
ANALYZE s_tmpach_701
;


-- 702	Number of persons by drug occurrence start month, by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_702
AS
WITH rawData  AS (
SELECT 
	de.drug_concept_id AS stratum_1,
	EXTRACT(YEAR FROM de.drug_exposure_start_date) * 100 + EXTRACT(MONTH FROM de.drug_exposure_start_date) AS stratum_2,
	COUNT(DISTINCT de.person_id) AS count_value
FROM 
	omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id,
	EXTRACT(YEAR FROM de.drug_exposure_start_date) * 100 + EXTRACT(MONTH FROM de.drug_exposure_start_date)
)
 SELECT
702 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_702
;


-- 703	Number of distinct drug exposure concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_703
AS
WITH rawData(count_value)  AS (
SELECT 
	COUNT(DISTINCT de.drug_concept_id) AS count_value
FROM 
	omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
GROUP BY 
	de.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
703 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_703
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_703
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_703
;
ANALYZE s_tmpach_dist_703
;
truncate table tempResults_703;
drop table tempResults_703;


-- 704	Number of persons with at least one drug occurrence, by drug_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_704
AS
WITH rawData  AS (
SELECT 
	de.drug_concept_id AS stratum_1,
	EXTRACT(YEAR FROM de.drug_exposure_start_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM de.drug_exposure_start_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.drug_exposure de 
ON 
	p.person_id = de.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id,
	EXTRACT(YEAR FROM de.drug_exposure_start_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM de.drug_exposure_start_date) - p.year_of_birth) / 10)
)
 SELECT
704 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_704
;


-- 705	Number of drug occurrence records, by drug_concept_id by drug_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_705
AS
SELECT
705 AS analysis_id,
	CAST(de.drug_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(de.drug_type_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(de.person_id) AS count_value
FROM
omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_CONCEPT_ID,
	de.drug_type_concept_id;
ANALYZE s_tmpach_705
;


-- 706	Distribution of age by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_706
AS
SELECT
de.drug_concept_id AS subject_id,
	p.gender_concept_id,
	de.drug_start_year - p.year_of_birth AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		de.person_id,
		de.drug_concept_id,
		MIN(EXTRACT(YEAR FROM de.drug_exposure_start_date)) AS drug_start_year
	FROM 
		omop.cmd.drug_exposure de
	JOIN 
		omop.cmd.observation_period op 
	ON 
		de.person_id = op.person_id
	AND 
		de.drug_exposure_start_date >= op.observation_period_start_date
	AND 
		de.drug_exposure_start_date <= op.observation_period_end_date
	GROUP BY 
		de.person_id,
		de.drug_concept_id
	) de 
ON 
	p.person_id = de.person_id
;
ANALYZE rawData_706
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_706
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_706
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_706
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
706 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_706
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_706
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_706
;
ANALYZE s_tmpach_dist_706
;
truncate table rawData_706;
drop table rawData_706;
truncate table tempResults_706;
drop table tempResults_706;


-- 709	Number of drug exposure records with invalid person_id
CREATE TEMP TABLE s_tmpach_709
AS
SELECT
709 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(de1.PERSON_ID) as count_value
FROM
omop.cmd.drug_exposure de1
	left join omop.cmd.person p1
	on p1.person_id = de1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_709
;


-- 710	Number of drug exposure records outside valid observation period
CREATE TEMP TABLE s_tmpach_710
AS
SELECT
710 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(de1.PERSON_ID) as count_value
FROM
omop.cmd.drug_exposure de1
	left join omop.cmd.observation_period op1
	on op1.person_id = de1.person_id
	and de1.drug_exposure_start_date >= op1.observation_period_start_date
	and de1.drug_exposure_start_date <= op1.observation_period_end_date
where op1.person_id is null
;
ANALYZE s_tmpach_710
;


-- 711	Number of drug exposure records with end date < start date
CREATE TEMP TABLE s_tmpach_711
AS
SELECT
711 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(de1.PERSON_ID) as count_value
FROM
omop.cmd.drug_exposure de1
where de1.drug_exposure_end_date < de1.drug_exposure_start_date
;
ANALYZE s_tmpach_711
;


-- 712	Number of drug exposure records with invalid provider_id
CREATE TEMP TABLE s_tmpach_712
AS
SELECT
712 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(de1.PERSON_ID) as count_value
FROM
omop.cmd.drug_exposure de1
	left join omop.cmd.provider p1
	on p1.provider_id = de1.provider_id
where de1.provider_id is not null
	and p1.provider_id is null
;
ANALYZE s_tmpach_712
;


-- 713	Number of drug exposure records with invalid visit_id
CREATE TEMP TABLE s_tmpach_713
AS
SELECT
713 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(de1.PERSON_ID) as count_value
FROM
omop.cmd.drug_exposure de1
	left join omop.cmd.visit_occurrence vo1
	on de1.visit_occurrence_id = vo1.visit_occurrence_id
where de1.visit_occurrence_id is not null
	and vo1.visit_occurrence_id is null
;
ANALYZE s_tmpach_713
;


-- 715	Distribution of days_supply by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_id)
CREATE TEMP TABLE tempResults_715
AS
WITH rawData(stratum_id, count_value)  AS (
SELECT 
	de.drug_concept_id AS stratum_id,
	de.days_supply AS count_value
FROM 
	omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
WHERE 
	de.days_supply IS NOT NULL
),
overallStats (stratum_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData
	group by stratum_id
),
statsView (stratum_id, count_value, total, rn) as
(
  select stratum_id, count_value, COUNT(*) as total, row_number() over (partition by stratum_id order by count_value) as rn
  FROM rawData
  group by stratum_id, count_value
),
priorStats (stratum_id, count_value, total, accumulated) as
(
  select s.stratum_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum_id = p.stratum_id and p.rn <= s.rn
  group by s.stratum_id, s.count_value, s.total, s.rn
)
 SELECT
715 as analysis_id,
  CAST(o.stratum_id AS VARCHAR(255)) AS stratum_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum_id = o.stratum_id
GROUP BY o.stratum_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_715
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_715
AS
SELECT
analysis_id, stratum_id as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_715
;
ANALYZE s_tmpach_dist_715
;
truncate table tempResults_715;
drop table tempResults_715;


-- 716	Distribution of refills by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_id)
CREATE TEMP TABLE tempResults_716
AS
WITH rawData(stratum_id, count_value)  AS (
SELECT 
	de.drug_concept_id AS stratum_id,
	de.refills AS count_value
FROM 
	omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
WHERE 
	de.refills IS NOT NULL
),
overallStats (stratum_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData
	group by stratum_id
),
statsView (stratum_id, count_value, total, rn) as
(
  select stratum_id, count_value, COUNT(*) as total, row_number() over (partition by stratum_id order by count_value) as rn
  FROM rawData
  group by stratum_id, count_value
),
priorStats (stratum_id, count_value, total, accumulated) as
(
  select s.stratum_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum_id = p.stratum_id and p.rn <= s.rn
  group by s.stratum_id, s.count_value, s.total, s.rn
)
 SELECT
716 as analysis_id,
  CAST(o.stratum_id AS VARCHAR(255)) AS stratum_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum_id = o.stratum_id
GROUP BY o.stratum_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_716
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_716
AS
SELECT
analysis_id, stratum_id as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_716
;
ANALYZE s_tmpach_dist_716
;
truncate table tempResults_716;
drop table tempResults_716;


-- 717	Distribution of quantity by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_id)
CREATE TEMP TABLE tempResults_717
AS
WITH rawData(stratum_id, count_value)  AS (
SELECT 
	de.drug_concept_id AS stratum_id,
	CAST(de.quantity AS NUMERIC) AS count_value
FROM 
	omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
WHERE 
	de.quantity IS NOT NULL
),
overallStats (stratum_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData
	group by stratum_id
),
statsView (stratum_id, count_value, total, rn) as
(
  select stratum_id, count_value, COUNT(*) as total, row_number() over (order by count_value) as rn
  FROM rawData
  group by stratum_id, count_value
),
priorStats (stratum_id, count_value, total, accumulated) as
(
  select s.stratum_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum_id = p.stratum_id and p.rn <= s.rn
  group by s.stratum_id, s.count_value, s.total, s.rn
)
 SELECT
717 as analysis_id,
  CAST(o.stratum_id AS VARCHAR(255)) AS stratum_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum_id = o.stratum_id
GROUP BY o.stratum_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_717
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_717
AS
SELECT
analysis_id, stratum_id as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_717
;
ANALYZE s_tmpach_dist_717
;
truncate table tempResults_717;
drop table tempResults_717;


-- 720	Number of drug exposure records by condition occurrence start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_720
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM de.drug_exposure_start_date) * 100 + EXTRACT(MONTH FROM de.drug_exposure_start_date) AS stratum_1,
	COUNT(de.person_id) AS count_value
FROM 
	omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date  
GROUP BY 
	EXTRACT(YEAR FROM de.drug_exposure_start_date)*100 + EXTRACT(MONTH FROM de.drug_exposure_start_date)
)
 SELECT
720 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_720
;


-- 725	Number of drug_exposure records, by drug_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_725
AS
SELECT
725 AS analysis_id,
	CAST(de.drug_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_source_concept_id;
ANALYZE s_tmpach_725
;


-- 730	Number of drug_exposure records inside a valid observation period
CREATE TEMP TABLE s_tmpach_730
AS
SELECT
730 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.drug_exposure de
JOIN 
	omop.cmd.observation_period op
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_730
;


-- 731	Proportion of people with at least one drug_exposure record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in drug_exposure (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_731
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT de.person_id) AS person_count
FROM 
	omop.cmd.drug_exposure de
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), de_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.drug_exposure
)
 SELECT
731 AS analysis_id,
	CASE WHEN de.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/de.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(de.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	de_total de
;
ANALYZE s_tmpach_731
;


-- 732	Proportion of drug_exposure records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of drug_exposure records outside a valid observation period (numerator)
-- stratum_3:   Number of drug_exposure records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_732
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.drug_exposure de
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_exposure_start_date >= op.observation_period_start_date
AND 
	de.drug_exposure_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), de_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.drug_exposure
)
 SELECT
732 AS analysis_id,
	CASE WHEN de.record_count != 0 THEN 
		CAST(CAST(1.0*op.record_count/de.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(de.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	de_total de
;
ANALYZE s_tmpach_732
;


-- 791	Number of total persons that have at least x drug exposures
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_791
AS
SELECT
791 AS analysis_id,
	CAST(de.drug_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(de.drg_cnt AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SUM(COUNT(de.person_id)) OVER (PARTITION BY de.drug_concept_id ORDER BY de.drg_cnt DESC) AS count_value
FROM
(
	SELECT 
		de.drug_concept_id,
		COUNT(de.drug_exposure_id) AS drg_cnt,
		de.person_id
	FROM 
		omop.cmd.drug_exposure de
	JOIN 
		omop.cmd.observation_period op 
	ON 
		de.person_id = op.person_id
	AND 
		de.drug_exposure_start_date >= op.observation_period_start_date
	AND 
		de.drug_exposure_start_date <= op.observation_period_end_date
	GROUP BY 
		de.person_id,
		de.drug_concept_id
	) de
GROUP BY 
	de.drug_concept_id, 
	de.drg_cnt
;
ANALYZE s_tmpach_791
;


-- 800	Number of persons with at least one observation occurrence, by observation_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_800
AS
SELECT
800 AS analysis_id,
	CAST(o.observation_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT o.person_id) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_concept_id;
ANALYZE s_tmpach_800
;


-- 801	Number of observation occurrence records, by observation_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_801
AS
SELECT
801 AS analysis_id,
	CAST(o.observation_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(o.person_id) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_concept_id;
ANALYZE s_tmpach_801
;


-- 802	Number of persons by observation occurrence start month, by observation_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_802
AS
WITH rawData  AS (
SELECT 
	o.observation_concept_id AS stratum_1,
	EXTRACT(YEAR FROM o.observation_date) * 100 + EXTRACT(MONTH FROM o.observation_date) AS stratum_2,
	COUNT(DISTINCT o.person_id) AS count_value
FROM 
	omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_concept_id,
	EXTRACT(YEAR FROM o.observation_date) * 100 + EXTRACT(MONTH FROM o.observation_date)
)
 SELECT
802 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_802
;


-- 803	Number of distinct observation occurrence concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_803
AS
WITH rawData(count_value)  AS (
SELECT 
	COUNT(DISTINCT o.observation_concept_id) AS count_value
FROM 
	omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
803 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_803
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_803
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_803
;
ANALYZE s_tmpach_dist_803
;
truncate table tempResults_803;
drop table tempResults_803;


-- 804	Number of persons with at least one observation occurrence, by observation_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_804
AS
WITH rawData  AS (
SELECT 
	o.observation_concept_id AS stratum_1,
	EXTRACT(YEAR FROM o.observation_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM o.observation_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.observation o 
ON 
	p.person_id = o.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_concept_id,
	EXTRACT(YEAR FROM o.observation_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM o.observation_date) - p.year_of_birth) / 10)
)
 SELECT
804 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_804
;


-- 805	Number of observation occurrence records, by observation_concept_id by observation_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_805
AS
SELECT
805 AS analysis_id,
	CAST(o.observation_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(o.observation_type_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(o.person_id) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_concept_id,
	o.observation_type_concept_id;
ANALYZE s_tmpach_805
;


-- 806	Distribution of age by observation_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_806
AS
SELECT
o.observation_concept_id AS subject_id,
	p.gender_concept_id,
	o.observation_start_year - p.year_of_birth AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		o.person_id,
		o.observation_concept_id,
		MIN(EXTRACT(YEAR FROM o.observation_date)) AS observation_start_year
	FROM 
		omop.cmd.observation o
	JOIN 
		omop.cmd.observation_period op 
	ON 
		o.person_id = op.person_id
	AND 
		o.observation_date >= op.observation_period_start_date
	AND 
		o.observation_date <= op.observation_period_end_date
	GROUP BY 
		o.person_id,
		o.observation_concept_id
	) o
ON 
	p.person_id = o.person_id
;
ANALYZE rawData_806
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_806
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_806
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_806
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
806 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_806
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_806
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_806
;
ANALYZE s_tmpach_dist_806
;
truncate table rawData_806;
drop table rawData_806;
truncate table tempResults_806;
drop table tempResults_806;


-- 807	Number of observation occurrence records, by observation_concept_id and unit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_807
AS
SELECT
807 AS analysis_id,
	CAST(o.observation_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(o.unit_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(o.person_id) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_CONCEPT_ID,
	o.unit_concept_id;
ANALYZE s_tmpach_807
;


-- 809	Number of observation records with invalid person_id
CREATE TEMP TABLE s_tmpach_809
AS
SELECT
809 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(o1.PERSON_ID) as count_value
FROM
omop.cmd.observation o1
	left join omop.cmd.person p1
	on p1.person_id = o1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_809
;


-- 810	Number of observation records outside valid observation period
CREATE TEMP TABLE s_tmpach_810
AS
SELECT
810 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(o1.PERSON_ID) as count_value
FROM
omop.cmd.observation o1
	left join omop.cmd.observation_period op1
	on op1.person_id = o1.person_id
	and o1.observation_date >= op1.observation_period_start_date
	and o1.observation_date <= op1.observation_period_end_date
where op1.person_id is null
;
ANALYZE s_tmpach_810
;


-- 812	Number of observation records with invalid provider_id
CREATE TEMP TABLE s_tmpach_812
AS
SELECT
812 as analysis_id,
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,  
	COUNT(o1.PERSON_ID) as count_value
FROM
omop.cmd.observation o1
	left join omop.cmd.provider p1
	on p1.provider_id = o1.provider_id
where o1.provider_id is not null
	and p1.provider_id is null
;
ANALYZE s_tmpach_812
;


-- 813	Number of observation records with invalid visit_id
CREATE TEMP TABLE s_tmpach_813
AS
SELECT
813 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(o1.PERSON_ID) as count_value
FROM
omop.cmd.observation o1
	left join omop.cmd.visit_occurrence vo1
	on o1.visit_occurrence_id = vo1.visit_occurrence_id
where o1.visit_occurrence_id is not null
	and vo1.visit_occurrence_id is null
;
ANALYZE s_tmpach_813
;


-- 814	Number of observation records with no value (numeric, string, or concept)
CREATE TEMP TABLE s_tmpach_814
AS
SELECT
814 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(o1.PERSON_ID) as count_value
FROM
omop.cmd.observation o1
where o1.value_as_number is null
	and o1.value_as_string is null
	and o1.value_as_concept_id is null
;
ANALYZE s_tmpach_814
;


--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE overallStats_815
AS
SELECT
o.subject_id AS stratum1_id,
	o.unit_concept_id AS stratum2_id,
	CAST(AVG(1.0 * o.count_value) AS NUMERIC) AS avg_value,
	CAST(STDDEV(count_value) AS NUMERIC) AS stdev_value,
	MIN(o.count_value) AS min_value,
	MAX(o.count_value) AS max_value,
	COUNT(*) AS total
FROM
(
	SELECT 
		o.observation_concept_id AS subject_id,
		o.unit_concept_id,
		CAST(o.value_as_number AS NUMERIC) AS count_value
	FROM 
		omop.cmd.observation o
	JOIN 
		omop.cmd.observation_period op 
	ON 
		o.person_id = op.person_id
	AND 
		o.observation_date >= op.observation_period_start_date
	AND 
		o.observation_date <= op.observation_period_end_date
	WHERE 
		o.unit_concept_id IS NOT NULL
	AND 
		o.value_as_number IS NOT NULL
	) o
GROUP BY 
	o.subject_id,
	o.unit_concept_id;
ANALYZE overallStats_815
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE statsView_815
AS
SELECT
o.subject_id AS stratum1_id,
	o.unit_concept_id AS stratum2_id,
	o.count_value,
	COUNT(*) AS total,
	ROW_NUMBER() OVER (PARTITION BY o.subject_id,o.unit_concept_id ORDER BY o.count_value) AS rn
FROM
(
	SELECT 
		o.observation_concept_id AS subject_id,
		o.unit_concept_id,
		CAST(o.value_as_number AS NUMERIC) AS count_value
	FROM 
		omop.cmd.observation o
	JOIN 
		omop.cmd.observation_period op 
	ON 
		o.person_id = op.person_id
	AND 
		o.observation_date >= op.observation_period_start_date
	AND 
		o.observation_date <= op.observation_period_end_date
	WHERE 
		o.unit_concept_id IS NOT NULL
	AND 
		o.value_as_number IS NOT NULL
	) o
GROUP BY 
	o.subject_id,
	o.unit_concept_id,
	o.count_value;
ANALYZE statsView_815
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_815
AS
WITH priorStats (stratum1_id, stratum2_id, count_value, total, accumulated)  AS (
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView_815 s
  join statsView_815 p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
815 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats_815 o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_815
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_815
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_815
;
ANALYZE s_tmpach_dist_815
;
truncate table overallStats_815;
drop table overallStats_815;
truncate table statsView_815;
drop table statsView_815;
truncate table tempResults_815;
drop table tempResults_815;


-- 820	Number of observation records by condition occurrence start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_820
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM o.observation_date) * 100 + EXTRACT(MONTH FROM o.observation_date) AS stratum_1,
	COUNT(o.person_id) AS count_value
FROM 
	omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM o.observation_date) * 100 + EXTRACT(MONTH FROM o.observation_date)
)
 SELECT
820 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_820
;


-- 822	Number of observation records, by observation_concept_id and value_as_concept_id, observation_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_822
AS
SELECT
822 AS analysis_id,
	CAST(o.observation_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(o.value_as_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_concept_id,
	o.value_as_concept_id;
ANALYZE s_tmpach_822
;


-- 823	Number of observation records, by observation_concept_id and qualifier_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_823
AS
SELECT
823 AS analysis_id,
	CAST(o.observation_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(o.qualifier_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_concept_id,
	o.qualifier_concept_id;
ANALYZE s_tmpach_823
;


-- 825	Number of observation records, by observation_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_825
AS
SELECT
825 AS analysis_id,
	CAST(o.observation_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.observation_source_concept_id;
ANALYZE s_tmpach_825
;


-- 826	Number of observation records, by value_as_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_826
AS
SELECT
826 AS analysis_id,
	CAST(o.value_as_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.value_as_concept_id;
ANALYZE s_tmpach_826
;


-- 827	Number of observation records, by unit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_827
AS
SELECT
827 AS analysis_id,
	CAST(o.unit_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
GROUP BY 
	o.unit_concept_id;
ANALYZE s_tmpach_827
;


-- 830	Number of observation records inside a valid observation period
CREATE TEMP TABLE s_tmpach_830
AS
SELECT
830 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.observation o
JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_830
;


-- 831	Proportion of people with at least one observation record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in observation (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_831
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT o.person_id) AS person_count
FROM 
	omop.cmd.observation o
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), o_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.observation
)
 SELECT
831 AS analysis_id,
	CASE WHEN ot.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/ot.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(ot.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	o_total ot
;
ANALYZE s_tmpach_831
;


-- 832	Proportion of observation records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of observation records outside a valid observation period (numerator)
-- stratum_3:   Number of observation records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_832
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.observation o
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	o.person_id = op.person_id
AND 
	o.observation_date >= op.observation_period_start_date
AND 
	o.observation_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), o_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.observation
)
 SELECT
832 AS analysis_id,
	CASE WHEN ot.record_count != 0 THEN 
		CAST(CAST(1.0*op.record_count/ot.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(ot.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	o_total ot
;
ANALYZE s_tmpach_832
;


-- 891	Number of total persons that have at least x observations
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_891
AS
SELECT
891 AS analysis_id,
	CAST(o.observation_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(o.obs_cnt AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SUM(COUNT(o.person_id)) OVER (PARTITION BY o.observation_concept_id ORDER BY o.obs_cnt DESC) AS count_value
FROM
(
	SELECT 
		o.observation_concept_id,
		COUNT(o.observation_id) AS obs_cnt,
		o.person_id
	FROM 
		omop.cmd.observation o
	JOIN 
		omop.cmd.observation_period op 
	ON 
		o.person_id = op.person_id
	AND 
		o.observation_date >= op.observation_period_start_date
	AND 
		o.observation_date <= op.observation_period_end_date
	GROUP BY 
		o.person_id,
		o.observation_concept_id
	) o
GROUP BY 
	o.observation_concept_id, 
	o.obs_cnt
;
ANALYZE s_tmpach_891
;


-- 900	Number of persons with at least one drug occurrence, by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_900
AS
SELECT
900 AS analysis_id,
	CAST(de.drug_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT de.person_id) AS count_value
FROM
omop.cmd.drug_era de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id;
ANALYZE s_tmpach_900
;


-- 901	Number of drug occurrence records, by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_901
AS
SELECT
901 AS analysis_id,
	CAST(de.drug_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(de.person_id) AS count_value
FROM
omop.cmd.drug_era de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id;
ANALYZE s_tmpach_901
;


-- 902	Number of persons by drug occurrence start month, by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_902
AS
WITH rawData  AS (
SELECT 
	de.drug_concept_id AS stratum_1,
	EXTRACT(YEAR FROM de.drug_era_start_date) * 100 + EXTRACT(MONTH FROM de.drug_era_start_date) AS stratum_2,
	COUNT(DISTINCT de.person_id) AS count_value
FROM 
	omop.cmd.drug_era de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id,
	EXTRACT(YEAR FROM de.drug_era_start_date) * 100 + EXTRACT(MONTH FROM de.drug_era_start_date)
)
 SELECT
902 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_902
;


-- 903	Number of distinct drug era concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_903
AS
WITH rawData(count_value)  AS (
SELECT
	COUNT(DISTINCT de.drug_concept_id) AS count_value
FROM 
	omop.cmd.drug_era de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
GROUP BY 
	de.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
903 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_903
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_903
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_903
;
ANALYZE s_tmpach_dist_903
;
truncate table tempResults_903;
drop table tempResults_903;


-- 904	Number of persons with at least one drug occurrence, by drug_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_904
AS
WITH rawData  AS (
SELECT 
	de.drug_concept_id AS stratum_1,
	EXTRACT(YEAR FROM de.drug_era_start_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM de.drug_era_start_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.drug_era de 
ON 
	p.person_id = de.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
GROUP BY 
	de.drug_concept_id,
	EXTRACT(YEAR FROM de.drug_era_start_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM de.drug_era_start_date) - p.year_of_birth) / 10)
)
 SELECT
904 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_904
;


-- 906	Distribution of age by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_906
AS
SELECT
de.drug_concept_id AS subject_id,
	p.gender_concept_id,
	de.drug_start_year - p.year_of_birth AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		de.person_id,
		de.drug_concept_id,
		MIN(EXTRACT(YEAR FROM de.drug_era_start_date)) AS drug_start_year
	FROM 
		omop.cmd.drug_era de
	JOIN 
		omop.cmd.observation_period op 
	ON 
		de.person_id = op.person_id
	AND 
		de.drug_era_start_date >= op.observation_period_start_date
	AND 
		de.drug_era_start_date <= op.observation_period_end_date	
	GROUP BY 
		de.person_id,
		de.drug_concept_id
	) de 
ON 
	p.person_id = de.person_id
;
ANALYZE rawData_906
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_906
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_906
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_906
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
906 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_906
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_906
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_906
;
ANALYZE s_tmpach_dist_906
;
truncate table rawData_906;
drop table rawData_906;
truncate table tempResults_906;
drop table tempResults_906;


-- 907	Distribution of drug era length, by drug_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE tempResults_907
AS
WITH rawData(stratum1_id, count_value)  AS (
SELECT 
	de.drug_concept_id AS stratum1_id,
	(CAST(de.drug_era_end_date AS DATE) - CAST(de.drug_era_start_date AS DATE)) AS count_value
FROM 
	omop.cmd.drug_era de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
),
overallStats (stratum1_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum1_id, 
  	CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
  group by stratum1_id
),
statsView (stratum1_id, count_value, total, rn) as
(
  select stratum1_id, 
		count_value, 
  	COUNT(*) as total, 
		row_number() over (partition by stratum1_id order by count_value) as rn
  FROM rawData
  group by stratum1_id, count_value
),
priorStats (stratum1_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and p.rn <= s.rn
  group by s.stratum1_id, s.count_value, s.total, s.rn
)
 SELECT
907 as analysis_id,
  CAST(p.stratum1_id AS VARCHAR(255)) as stratum_1,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id
GROUP BY p.stratum1_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_907
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_907
AS
SELECT
analysis_id, stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_907
;
ANALYZE s_tmpach_dist_907
;
truncate table tempResults_907;
drop table tempResults_907;


-- 908	Number of drug eras with invalid person
CREATE TEMP TABLE s_tmpach_908
AS
SELECT
908 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(de1.PERSON_ID) as count_value
FROM
omop.cmd.drug_era de1
	left join omop.cmd.person p1
	on p1.person_id = de1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_908
;


-- 910	Number of drug_era records outside a valid observation period
CREATE TEMP TABLE s_tmpach_910
AS
SELECT
910 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.drug_era de
LEFT JOIN 
	omop.cmd.observation_period op
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
WHERE 
	op.person_id IS NULL
;
ANALYZE s_tmpach_910
;


-- 911	Number of drug eras with end date < start date
CREATE TEMP TABLE s_tmpach_911
AS
SELECT
911 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(de1.PERSON_ID) as count_value
FROM
omop.cmd.drug_era de1
where de1.drug_era_end_date < de1.drug_era_start_date
;
ANALYZE s_tmpach_911
;


-- 920	Number of drug era records by drug era start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_920
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM de.drug_era_start_date) * 100 + EXTRACT(MONTH FROM de.drug_era_start_date) AS stratum_1,
	COUNT(de.person_id) AS count_value
FROM 
	omop.cmd.drug_era de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM de.drug_era_start_date)*100 + EXTRACT(MONTH FROM de.drug_era_start_date)
)
 SELECT
920 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_920
;


-- 930	Number of drug_era records inside a valid observation period
CREATE TEMP TABLE s_tmpach_930
AS
SELECT
930 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.drug_era de
JOIN 
	omop.cmd.observation_period op
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_930
;


-- 931	Proportion of people with at least one drug_era record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in drug_era (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_931
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT de.person_id) AS person_count
FROM 
	omop.cmd.drug_era de
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), de_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.drug_era
)
 SELECT
931 AS analysis_id,
	CASE WHEN det.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/det.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(det.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	de_total det
;
ANALYZE s_tmpach_931
;


-- 932	Proportion of drug_era records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of drug_era records outside a valid observation period (numerator)
-- stratum_3:   Number of drug_era records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_932
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.drug_era de
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.drug_era_start_date >= op.observation_period_start_date
AND 
	de.drug_era_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), de_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.drug_era
)
 SELECT
932 AS analysis_id,
	CASE WHEN det.record_count != 0 THEN 
		CAST(CAST(1.0*op.record_count/det.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE
		CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(det.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	de_total det
;
ANALYZE s_tmpach_932
;


-- 1000	Number of persons with at least one condition occurrence, by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1000
AS
SELECT
1000 AS analysis_id,
	CAST(ce.condition_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT ce.person_id) AS count_value
FROM
omop.cmd.condition_era ce
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date	
GROUP BY 
	ce.condition_concept_id;
ANALYZE s_tmpach_1000
;


-- 1001	Number of condition occurrence records, by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1001
AS
SELECT
1001 AS analysis_id,
	CAST(ce.condition_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(ce.person_id) AS count_value
FROM
omop.cmd.condition_era ce
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date	
GROUP BY 
	ce.condition_concept_id;
ANALYZE s_tmpach_1001
;


-- 1002	Number of persons by condition occurrence start month, by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1002
AS
WITH rawData  AS (
SELECT 
	ce.condition_concept_id AS stratum_1,
	EXTRACT(YEAR FROM ce.condition_era_start_date) * 100 + EXTRACT(MONTH FROM ce.condition_era_start_date) AS stratum_2,
	COUNT(DISTINCT ce.person_id) AS count_value
FROM 
	omop.cmd.condition_era ce
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date	
GROUP BY 
	ce.condition_concept_id,
	EXTRACT(YEAR FROM ce.condition_era_start_date) * 100 + EXTRACT(MONTH FROM ce.condition_era_start_date)
)
 SELECT
1002 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_1002
;


-- 1003	Number of distinct condition era concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_1003
AS
WITH rawData(count_value)  AS (
SELECT 
	COUNT(DISTINCT ce.condition_concept_id) AS count_value
FROM 
	omop.cmd.condition_era ce
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date	
GROUP BY 
	ce.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
1003 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1003
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_1003
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1003
;
ANALYZE s_tmpach_dist_1003
;
truncate table tempResults_1003;
drop table tempResults_1003;


-- 1004	Number of persons with at least one condition occurrence, by condition_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1004
AS
WITH rawData  AS (
SELECT 
	ce.condition_concept_id AS stratum_1,
	EXTRACT(YEAR FROM ce.condition_era_start_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM ce.condition_era_start_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.condition_era ce 
ON 
	p.person_id = ce.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date	
GROUP BY 
	ce.condition_concept_id,
	EXTRACT(YEAR FROM ce.condition_era_start_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM ce.condition_era_start_date) - p.year_of_birth) / 10)
)
 SELECT
1004 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1004
;


-- 1006	Distribution of age by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_1006
AS
SELECT
ce.condition_concept_id AS subject_id,
	p.gender_concept_id,
	ce.condition_start_year - p.year_of_birth AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		ce.person_id,
		ce.condition_concept_id,
		MIN(EXTRACT(YEAR FROM ce.condition_era_start_date)) AS condition_start_year
	FROM 
		omop.cmd.condition_era ce
	JOIN 
		omop.cmd.observation_period op 
	ON 
		ce.person_id = op.person_id
	AND 
		ce.condition_era_start_date >= op.observation_period_start_date
	AND 
		ce.condition_era_start_date <= op.observation_period_end_date			
	GROUP BY 
		ce.person_id,
		ce.condition_concept_id
	) ce 
ON 
	p.person_id = ce.person_id
;
ANALYZE rawData_1006
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_1006
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_1006
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_1006
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
1006 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1006
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1006
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1006
;
ANALYZE s_tmpach_dist_1006
;
truncate table rawData_1006;
drop table rawData_1006;
truncate table tempResults_1006;
drop table tempResults_1006;


-- 1007	Distribution of condition era length, by condition_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE tempResults_1007
AS
WITH rawData(stratum1_id, count_value)  AS (
SELECT 
	ce.condition_concept_id AS stratum1_id,
	(CAST(ce.condition_era_end_date AS DATE) - CAST(ce.condition_era_start_date AS DATE)) AS count_value
FROM 
	omop.cmd.condition_era ce
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date	
),
overallStats (stratum1_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum1_id, 
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
  group by stratum1_id
),
statsView (stratum1_id, count_value, total, rn) as
(
  select stratum1_id, 
		count_value, 
  	COUNT(*) as total, 
		row_number() over (partition by stratum1_id order by count_value) as rn
  FROM rawData
  group by stratum1_id, count_value
),
priorStats (stratum1_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and p.rn <= s.rn
  group by s.stratum1_id, s.count_value, s.total, s.rn
)
 SELECT
1007 as analysis_id,
  CAST(p.stratum1_id AS VARCHAR(255)) as stratum_1,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id
GROUP BY p.stratum1_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1007
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1007
AS
SELECT
analysis_id, stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1007
;
ANALYZE s_tmpach_dist_1007
;
truncate table tempResults_1007;
drop table tempResults_1007;


-- 1008	Number of condition eras with invalid person
CREATE TEMP TABLE s_tmpach_1008
AS
SELECT
1008 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(ce1.PERSON_ID) as count_value
FROM
omop.cmd.condition_era ce1
	left join omop.cmd.person p1
	on p1.person_id = ce1.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_1008
;


-- 1010	Number of condition_era records outside a valid observation period
CREATE TEMP TABLE s_tmpach_1010
AS
SELECT
1010 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.condition_era ce
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date
WHERE 
	op.person_id IS NULL
;
ANALYZE s_tmpach_1010
;


-- 1011	Number of condition eras with end date < start date
CREATE TEMP TABLE s_tmpach_1011
AS
SELECT
1011 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(ce1.PERSON_ID) as count_value
FROM
omop.cmd.condition_era ce1
where ce1.condition_era_end_date < ce1.condition_era_start_date
;
ANALYZE s_tmpach_1011
;


-- 1020	Number of condition era records by condition era start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1020
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM ce.condition_era_start_date) * 100 + EXTRACT(MONTH FROM ce.condition_era_start_date) AS stratum_1,
	COUNT(ce.person_id) AS count_value
FROM 
	omop.cmd.condition_era ce
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date	
GROUP BY 
	EXTRACT(YEAR FROM ce.condition_era_start_date)*100 + EXTRACT(MONTH FROM ce.condition_era_start_date)
)
 SELECT
1020 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1020
;


-- 1030	Number of condition_era records inside a valid observation period
CREATE TEMP TABLE s_tmpach_1030
AS
SELECT
1030 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.condition_era ce
JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_1030
;


-- 1031	Proportion of people with at least one condition_era record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in condition_era (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_1031
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT ce.person_id) AS person_count
FROM 
	omop.cmd.condition_era ce
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), ce_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.condition_era
)
 SELECT
1031 AS analysis_id,
	CASE WHEN cet.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/cet.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(cet.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	ce_total cet
;
ANALYZE s_tmpach_1031
;


-- 1032	Proportion of condition_era records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of condition_era records outside a valid observation period (numerator)
-- stratum_3:   Number of condition_era records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_1032
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.condition_era ce
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	ce.person_id = op.person_id
AND 
	ce.condition_era_start_date >= op.observation_period_start_date
AND 
	ce.condition_era_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), ce_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.condition_era
)
 SELECT
1032 AS analysis_id,
	CASE WHEN cet.record_count != 0 THEN
		CAST(CAST(1.0*op.record_count/cet.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(cet.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	ce_total cet
;
ANALYZE s_tmpach_1032
;


-- 1100	Number of persons by location 3-digit zip
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1100
AS
WITH rawData  AS (
  select
    left(l1.zip,3) as stratum_1,
    COUNT(distinct person_id) as count_value
  from omop.cmd.person p1
    inner join omop.cmd.location l1
    on p1.location_id = l1.location_id
  where p1.location_id is not null
    and l1.zip is not null
  group by left(l1.zip,3)
)
 SELECT
1100 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(null as varchar(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_1100
;


-- 1101	Number of persons by location state
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1101
AS
SELECT
1101 as analysis_id,  
	CAST(l1.state AS VARCHAR(255)) as stratum_1, 
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct person_id) as count_value
FROM
omop.cmd.person p1
	inner join omop.cmd.location l1
	on p1.location_id = l1.location_id
where p1.location_id is not null
	and l1.state is not null
group by l1.state;
ANALYZE s_tmpach_1101
;


-- 1102	Number of care sites by location 3-digit zip
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1102
AS
WITH rawData  AS (
  select
    left(l1.zip,3) as stratum_1,
    COUNT(distinct care_site_id) as count_value
  from omop.cmd.care_site cs1
    inner join omop.cmd.location l1
    on cs1.location_id = l1.location_id
  where cs1.location_id is not null
    and l1.zip is not null
  group by left(l1.zip,3)
)
 SELECT
1102 as analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) as stratum_1,
  cast(null as varchar(255)) as stratum_2,
  cast(null as varchar(255)) as stratum_3,
  cast(null as varchar(255)) as stratum_4,
  cast(null as varchar(255)) as stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_1102
;


-- 1103	Number of care sites by location state
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1103
AS
SELECT
1103 as analysis_id,  
	CAST(l1.state AS VARCHAR(255)) as stratum_1, 
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct care_site_id) as count_value
FROM
omop.cmd.care_site cs1
	inner join omop.cmd.location l1
	on cs1.location_id = l1.location_id
where cs1.location_id is not null
	and l1.state is not null
group by l1.state;
ANALYZE s_tmpach_1103
;


-- 1200	Number of persons by place of service
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1200
AS
SELECT
1200 as analysis_id,  
	CAST(cs1.place_of_service_concept_id AS VARCHAR(255)) as stratum_1, 
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(person_id) as count_value
FROM
omop.cmd.person p1
	inner join omop.cmd.care_site cs1
	on p1.care_site_id = cs1.care_site_id
where p1.care_site_id is not null
	and cs1.place_of_service_concept_id is not null
group by cs1.place_of_service_concept_id;
ANALYZE s_tmpach_1200
;


-- 1201	Number of visits by place of service
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1201
AS
SELECT
1201 as analysis_id,  
	CAST(cs1.place_of_service_concept_id AS VARCHAR(255)) as stratum_1, 
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(visit_occurrence_id) as count_value
FROM
omop.cmd.visit_occurrence vo1
	inner join omop.cmd.care_site cs1
	on vo1.care_site_id = cs1.care_site_id
where vo1.care_site_id is not null
	and cs1.place_of_service_concept_id is not null
group by cs1.place_of_service_concept_id;
ANALYZE s_tmpach_1201
;


-- 1202	Number of care sites by place of service
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1202
AS
SELECT
1202 as analysis_id,  
	CAST(cs1.place_of_service_concept_id AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(care_site_id) as count_value
FROM
omop.cmd.care_site cs1
where cs1.place_of_service_concept_id is not null
group by cs1.place_of_service_concept_id;
ANALYZE s_tmpach_1202
;


-- 1203	Number of visits by place of service discharge type
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1203
AS
SELECT
1203 AS analysis_id,
	CAST(vo.discharge_to_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date
WHERE 
	vo.discharge_to_concept_id != 0
GROUP BY 
	vo.discharge_to_concept_id;
ANALYZE s_tmpach_1203
;


-- 1300	Number of persons with at least one visit detail, by visit_detail_concept_id
-- restricted to visits overlapping with observation period
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1300
AS
SELECT
1300 AS analysis_id,
	CAST(vd.visit_detail_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT vd.person_id) AS count_value
FROM
omop.cmd.visit_detail vd
JOIN 
    omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	vd.visit_detail_concept_id
;
ANALYZE s_tmpach_1300
;


-- 1301	Number of visit detail records, by visit_detail_concept_id
-- restricted to visits overlapping with observation period
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1301
AS
SELECT
1301 AS analysis_id,
	CAST(vd.visit_detail_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(vd.person_id) AS count_value
FROM
omop.cmd.visit_detail vd
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	vd.visit_detail_concept_id;
ANALYZE s_tmpach_1301
;


-- 1302	Number of persons by visit detail start month, by visit_detail_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1302
AS
WITH rawData  AS (
SELECT
	vd.visit_detail_concept_id AS stratum_1,
	EXTRACT(YEAR FROM vd.visit_detail_start_date)*100 + EXTRACT(MONTH FROM vd.visit_detail_start_date) AS stratum_2,
	COUNT(DISTINCT vd.person_id) AS count_value
FROM
	omop.cmd.visit_detail vd 
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	vd.visit_detail_concept_id,
	EXTRACT(YEAR FROM vd.visit_detail_start_date)*100 + EXTRACT(MONTH FROM vd.visit_detail_start_date)
)
 SELECT
1302 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1302
;


-- 1303	Number of distinct visit detail concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_1303
AS
WITH rawData(person_id, count_value)  AS (
SELECT 
	vd.person_id,
	COUNT(DISTINCT vd.visit_detail_concept_id) AS count_value
FROM 
	omop.cmd.visit_detail vd
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	vd.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) AS
(
SELECT 
	CAST(AVG(1.0 * count_value) AS NUMERIC) AS avg_value,
	CAST(STDDEV(count_value) AS NUMERIC) AS stdev_value,
	MIN(count_value) AS min_value,
	MAX(count_value) AS max_value,
	COUNT(*) AS total
FROM 
	rawData
),
statsView (count_value, total, rn) AS
(
SELECT 
	count_value,
	COUNT(*) AS total,
	ROW_NUMBER() OVER (ORDER BY count_value) AS rn
FROM 
	rawData
GROUP BY 
	count_value
),
priorStats (count_value, total, accumulated) AS
(
SELECT 
	s.count_value,
	s.total,
	SUM(p.total) AS accumulated
FROM 
	statsView s
JOIN 
	statsView p ON p.rn <= s.rn
GROUP BY 
	s.count_value,
	s.total,
	s.rn
)
 SELECT
1303 AS analysis_id,
	o.total AS count_value,
	o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(CASE WHEN p.accumulated >= .50 * o.total THEN count_value ELSE o.max_value END) AS median_value,
	MIN(CASE WHEN p.accumulated >= .10 * o.total THEN count_value ELSE o.max_value END) AS p10_value,
	MIN(CASE WHEN p.accumulated >= .25 * o.total THEN count_value ELSE o.max_value END) AS p25_value,
	MIN(CASE WHEN p.accumulated >= .75 * o.total THEN count_value ELSE o.max_value END) AS p75_value,
	MIN(CASE WHEN p.accumulated >= .90 * o.total THEN count_value ELSE o.max_value END) AS p90_value
FROM
priorStats p
CROSS JOIN 
	overallStats o
GROUP BY 
	o.total,
	o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value
;
ANALYZE tempResults_1303
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_1303
AS
SELECT
analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value,
	min_value,
	max_value,
	avg_value,
	stdev_value,
	median_value,
	p10_value,
	p25_value,
	p75_value,
	p90_value
FROM
tempResults_1303
;
ANALYZE s_tmpach_dist_1303
;
TRUNCATE TABLE tempResults_1303;
DROP TABLE tempResults_1303;


-- 1304	Number of persons with at least one visit detail, by visit_detail_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1304
AS
WITH rawData  AS (
SELECT 
	vd.visit_detail_concept_id AS stratum_1,
	EXTRACT(YEAR FROM vd.visit_detail_start_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM vd.visit_detail_start_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.visit_detail vd 
ON 
	p.person_id = vd.person_id 
JOIN
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	vd.visit_detail_concept_id,
    EXTRACT(YEAR FROM vd.visit_detail_start_date),
    p.gender_concept_id,
    FLOOR((EXTRACT(YEAR FROM vd.visit_detail_start_date) - p.year_of_birth)/10)
)
 SELECT
1304 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1304
;


-- 1306	Distribution of age by visit_detail_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_1306
AS
WITH rawData (stratum1_id, stratum2_id, count_value)  AS (
SELECT 
	vd.visit_detail_concept_id AS stratum1_id,
	p.gender_concept_id AS stratum2_id,
	vd.visit_detail_start_year - p.year_of_birth AS count_value
FROM 
	omop.cmd.person p
JOIN (
	SELECT 
		vd.person_id,
		vd.visit_detail_concept_id,
		MIN(EXTRACT(YEAR FROM vd.visit_detail_start_date)) AS visit_detail_start_year
	FROM 
		omop.cmd.visit_detail vd
	JOIN 
		omop.cmd.observation_period op 
	ON 
		vd.person_id = op.person_id
	AND	
		vd.visit_detail_start_date >= op.observation_period_start_date  
	AND 
		vd.visit_detail_start_date <= op.observation_period_end_date
	GROUP BY 
		vd.person_id,
		vd.visit_detail_concept_id
	) vd 
ON 
	p.person_id = vd.person_id
),
overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total) AS
(
SELECT 
	stratum1_id,
	stratum2_id,
	CAST(AVG(1.0 * count_value) AS NUMERIC) AS avg_value,
	CAST(STDDEV(count_value) AS NUMERIC) AS stdev_value,
	MIN(count_value) AS min_value,
	MAX(count_value) AS max_value,
	COUNT(*) AS total
FROM 
	rawData
GROUP BY 
	stratum1_id,
	stratum2_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) AS
(
SELECT 
	stratum1_id,
	stratum2_id,
	count_value,
	COUNT(*) AS total,
	ROW_NUMBER() OVER (PARTITION BY stratum1_id,stratum2_id ORDER BY count_value) AS rn
FROM 
	rawData
GROUP BY 
	stratum1_id,
	stratum2_id,
	count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) AS
(
SELECT 
	s.stratum1_id,
	s.stratum2_id,
	s.count_value,
	s.total,
	SUM(p.total) AS accumulated
FROM 
	statsView s
JOIN 
	statsView p ON s.stratum1_id = p.stratum1_id 
				AND s.stratum2_id = p.stratum2_id 
				AND p.rn <= s.rn
GROUP BY 
	s.stratum1_id,
	s.stratum2_id,
	s.count_value,
	s.total,
	s.rn
)
 SELECT
1306 AS analysis_id,
	CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
	CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
	o.total as count_value,
	o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(CASE WHEN p.accumulated >= .50 * o.total THEN count_value ELSE o.max_value END) AS median_value,
	MIN(CASE WHEN p.accumulated >= .10 * o.total THEN count_value ELSE o.max_value END) AS p10_value,
	MIN(CASE WHEN p.accumulated >= .25 * o.total THEN count_value ELSE o.max_value END) AS p25_value,
	MIN(CASE WHEN p.accumulated >= .75 * o.total THEN count_value ELSE o.max_value END) AS p75_value,
	MIN(CASE WHEN p.accumulated >= .90 * o.total THEN count_value ELSE o.max_value END) AS p90_value
FROM
priorStats p
JOIN 
	overallStats o ON p.stratum1_id = o.stratum1_id AND p.stratum2_id = o.stratum2_id 
GROUP BY 
	o.stratum1_id, 
	o.stratum2_id, 
	o.total, 
	o.min_value, 
	o.max_value, 
	o.avg_value, 
	o.stdev_value
;
ANALYZE tempResults_1306
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1306
AS
SELECT
analysis_id,
	stratum1_id AS stratum_1,
	stratum2_id AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value,
	min_value,
	max_value,
	avg_value,
	stdev_value,
	median_value,
	p10_value,
	p25_value,
	p75_value,
	p90_value
FROM
tempResults_1306
;
ANALYZE s_tmpach_dist_1306
;
TRUNCATE TABLE tempResults_1306;
DROP TABLE tempResults_1306;


-- 1307	Number of visit detail records with invalid person_id
CREATE TEMP TABLE s_tmpach_1307
AS
SELECT
1307 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(vd.person_id) AS count_value
FROM
omop.cmd.visit_detail vd
LEFT JOIN 
	omop.cmd.person p 
ON 
	p.person_id = vd.person_id
WHERE 
	p.person_id IS NULL;
ANALYZE s_tmpach_1307
;


-- 1309	Number of visit detail records with invalid care_site_id
CREATE TEMP TABLE s_tmpach_1309
AS
SELECT
1309 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(vd.person_id) AS count_value
FROM
omop.cmd.visit_detail vd
LEFT JOIN 
	omop.cmd.care_site cs 
ON 
	vd.care_site_id = cs.care_site_id
WHERE 
	vd.care_site_id IS NOT NULL 
AND 
	cs.care_site_id IS NULL;
ANALYZE s_tmpach_1309
;


-- 1310	Number of visit_detail records outside a valid observation period
CREATE TEMP TABLE s_tmpach_1310
AS
SELECT
1310 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.visit_detail vd
LEFT JOIN 
	omop.cmd.observation_period op
ON 
	op.person_id = vd.person_id
AND 
	vd.visit_detail_start_date >= op.observation_period_start_date
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
WHERE 
	op.person_id IS NULL
;
ANALYZE s_tmpach_1310
;


-- 1311	Number of visit detail records with end date < start date
CREATE TEMP TABLE s_tmpach_1311
AS
SELECT
1311 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(person_id) AS count_value
FROM
omop.cmd.visit_detail
WHERE 
	visit_detail_end_date < visit_detail_start_date;
ANALYZE s_tmpach_1311
;


-- 1312	Number of persons with at least one visit detail by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1312
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM vd.visit_detail_start_date) AS stratum_1,
	p.gender_concept_id AS stratum_2,
	FLOOR((EXTRACT(YEAR FROM vd.visit_detail_start_date) - p.year_of_birth) / 10) AS stratum_3,
	COUNT(DISTINCT vd.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.visit_detail vd 
ON 
	vd.person_id = p.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM vd.visit_detail_start_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM vd.visit_detail_start_date) - p.year_of_birth) / 10)
)
 SELECT
1312 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS varchar(255)) AS stratum_2,
	CAST(stratum_3 AS varchar(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1312
;


-- 1313	Distribution of length of stay by visit_detail_concept_id
-- restrict to visits inside observation period
--HINT DISTRIBUTE_ON_KEY(stratum_id) 
CREATE TEMP TABLE tempResults_1313
AS
WITH rawData(stratum_id, count_value)  AS (
SELECT 
	vd.visit_detail_concept_id AS stratum_id,
	(CAST(vd.visit_detail_END_date AS DATE) - CAST(vd.visit_detail_start_date AS DATE)) AS count_value
FROM 
	omop.cmd.visit_detail vd
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
),
overallStats (stratum_id, avg_value, stdev_value, min_value, max_value, total) AS
(
SELECT 
	stratum_id,
	CAST(AVG(1.0 * count_value) AS NUMERIC) AS avg_value,
	CAST(STDDEV(count_value) AS NUMERIC) AS stdev_value,
	MIN(count_value) AS min_value,
	MAX(count_value) AS max_value,
	COUNT(*) AS total
FROM 
	rawData
GROUP BY 
	stratum_id
),
statsView (stratum_id, count_value, total, rn) AS
(
SELECT 
	stratum_id,
	count_value,
	COUNT(*) AS total,
	ROW_NUMBER() OVER (ORDER BY count_value) AS rn
FROM 
	rawData
GROUP BY 
	stratum_id,
	count_value
),
priorStats (stratum_id, count_value, total, accumulated) AS
(
SELECT 
	s.stratum_id,
	s.count_value,
	s.total,
	SUM(p.total) AS accumulated
FROM 
	statsView s
JOIN 
	statsView p 
ON 
	s.stratum_id = p.stratum_id
AND 
	p.rn <= s.rn
GROUP BY 
	s.stratum_id,
	s.count_value,
	s.total,
	s.rn
)
 SELECT
1313 AS analysis_id,
	CAST(o.stratum_id AS VARCHAR(255)) AS stratum_id,
	o.total as count_value,
	o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(CASE WHEN p.accumulated >= .50 * o.total THEN count_value ELSE o.max_value END) AS median_value,
	MIN(CASE WHEN p.accumulated >= .10 * o.total THEN count_value ELSE o.max_value END) AS p10_value,
	MIN(CASE WHEN p.accumulated >= .25 * o.total THEN count_value ELSE o.max_value END) AS p25_value,
	MIN(CASE WHEN p.accumulated >= .75 * o.total THEN count_value ELSE o.max_value END) AS p75_value,
	MIN(CASE WHEN p.accumulated >= .90 * o.total THEN count_value ELSE o.max_value END) AS p90_value
FROM
priorStats p
JOIN 
	overallStats o ON p.stratum_id = o.stratum_id
GROUP BY 
	o.stratum_id, 
	o.total, 
	o.min_value, 
	o.max_value, 
	o.avg_value, 
	o.stdev_value
;
ANALYZE tempResults_1313
;
--HINT DISTRIBUTE_ON_KEY(stratum_1) 
CREATE TEMP TABLE s_tmpach_dist_1313
AS
SELECT
analysis_id,
	stratum_id AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value,
	min_value,
	max_value,
	avg_value,
	stdev_value,
	median_value,
	p10_value,
	p25_value,
	p75_value,
	p90_value
FROM
tempResults_1313;
ANALYZE s_tmpach_dist_1313
;
TRUNCATE TABLE tempResults_1313;
DROP TABLE tempResults_1313;


-- 1320	Number of visit detail records by visit detail start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1320
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM vd.visit_detail_start_date) * 100 + EXTRACT(MONTH FROM vd.visit_detail_start_date) AS stratum_1,
	COUNT(vd.person_id) AS count_value
FROM 
	omop.cmd.visit_detail vd
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM vd.visit_detail_start_date) * 100 + EXTRACT(MONTH FROM vd.visit_detail_start_date)
)
 SELECT
1320 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1320
;


-- 1321	Number of persons by visit start year 
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1321
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM vd.visit_detail_start_date) AS stratum_1,
	COUNT(DISTINCT vd.person_id) AS count_value
FROM 
	omop.cmd.visit_detail vd
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	EXTRACT(YEAR FROM vd.visit_detail_start_date)
)
 SELECT
1321 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(NULL AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_1321
;


-- 1325	Number of visit_detail records, by visit_detail_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1325
AS
SELECT
1325 AS analysis_id,
	CAST(vd.visit_detail_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.visit_detail vd
JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND	
	vd.visit_detail_start_date >= op.observation_period_start_date  
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
GROUP BY 
	visit_detail_source_concept_id;
ANALYZE s_tmpach_1325
;


-- 1326	Number of records by domain by visit detail concept id
CREATE TEMP TABLE s_tmpach_1326
AS
SELECT
1326 AS analysis_id,
	CAST(v.visit_detail_concept_id AS VARCHAR(255)) AS stratum_1,
	v.cdm_table AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	v.record_count AS count_value
FROM
(
	SELECT 'drug_exposure' cdm_table,
		COALESCE(vd.visit_detail_concept_id, 0) visit_detail_concept_id,
		COUNT(*) record_count
	FROM 
		omop.cmd.drug_exposure de
	LEFT JOIN 
		omop.cmd.visit_detail vd 
	ON 
		de.visit_occurrence_id = vd.visit_occurrence_id
	GROUP BY 
		vd.visit_detail_concept_id
	UNION
	SELECT 
		'condition_occurrence' cdm_table,
		COALESCE(vd.visit_detail_concept_id, 0) visit_detail_concept_id,
		COUNT(*) record_count
	FROM 
		omop.cmd.condition_occurrence co
	LEFT JOIN 
		omop.cmd.visit_detail vd 
	ON 
		co.visit_occurrence_id = vd.visit_occurrence_id
	GROUP BY 
		vd.visit_detail_concept_id
	UNION
	SELECT 
		'device_exposure' cdm_table,
		COALESCE(visit_detail_concept_id, 0) visit_detail_concept_id,
		COUNT(*) record_count
	FROM 
		omop.cmd.device_exposure de
	LEFT JOIN 
		omop.cmd.visit_detail vd 
	ON 
		de.visit_occurrence_id = vd.visit_occurrence_id
	GROUP BY 
		vd.visit_detail_concept_id
	UNION
	SELECT 
		'procedure_occurrence' cdm_table,
		COALESCE(vd.visit_detail_concept_id, 0) visit_detail_concept_id,
		COUNT(*) record_count
	FROM 
		omop.cmd.procedure_occurrence po
	LEFT JOIN 
		omop.cmd.visit_detail vd 
	ON 
		po.visit_occurrence_id = vd.visit_occurrence_id
	GROUP BY 
		vd.visit_detail_concept_id
	UNION
	SELECT 
		'measurement' cdm_table,
		COALESCE(vd.visit_detail_concept_id, 0) visit_detail_concept_id,
		COUNT(*) record_count
	FROM 
		omop.cmd.measurement m
	LEFT JOIN 
		omop.cmd.visit_detail vd 
	ON 
		m.visit_occurrence_id = vd.visit_occurrence_id
	GROUP BY 
		vd.visit_detail_concept_id
	UNION
	SELECT 
		'observation' cdm_table,
		COALESCE(vd.visit_detail_concept_id, 0) visit_detail_concept_id,
		COUNT(*) record_count
	FROM 
		omop.cmd.observation o
	LEFT JOIN 
		omop.cmd.visit_detail vd 
	ON 
		o.visit_occurrence_id = vd.visit_occurrence_id
	GROUP BY 
		vd.visit_detail_concept_id
	) v;
ANALYZE s_tmpach_1326
;


-- 1330	Number of visit_detail records inside a valid observation period
CREATE TEMP TABLE s_tmpach_1330
AS
SELECT
1330 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.visit_detail vd
JOIN 
	omop.cmd.observation_period op
ON 
	op.person_id = vd.person_id
AND 
	vd.visit_detail_start_date >= op.observation_period_start_date
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_1330
;


-- 1331	Proportion of people with at least one visit_detail record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in visit_detail (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_1331
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT vd.person_id) AS person_count
FROM 
	omop.cmd.visit_detail vd
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND 
	vd.visit_detail_start_date >= op.observation_period_start_date
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), vd_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.visit_detail
)
 SELECT
1331 AS analysis_id,
	CASE WHEN vdt.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/vdt.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(vdt.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	vd_total vdt
;
ANALYZE s_tmpach_1331
;


-- 1332	Proportion of visit_detail records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of visit_detail records outside a valid observation period (numerator)
-- stratum_3:   Number of visit_detail records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_1332
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.visit_detail vd
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	vd.person_id = op.person_id
AND 
	vd.visit_detail_start_date >= op.observation_period_start_date
AND 
	vd.visit_detail_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), vd_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.visit_detail
)
 SELECT
1332 AS analysis_id,
	CASE WHEN vdt.record_count != 0 THEN 
		CAST(CAST(1.0*op.record_count/vdt.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(vdt.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	vd_total vdt
;
ANALYZE s_tmpach_1332
;


-- 1406	Length of payer plan (days) of first payer plan period by gender
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE tempResults_1406
AS
WITH rawData(stratum1_id, count_value)  AS (
  select p1.gender_concept_id as stratum1_id,
    (CAST(ppp1.payer_plan_period_end_date AS DATE) - CAST(ppp1.payer_plan_period_start_date AS DATE)) as count_value
  from omop.cmd.person p1
	inner join 
	(select person_id, 
		payer_plan_period_START_DATE, 
		payer_plan_period_END_DATE, 
		ROW_NUMBER() over (PARTITION by person_id order by payer_plan_period_start_date asc) as rn1
		 from omop.cmd.payer_plan_period
	) ppp1
	on p1.PERSON_ID = ppp1.PERSON_ID
	where ppp1.rn1 = 1
),
overallStats (stratum1_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum1_id, 
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
  group by stratum1_id
),
statsView (stratum1_id, count_value, total, rn) as
(
  select stratum1_id, 
  	count_value, 
  	COUNT(*) as total, 
		row_number() over (partition by stratum1_id order by count_value) as rn
  FROM rawData
  group by stratum1_id, count_value
),
priorStats (stratum1_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and p.rn <= s.rn
  group by s.stratum1_id, s.count_value, s.total, s.rn
)
 SELECT
1406 as analysis_id,
  CAST(p.stratum1_id AS VARCHAR(255)) as stratum_1,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id
GROUP BY p.stratum1_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1406
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1406
AS
SELECT
analysis_id, stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1406
;
ANALYZE s_tmpach_dist_1406
;
truncate table tempResults_1406;
drop table tempResults_1406;


-- 1407	Length of payer plan (days) of first payer plan period by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_id)
CREATE TEMP TABLE tempResults_1407
AS
WITH rawData(stratum_id, count_value)  AS (
  select floor((EXTRACT(YEAR FROM ppp1.payer_plan_period_START_DATE) - p1.YEAR_OF_BIRTH)/10) as stratum_id,
    (CAST(ppp1.payer_plan_period_end_date AS DATE) - CAST(ppp1.payer_plan_period_start_date AS DATE)) as count_value
  from omop.cmd.person p1
	inner join 
	(select person_id, 
		payer_plan_period_START_DATE, 
		payer_plan_period_END_DATE, 
		ROW_NUMBER() over (PARTITION by person_id order by payer_plan_period_start_date asc) as rn1
		 from omop.cmd.payer_plan_period
	) ppp1
	on p1.PERSON_ID = ppp1.PERSON_ID
	where ppp1.rn1 = 1
),
overallStats (stratum_id, avg_value, stdev_value, min_value, max_value, total) as
(
  select stratum_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData
  group by stratum_id
),
statsView (stratum_id, count_value, total, rn) as
(
  select stratum_id, count_value, COUNT(*) as total, row_number() over (order by count_value) as rn
  FROM rawData
  group by stratum_id, count_value
),
priorStats (stratum_id, count_value, total, accumulated) as
(
  select s.stratum_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum_id = p.stratum_id and p.rn <= s.rn
  group by s.stratum_id, s.count_value, s.total, s.rn
)
 SELECT
1407 as analysis_id,
  CAST(o.stratum_id AS VARCHAR(255)) AS stratum_id,
  o.total as count_value,
  o.min_value,
  o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum_id = o.stratum_id
GROUP BY o.stratum_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1407
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1407
AS
SELECT
analysis_id, stratum_id as stratum_1, 
cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1407
;
ANALYZE s_tmpach_dist_1407
;
truncate table tempResults_1407;
drop table tempResults_1407;


-- 1408	Number of persons by length of payer plan period, in 30d increments
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1408
AS
SELECT
1408 as analysis_id,  
	CAST(floor((CAST(ppp1.payer_plan_period_end_date AS DATE) - CAST(ppp1.payer_plan_period_start_date AS DATE))/30) AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.person_id) as count_value
FROM
omop.cmd.person p1
	inner join 
	(select person_id, 
		payer_plan_period_START_DATE, 
		payer_plan_period_END_DATE, 
		ROW_NUMBER() over (PARTITION by person_id order by payer_plan_period_start_date asc) as rn1
		 from omop.cmd.payer_plan_period
	) ppp1
	on p1.PERSON_ID = ppp1.PERSON_ID
	where ppp1.rn1 = 1
group by CAST(floor((CAST(ppp1.payer_plan_period_end_date AS DATE) - CAST(ppp1.payer_plan_period_start_date AS DATE))/30) AS VARCHAR(255))
;
ANALYZE s_tmpach_1408
;


-- 1409	Number of persons with continuous payer plan in each year
-- Note: using temp table instead of nested query because this gives vastly improved
CREATE TEMP TABLE temp_dates_1409
AS
SELECT
distinct 
  EXTRACT(YEAR FROM payer_plan_period_start_date) as obs_year 
FROM
omop.cmd.payer_plan_period
;
ANALYZE temp_dates_1409
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1409
AS
SELECT
1409 as analysis_id,  
	CAST(t1.obs_year AS VARCHAR(255)) as stratum_1, 
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.PERSON_ID) as count_value
FROM
omop.cmd.person p1
	inner join 
    omop.cmd.payer_plan_period ppp1
	on p1.person_id = ppp1.person_id
	,
	temp_dates_1409 t1 
where EXTRACT(YEAR FROM ppp1.payer_plan_period_START_DATE) <= t1.obs_year
	and EXTRACT(YEAR FROM ppp1.payer_plan_period_END_DATE) >= t1.obs_year
group by t1.obs_year
;
ANALYZE s_tmpach_1409
;
truncate table temp_dates_1409;
drop table temp_dates_1409;


-- 1410	Number of persons with continuous payer plan in each month
-- Note: using temp table instead of nested query because this gives vastly improved performance in Oracle
--HINT DISTRIBUTE_ON_KEY(obs_month)
CREATE TEMP TABLE temp_dates_1410
AS
SELECT
DISTINCT 
  EXTRACT(YEAR FROM payer_plan_period_start_date)*100 + EXTRACT(MONTH FROM payer_plan_period_start_date) AS obs_month,
  TO_DATE(TO_CHAR(EXTRACT(YEAR FROM payer_plan_period_start_date),'0000')||'-'||TO_CHAR(EXTRACT(MONTH FROM payer_plan_period_start_date),'00')||'-'||TO_CHAR(1,'00'), 'YYYY-MM-DD') as obs_month_start,
  (DATE_TRUNC('MONTH', payer_plan_period_start_date) + INTERVAL '1 MONTH - 1 day')::DATE as obs_month_end
FROM
omop.cmd.payer_plan_period
;
ANALYZE temp_dates_1410
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1410
AS
SELECT
1410 as analysis_id, 
	CAST(obs_month AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.PERSON_ID) as count_value
FROM
omop.cmd.person p1
	inner join 
  omop.cmd.payer_plan_period ppp1
	on p1.person_id = ppp1.person_id
	,
	temp_dates_1410
where ppp1.payer_plan_period_START_DATE <= obs_month_start
	and ppp1.payer_plan_period_END_DATE >= obs_month_end
group by obs_month
;
ANALYZE s_tmpach_1410
;
TRUNCATE TABLE temp_dates_1410;
DROP TABLE temp_dates_1410;


-- 1411	Number of persons by payer plan period start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1411
AS
SELECT
1411 as analysis_id, 
	TO_DATE(TO_CHAR(EXTRACT(YEAR FROM payer_plan_period_start_date),'0000')||'-'||TO_CHAR(EXTRACT(MONTH FROM payer_plan_period_start_date),'00')||'-'||TO_CHAR(1,'00'), 'YYYY-MM-DD') as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.PERSON_ID) as count_value
FROM
omop.cmd.person p1
	inner join omop.cmd.payer_plan_period ppp1
	on p1.person_id = ppp1.person_id
group by TO_DATE(TO_CHAR(EXTRACT(YEAR FROM payer_plan_period_start_date),'0000')||'-'||TO_CHAR(EXTRACT(MONTH FROM payer_plan_period_start_date),'00')||'-'||TO_CHAR(1,'00'), 'YYYY-MM-DD')
;
ANALYZE s_tmpach_1411
;


-- 1412	Number of persons by payer plan period end month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1412
AS
SELECT
1412 as analysis_id,  
	TO_DATE(TO_CHAR(EXTRACT(YEAR FROM payer_plan_period_start_date),'0000')||'-'||TO_CHAR(EXTRACT(MONTH FROM payer_plan_period_start_date),'00')||'-'||TO_CHAR(1,'00'), 'YYYY-MM-DD') as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.PERSON_ID) as count_value
FROM
omop.cmd.person p1
	inner join omop.cmd.payer_plan_period ppp1
	on p1.person_id = ppp1.person_id
group by TO_DATE(TO_CHAR(EXTRACT(YEAR FROM payer_plan_period_start_date),'0000')||'-'||TO_CHAR(EXTRACT(MONTH FROM payer_plan_period_start_date),'00')||'-'||TO_CHAR(1,'00'), 'YYYY-MM-DD')
;
ANALYZE s_tmpach_1412
;


-- 1413	Number of persons by number of payer plan periods
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1413
AS
SELECT
1413 as analysis_id,  
	CAST(ppp1.num_periods AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.PERSON_ID) as count_value
FROM
omop.cmd.person p1
	inner join (select person_id, COUNT(payer_plan_period_start_date) as num_periods from omop.cmd.payer_plan_period group by PERSON_ID) ppp1
	on p1.person_id = ppp1.person_id
group by ppp1.num_periods
;
ANALYZE s_tmpach_1413
;


-- 1414	Number of persons with payer plan period before year-of-birth
CREATE TEMP TABLE s_tmpach_1414
AS
SELECT
1414 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct p1.PERSON_ID) as count_value
FROM
omop.cmd.person p1
	inner join (select person_id, MIN(EXTRACT(YEAR FROM payer_plan_period_start_date)) as first_obs_year from omop.cmd.payer_plan_period group by PERSON_ID) ppp1
	on p1.person_id = ppp1.person_id
where p1.year_of_birth > ppp1.first_obs_year
;
ANALYZE s_tmpach_1414
;


-- 1415	Number of persons with payer plan period end < start
CREATE TEMP TABLE s_tmpach_1415
AS
SELECT
1415 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(ppp1.PERSON_ID) as count_value
FROM
omop.cmd.payer_plan_period ppp1
where ppp1.payer_plan_period_end_date < ppp1.payer_plan_period_start_date
;
ANALYZE s_tmpach_1415
;


-- 1425	Number of payer_plan_period records, by payer_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1425 
AS
SELECT
1425 as analysis_id,
       cast(payer_source_concept_id AS varchar(255)) AS stratum_1,
       cast(null AS varchar(255)) AS stratum_2,
       cast(null as varchar(255)) as stratum_3,
       cast(null as varchar(255)) as stratum_4,
       cast(null as varchar(255)) as stratum_5,
       COUNT(*) AS count_value
FROM
omop.cmd.payer_plan_period
 group by payer_source_concept_id;
ANALYZE s_tmpach_1425 
  ;


-- 1800	Number of persons with at least one measurement occurrence, by measurement_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1800
AS
SELECT
1800 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT m.person_id) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date	
GROUP BY 
	m.measurement_concept_id;
ANALYZE s_tmpach_1800
;


-- 1801	Number of measurement occurrence records, by measurement_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1801
AS
SELECT
1801 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(m.person_id) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date	
GROUP BY 
	m.measurement_concept_id;
ANALYZE s_tmpach_1801
;


-- 1802	Number of persons by measurement occurrence start month, by measurement_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1802
AS
WITH rawData  AS (
SELECT 
	m.measurement_concept_id AS stratum_1,
	EXTRACT(YEAR FROM m.measurement_date) * 100 + EXTRACT(MONTH FROM m.measurement_date) AS stratum_2,
	COUNT(DISTINCT m.person_id) AS count_value
FROM 
	omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date	
GROUP BY 
	m.measurement_concept_id,
	EXTRACT(YEAR FROM m.measurement_date) * 100 + EXTRACT(MONTH FROM m.measurement_date)
)
 SELECT
1802 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_1802
;


-- 1803	Number of distinct measurement occurrence concepts per person
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE tempResults_1803
AS
WITH rawData(count_value)  AS (
SELECT 
	COUNT(DISTINCT m.measurement_concept_id) AS count_value
FROM 
	omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.person_id
),
overallStats (avg_value, stdev_value, min_value, max_value, total) as
(
  select CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  from rawData
),
statsView (count_value, total, rn) as
(
  select count_value, 
  	COUNT(*) as total, 
		row_number() over (order by count_value) as rn
  FROM rawData
  group by count_value
),
priorStats (count_value, total, accumulated) as
(
  select s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on p.rn <= s.rn
  group by s.count_value, s.total, s.rn
)
 SELECT
1803 as analysis_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
CROSS JOIN overallStats o
GROUP BY o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1803
;
--HINT DISTRIBUTE_ON_KEY(count_value)
CREATE TEMP TABLE s_tmpach_dist_1803
AS
SELECT
analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1803
;
ANALYZE s_tmpach_dist_1803
;
truncate table tempResults_1803;
drop table tempResults_1803;


-- 1804	Number of persons with at least one measurement occurrence, by measurement_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1804
AS
WITH rawData  AS (
SELECT 
	m.measurement_concept_id AS stratum_1,
	EXTRACT(YEAR FROM m.measurement_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM m.measurement_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.measurement m 
ON 
	p.person_id = m.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.measurement_concept_id,
	EXTRACT(YEAR FROM m.measurement_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM m.measurement_date) - p.year_of_birth) / 10)
)
 SELECT
1804 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1804
;


-- 1805	Number of measurement records, by measurement_concept_id by measurement_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1805
AS
SELECT
1805 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(m.measurement_type_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(m.person_id) AS count_value
FROM
omop.cmd.measurement m
	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.measurement_concept_id,
	m.measurement_type_concept_id;
ANALYZE s_tmpach_1805
;


-- 1806	Distribution of age by measurement_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_1806
AS
SELECT
o.measurement_concept_id AS subject_id,
	p.gender_concept_id,
	o.measurement_start_year - p.year_of_birth AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		m.person_id,
		m.measurement_concept_id,
		MIN(EXTRACT(YEAR FROM m.measurement_date)) AS measurement_start_year
	FROM 
		omop.cmd.measurement m
	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
	GROUP BY 
		m.person_id,
		m.measurement_concept_id
	) o
ON 
	p.person_id = o.person_id
;
ANALYZE rawData_1806
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_1806
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_1806
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_1806
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
1806 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1806
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1806
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1806
;
ANALYZE s_tmpach_dist_1806
;
truncate table rawData_1806;
drop table rawData_1806;
truncate table tempResults_1806;
drop table tempResults_1806;


-- 1807	Number of measurement occurrence records, by measurement_concept_id and unit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1807
AS
SELECT
1807 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(m.unit_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(m.person_id) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.measurement_concept_id,
	m.unit_concept_id;
ANALYZE s_tmpach_1807
;


-- 1809	Number of measurement records with invalid person_id
CREATE TEMP TABLE s_tmpach_1809
AS
SELECT
1809 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(m.PERSON_ID) as count_value
FROM
omop.cmd.measurement m
	left join omop.cmd.person p1 on p1.person_id = m.person_id
where p1.person_id is null
;
ANALYZE s_tmpach_1809
;


-- 1810	Number of measurement records outside valid observation period
CREATE TEMP TABLE s_tmpach_1810
AS
SELECT
1810 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(m.PERSON_ID) as count_value
FROM
omop.cmd.measurement m
	left join omop.cmd.observation_period op on op.person_id = m.person_id
	and m.measurement_date >= op.observation_period_start_date
	and m.measurement_date <= op.observation_period_end_date
where op.person_id is null
;
ANALYZE s_tmpach_1810
;


-- 1811	Number of measurement records with a value (with a mapped, non-null value_as_number)
CREATE TEMP TABLE s_tmpach_1811
AS
SELECT
1811 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(m.person_id) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
WHERE 
	m.value_as_number IS NOT NULL
OR 
	m.value_as_concept_id != 0;
ANALYZE s_tmpach_1811
;


-- 1812	Number of measurement records with invalid provider_id
CREATE TEMP TABLE s_tmpach_1812
AS
SELECT
1812 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(m.PERSON_ID) as count_value
FROM
omop.cmd.measurement m
	left join omop.cmd.provider p on p.provider_id = m.provider_id
where m.provider_id is not null
	and p.provider_id is null
;
ANALYZE s_tmpach_1812
;


-- 1813	Number of observation records with invalid visit_id
CREATE TEMP TABLE s_tmpach_1813
AS
SELECT
1813 as analysis_id, 
cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
COUNT(m.PERSON_ID) as count_value
FROM
omop.cmd.measurement m
	left join omop.cmd.visit_occurrence vo on m.visit_occurrence_id = vo.visit_occurrence_id
where m.visit_occurrence_id is not null
	and vo.visit_occurrence_id is null
;
ANALYZE s_tmpach_1813
;


-- 1814	Number of measurement records with no value (numeric or concept)
CREATE TEMP TABLE s_tmpach_1814
AS
SELECT
1814 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(m.person_id) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
WHERE 
	m.value_as_number IS NULL
AND 
	(m.value_as_concept_id IS NULL OR m.value_as_concept_id = 0);
ANALYZE s_tmpach_1814
;


-- 1815  Distribution of numeric values, by measurement_concept_id and unit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE statsView_1815
AS
SELECT
m.subject_id AS stratum1_id,
	m.unit_concept_id AS stratum2_id,
	m.count_value,
	COUNT(*) AS total,
	ROW_NUMBER() OVER (PARTITION BY m.subject_id,m.unit_concept_id ORDER BY m.count_value) AS rn
FROM
(
	SELECT 
		m.measurement_concept_id AS subject_id,
		m.unit_concept_id,
		CAST(m.value_as_number AS NUMERIC) AS count_value
	FROM 
		omop.cmd.measurement m
	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
	WHERE 
		m.unit_concept_id IS NOT NULL
	AND 
		m.value_as_number IS NOT NULL
	) m
GROUP BY 
	m.subject_id, 
	m.unit_concept_id, 
	m.count_value
;
ANALYZE statsView_1815
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_1815
AS
SELECT
1815 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView_1815 s
  join statsView_1815 p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
) p
join 
(
	SELECT 
		m.subject_id AS stratum1_id,
		m.unit_concept_id AS stratum2_id,
		CAST(AVG(1.0 * m.count_value) AS NUMERIC) AS avg_value,
		CAST(STDDEV(m.count_value) AS NUMERIC) AS stdev_value,
		MIN(m.count_value) AS min_value,
		MAX(m.count_value) AS max_value,
		COUNT(*) AS total
	FROM 
	(
		SELECT 
			m.measurement_concept_id AS subject_id,
			m.unit_concept_id,
			CAST(m.value_as_number AS NUMERIC) AS count_value
		FROM 
			omop.cmd.measurement m
		JOIN 
			omop.cmd.observation_period op 
		ON 
			m.person_id = op.person_id
		AND 
			m.measurement_date >= op.observation_period_start_date
		AND 
			m.measurement_date <= op.observation_period_end_date				
		WHERE 
			m.unit_concept_id IS NOT NULL
		AND 
			m.value_as_number IS NOT NULL
	) m
	GROUP BY 
		m.subject_id, 
		m.unit_concept_id
) o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1815
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1815
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1815
;
ANALYZE s_tmpach_dist_1815
;
truncate table statsView_1815;
drop table statsView_1815;
truncate table tempResults_1815;
drop table tempResults_1815;


-- 1816	Distribution of low range, by measurement_concept_id and unit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE overallStats_1816
AS
SELECT
m.subject_id AS stratum1_id,
	m.unit_concept_id AS stratum2_id,
	CAST(AVG(1.0 * m.count_value) AS NUMERIC) AS avg_value,
	CAST(STDDEV(m.count_value) AS NUMERIC) AS stdev_value,
	MIN(m.count_value) AS min_value,
	MAX(m.count_value) AS max_value,
	COUNT(*) AS total
FROM
(
	SELECT 
		m.measurement_concept_id AS subject_id,
		m.unit_concept_id,
		CAST(m.range_low AS NUMERIC) AS count_value
	FROM 
		omop.cmd.measurement m
	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
	WHERE 
		m.unit_concept_id IS NOT NULL
	AND 
		m.value_as_number IS NOT NULL
	AND 
		m.range_low IS NOT NULL
	AND 
		m.range_high IS NOT NULL
	) m
GROUP BY 
	m.subject_id, 
	m.unit_concept_id
;
ANALYZE overallStats_1816
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE statsView_1816
AS
SELECT
m.subject_id AS stratum1_id,
	m.unit_concept_id AS stratum2_id,
	m.count_value,
	COUNT(*) AS total,
	ROW_NUMBER() OVER (PARTITION BY m.subject_id,m.unit_concept_id ORDER BY m.count_value) AS rn
FROM
(
	SELECT 
		m.measurement_concept_id AS subject_id,
		m.unit_concept_id,
		CAST(m.range_low AS NUMERIC) AS count_value
	FROM 
		omop.cmd.measurement m
  	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
	WHERE 
		m.unit_concept_id IS NOT NULL
	AND 
		m.value_as_number IS NOT NULL
	AND 
		m.range_low IS NOT NULL
	AND 
		m.range_high IS NOT NULL
	) m
GROUP BY 
	m.subject_id, 
	m.unit_concept_id, 
	m.count_value
;
ANALYZE statsView_1816
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_1816
AS
SELECT
1816 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView_1816 s
  join statsView_1816 p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
) p
join overallStats_1816 o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1816
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1816
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
  cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
  count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1816
;
ANALYZE s_tmpach_dist_1816
;
truncate table overallStats_1816;
drop table overallStats_1816;
truncate table statsView_1816;
drop table statsView_1816;
truncate table tempResults_1816;
drop table tempResults_1816;


-- 1817	Distribution of high range, by observation_concept_id and unit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE overallStats_1817
AS
SELECT
m.subject_id AS stratum1_id,
	m.unit_concept_id AS stratum2_id,
	CAST(AVG(1.0 * m.count_value) AS NUMERIC) AS avg_value,
	CAST(STDDEV(m.count_value) AS NUMERIC) AS stdev_value,
	MIN(m.count_value) AS min_value,
	MAX(m.count_value) AS max_value,
	COUNT(*) AS total
FROM
(
	SELECT 
		measurement_concept_id AS subject_id,
		unit_concept_id,
		CAST(range_high AS NUMERIC) AS count_value
	FROM 
		omop.cmd.measurement m
	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
	WHERE 
		m.unit_concept_id IS NOT NULL
	AND 
		m.value_as_number IS NOT NULL
	AND 
		m.range_low IS NOT NULL
	AND 
		m.range_high IS NOT NULL
	) m
GROUP BY 
	m.subject_id, 
	m.unit_concept_id
;
ANALYZE overallStats_1817
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE statsView_1817
AS
SELECT
m.subject_id AS stratum1_id,
	m.unit_concept_id AS stratum2_id,
	m.count_value,
	COUNT(*) AS total,
	ROW_NUMBER() OVER (PARTITION BY m.subject_id,m.unit_concept_id ORDER BY m.count_value) AS rn
FROM
(
	SELECT 
		m.measurement_concept_id AS subject_id,
		m.unit_concept_id,
		CAST(m.range_high AS NUMERIC) AS count_value
	FROM 
		omop.cmd.measurement m
	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
	WHERE 
		m.unit_concept_id IS NOT NULL
	AND 
		m.value_as_number IS NOT NULL
	AND 
		m.range_low IS NOT NULL
	AND 
		m.range_high IS NOT NULL
	) m
GROUP BY 
	m.subject_id, 
	m.unit_concept_id, 
	m.count_value
;
ANALYZE statsView_1817
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_1817
AS
SELECT
1817 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView_1817 s
  join statsView_1817 p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
) p
join overallStats_1817 o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_1817
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_1817
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
  cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
  count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_1817
;
ANALYZE s_tmpach_dist_1817
;
truncate table overallStats_1817;
drop table overallStats_1817;
truncate table statsView_1817;
drop table statsView_1817;
truncate table tempResults_1817;
drop table tempResults_1817;


-- 1818	Number of observation records below/within/above normal range, by observation_concept_id and unit_concept_id
--HINT DISTRIBUTE_ON_KEY(person_id)
CREATE TEMP TABLE rawData_1818
AS
SELECT
m.person_id,
	m.measurement_concept_id,
	m.unit_concept_id,
	CAST(CASE 
			WHEN m.value_as_number < m.range_low
				THEN 'Below Range Low'
			WHEN m.value_as_number >= m.range_low AND m.value_as_number <= m.range_high
				THEN 'Within Range'
			WHEN m.value_as_number > m.range_high
				THEN 'Above Range High'
			ELSE 'Other'
			END AS VARCHAR(255)) AS stratum_3
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
WHERE 
	m.value_as_number IS NOT NULL
AND 
	m.unit_concept_id IS NOT NULL
AND 
	m.range_low IS NOT NULL
AND 
	m.range_high IS NOT NULL;
ANALYZE rawData_1818
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1818
AS
SELECT
1818 AS analysis_id,
	CAST(measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(unit_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(person_id) AS count_value
FROM
rawData_1818
GROUP BY 
	measurement_concept_id,
	unit_concept_id,
	stratum_3;
ANALYZE s_tmpach_1818
;
TRUNCATE TABLE rawData_1818;
DROP TABLE rawData_1818;


-- 1819	Number of measurement records, by concept_id, with a value (with a mapped, non-null value_as_number)
CREATE TEMP TABLE s_tmpach_1819
AS
SELECT
1819 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(m.person_id) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
WHERE 
	m.value_as_number IS NOT NULL
OR 
	m.value_as_concept_id != 0
GROUP BY 
	m.measurement_concept_id;
ANALYZE s_tmpach_1819
;


-- 1820	Number of measurement records  by measurement start month
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1820
AS
WITH rawData  AS (
SELECT 
	EXTRACT(YEAR FROM m.measurement_date) * 100 + EXTRACT(MONTH FROM m.measurement_date) AS stratum_1,
	COUNT(m.person_id) AS count_value
FROM 
	omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	EXTRACT(YEAR FROM m.measurement_date) * 100 + EXTRACT(MONTH FROM m.measurement_date)
)
 SELECT
1820 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_1820
;


-- 1821	Number of measurement records with no numeric value
CREATE TEMP TABLE s_tmpach_1821
AS
SELECT
1821 as analysis_id,  
	cast(null as varchar(255)) as stratum_1, cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(m.PERSON_ID) as count_value
FROM
omop.cmd.measurement m
where m.value_as_number is null
;
ANALYZE s_tmpach_1821
;


-- 1822	Number of measurement records, by measurement_concept_id and value_as_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1822
AS
SELECT
1822 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(m.value_as_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.measurement_concept_id,
	m.value_as_concept_id;
ANALYZE s_tmpach_1822
;


-- 1823	Number of measurement records, by measurement_concept_id and operator_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1823
AS
SELECT
1823 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(m.operator_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.measurement_concept_id,
	m.operator_concept_id;
ANALYZE s_tmpach_1823
;


-- 1825	Number of measurement records, by measurement_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1825
AS
SELECT
1825 AS analysis_id,
	CAST(m.measurement_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.measurement_source_concept_id;
ANALYZE s_tmpach_1825
;


-- 1826	Number of measurement records, by value_as_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1826
AS
SELECT
1826 AS analysis_id,
	CAST(m.value_as_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.value_as_concept_id;
ANALYZE s_tmpach_1826
;


-- 1827	Number of measurement records, by unit_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1827
AS
SELECT
1827 AS analysis_id,
	CAST(m.unit_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date		
GROUP BY 
	m.unit_concept_id;
ANALYZE s_tmpach_1827
;


-- 1830	Number of measurement records inside a valid observation period
CREATE TEMP TABLE s_tmpach_1830
AS
SELECT
1830 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date
;
ANALYZE s_tmpach_1830
;


-- 1831	Proportion of people with at least one measurement record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in measurement (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_1831
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT m.person_id) AS person_count
FROM 
	omop.cmd.measurement m
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), m_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.measurement
)
 SELECT
1831 AS analysis_id,
	CASE WHEN mt.person_count != 0 THEN 
		CAST(CAST(1.0*op.person_count/mt.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(mt.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	m_total mt
;
ANALYZE s_tmpach_1831
;


-- 1832	Proportion of measurement records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of measurement records outside a valid observation period (numerator)
-- stratum_3:   Number of measurement records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_1832
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.measurement m
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), m_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.measurement
)
 SELECT
1832 AS analysis_id,
	CASE WHEN mt.record_count != 0 THEN 
		CAST(CAST(1.0*op.record_count/mt.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(mt.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	m_total mt
;
ANALYZE s_tmpach_1832
;


-- 1833	Proportion of measurement records with a valid observation period, but no value; stratified by measurement_concept_id
--
-- stratum_1:   measurement_concept_id
-- stratum_2:   Number of measurement records with no value for the given measurement_concept_id
-- stratum_3:   Proportion == stratum_2/count_value
-- count_value: Number of measurement records for the given measurement_concept_id
--
CREATE TEMP TABLE s_tmpach_1833
AS
SELECT
1833 AS analysis_id,
	m.measurement_concept_id AS stratum_1,
	CAST(SUM(CASE WHEN m.value_as_number IS NULL 
	          AND COALESCE(m.value_as_concept_id,0) = 0 
	    THEN 1 ELSE 0 END) AS VARCHAR(255))  AS stratum_2,
	CAST(CAST(1.0*SUM(CASE WHEN m.value_as_number IS NULL AND COALESCE(m.value_as_concept_id,0) = 0 
	                  THEN 1 ELSE 0 END)/(CASE WHEN COUNT(*)=0 THEN 1 ELSE COUNT(*) END) AS NUMERIC) AS VARCHAR(255)) AS stratum_3, 
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.measurement m
JOIN 
	omop.cmd.observation_period op 
ON 
	m.person_id = op.person_id
AND 
	m.measurement_date >= op.observation_period_start_date
AND 
	m.measurement_date <= op.observation_period_end_date
GROUP BY 
	m.measurement_concept_id
;
ANALYZE s_tmpach_1833
;


-- 1891	Number of total persons that have at least x measurements
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1891
AS
SELECT
1891 AS analysis_id,
	CAST(m.measurement_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(m.meas_cnt AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SUM(COUNT(m.person_id)) OVER (PARTITION BY m.measurement_concept_id ORDER BY m.meas_cnt DESC) AS count_value
FROM
(
	SELECT 
		m.measurement_concept_id,
		COUNT(m.measurement_id) AS meas_cnt,
		m.person_id
	FROM 
		omop.cmd.measurement m
	JOIN 
		omop.cmd.observation_period op 
	ON 
		m.person_id = op.person_id
	AND 
		m.measurement_date >= op.observation_period_start_date
	AND 
		m.measurement_date <= op.observation_period_end_date		
	GROUP BY 
		m.person_id,
		m.measurement_concept_id
	) m
GROUP BY 
	m.measurement_concept_id,
	m.meas_cnt;
ANALYZE s_tmpach_1891
;


-- 1900	completeness report
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_1900
AS
SELECT
1900 as analysis_id, 
  cast(table_name as varchar(255)) as stratum_1, 
  cast(column_name as varchar(255)) as stratum_2, 
  source_value as stratum_3, 
  cast(null as varchar(255)) as stratum_4, 
  cast(null as varchar(255)) as stratum_5,
cnt as count_value
FROM
(
  select 'measurement' as table_name, 'measurement_source_value' as column_name, measurement_source_value as source_value, COUNT(*) as cnt from omop.cmd.measurement where measurement_concept_id = 0 group by measurement_source_value 
  union
  select 'measurement' as table_name, 'unit_source_value' as column_name, unit_source_value as source_value, COUNT(*) as cnt from omop.cmd.measurement where unit_concept_id = 0 group by unit_source_value 
  union
  select 'procedure_occurrence' as table_name,'procedure_source_value' as column_name, procedure_source_value as source_value, COUNT(*) as cnt from omop.cmd.procedure_occurrence where procedure_concept_id = 0 group by procedure_source_value 
  union
  select 'procedure_occurrence' as table_name,'modifier_source_value' as column_name, modifier_source_value as source_value, COUNT(*) as cnt from omop.cmd.procedure_occurrence where modifier_concept_id = 0 group by modifier_source_value 
  union
  select 'drug_exposure' as table_name, 'drug_source_value' as column_name, drug_source_value as source_value, COUNT(*) as cnt from omop.cmd.drug_exposure where drug_concept_id = 0 group by drug_source_value 
  union
  select 'drug_exposure' as table_name, 'route_source_value' as column_name, route_source_value as source_value, COUNT(*) as cnt from omop.cmd.drug_exposure where route_concept_id = 0 group by route_source_value 
  union
  select 'condition_occurrence' as table_name, 'condition_source_value' as column_name, condition_source_value as source_value, COUNT(*) as cnt from omop.cmd.condition_occurrence where condition_concept_id = 0 group by condition_source_value 
  union
  select 'condition_occurrence' as table_name, 'condition_status_source_value' as column_name, condition_status_source_value as source_value, COUNT(*) as cnt from omop.cmd.condition_occurrence where condition_status_concept_id = 0 group by condition_status_source_value 
  union
  select 'observation' as table_name, 'observation_source_value' as column_name, observation_source_value as source_value, COUNT(*) as cnt from omop.cmd.observation where observation_concept_id = 0 group by observation_source_value                  
  union
  select 'observation' as table_name, 'unit_source_value' as column_name, unit_source_value as source_value, COUNT(*) as cnt from omop.cmd.observation where unit_concept_id = 0 group by unit_source_value                  
  union
  select 'observation' as table_name, 'qualifier_source_value' as column_name, qualifier_source_value as source_value, COUNT(*) as cnt from omop.cmd.observation where qualifier_concept_id = 0 group by qualifier_source_value
  union
  select 'payer_plan_period' as table_name, 'payer_source_value' as column_name, payer_source_value as source_value, COUNT(*) as cnt from omop.cmd.payer_plan_period where payer_concept_id = 0 group by payer_source_value                    
  union
  select 'payer_plan_period' as table_name, 'plan_source_value' as column_name, plan_source_value as source_value, COUNT(*) as cnt from omop.cmd.payer_plan_period where plan_concept_id = 0 group by plan_source_value                    
  union
  select 'payer_plan_period' as table_name, 'sponsor_source_value' as column_name, sponsor_source_value as source_value, COUNT(*) as cnt from omop.cmd.payer_plan_period where sponsor_concept_id = 0 group by sponsor_source_value                    
  union
  select 'payer_plan_period' as table_name, 'stop_reason_source_value' as column_name, stop_reason_source_value as source_value, COUNT(*) as cnt from omop.cmd.payer_plan_period where stop_reason_concept_id = 0 group by stop_reason_source_value                    
  union
  select 'provider' as table_name, 'specialty_source_value' as column_name, specialty_source_value as source_value, COUNT(*) as cnt from omop.cmd.provider where specialty_concept_id = 0 group by specialty_source_value
  union  
  select 'provider' as table_name, 'gender_source_value' as column_name, gender_source_value as source_value, COUNT(*) as cnt from omop.cmd.provider where gender_concept_id = 0 group by gender_source_value
  union  
  select 'person' as table_name, 'gender_source_value' as column_name, gender_source_value as source_value, COUNT(*) as cnt from omop.cmd.person where gender_concept_id = 0 group by gender_source_value                    
  union
  select 'person' as table_name, 'race_source_value' as column_name, race_source_value as source_value, COUNT(*) as cnt from omop.cmd.person where race_concept_id = 0 group by race_source_value                    
  union
  select 'person' as table_name, 'ethnicity_source_value' as column_name, ethnicity_source_value as source_value, COUNT(*) as cnt from omop.cmd.person where ethnicity_concept_id = 0 group by ethnicity_source_value                    
  union
  select 'specimen' as table_name, 'specimen_source_value' as column_name, specimen_source_value as source_value, COUNT(*) as cnt from omop.cmd.specimen where specimen_concept_id = 0 group by specimen_source_value                    
  union
  select 'specimen' as table_name, 'unit_source_value' as column_name, unit_source_value as source_value, COUNT(*) as cnt from omop.cmd.specimen where unit_concept_id = 0 group by unit_source_value                    
  union
  select 'specimen' as table_name, 'anatomic_site_source_value' as column_name, anatomic_site_source_value as source_value, COUNT(*) as cnt from omop.cmd.specimen where anatomic_site_concept_id = 0 group by anatomic_site_source_value                    
  union
  select 'specimen' as table_name, 'disease_status_source_value' as column_name, disease_status_source_value as source_value, COUNT(*) as cnt from omop.cmd.specimen where disease_status_concept_id = 0 group by disease_status_source_value                    
  union
  select 'visit_occurrence' as table_name, 'visit_source_value' as column_name, visit_source_value as source_value, COUNT(*) as cnt from omop.cmd.visit_occurrence where visit_concept_id = 0 group by visit_source_value
  union
	select 'visit_occurrence' as table_name, 'admitting_source_value' as column_name, admitting_source_value as source_value, COUNT(*) as cnt from omop.cmd.visit_occurrence where admitting_source_concept_id = 0 group by admitting_source_value
    union
    select 'visit_occurrence' as table_name, 'discharge_to_source_value' as column_name, discharge_to_source_value as source_value, COUNT(*) as cnt from omop.cmd.visit_occurrence where discharge_to_concept_id = 0 group by discharge_to_source_value
  union
  select 'device_exposure' as table_name, 'device_source_value' as column_name, device_source_value as source_value, COUNT(*) as cnt from omop.cmd.device_exposure where device_concept_id = 0 group by device_source_value
  union
  select 'death' as table_name, 'cause_source_value' as column_name, cause_source_value as source_value, COUNT(*) as cnt from omop.cmd.death where cause_concept_id = 0 group by cause_source_value
) a
where cnt >= 1 
;
ANALYZE s_tmpach_1900
 ;


-- 2000	patients with at least 1 Dx and 1 Rx
CREATE TEMP TABLE s_tmpach_2000
AS
SELECT
2000 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	CAST(d.cnt AS BIGINT) AS count_value
FROM
(
SELECT COUNT(*) cnt
FROM (
  SELECT DISTINCT person_id
  FROM (
    SELECT
      co.person_id
    FROM
      omop.cmd.condition_occurrence co
    JOIN
      omop.cmd.observation_period op
    ON
      co.person_id = op.person_id
    AND
      co.condition_start_date >= op.observation_period_start_date
    AND
      co.condition_start_date <= op.observation_period_end_date
    ) a
	INTERSECT
	SELECT DISTINCT person_id
  FROM (
    SELECT
      de.person_id
    FROM
      omop.cmd.drug_exposure de
    JOIN
      omop.cmd.observation_period op
    ON
      de.person_id = op.person_id
    AND
      de.drug_exposure_start_date >= op.observation_period_start_date
    AND
      de.drug_exposure_start_date <= op.observation_period_end_date
    ) b
	) c
) d;
ANALYZE s_tmpach_2000
;


-- 2001	patients with at least 1 Dx and 1 Proc
CREATE TEMP TABLE s_tmpach_2001
AS
SELECT
2001 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	CAST(d.cnt AS BIGINT) AS count_value
FROM
(
SELECT COUNT(*) cnt
FROM (
  SELECT DISTINCT person_id
	FROM (
    SELECT
      co.person_id
    FROM
      omop.cmd.condition_occurrence co
    JOIN
      omop.cmd.observation_period op
    ON
      co.person_id = op.person_id
    AND
      co.condition_start_date >= op.observation_period_start_date
    AND
      co.condition_start_date <= op.observation_period_end_date
  ) a
	INTERSECT
	SELECT DISTINCT person_id
	FROM (
    SELECT
      po.person_id
    FROM
      omop.cmd.procedure_occurrence po
    JOIN
      omop.cmd.observation_period op
    ON
      po.person_id = op.person_id
    AND
      po.procedure_date >= op.observation_period_start_date
    AND
      po.procedure_date <= op.observation_period_end_date
    ) b
	) c
) d;
ANALYZE s_tmpach_2001
;


-- 2002	patients with at least 1 Mes and 1 Dx and 1 Rx
CREATE TEMP TABLE s_tmpach_2002
AS
SELECT
2002 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	CAST(e.cnt AS BIGINT) AS count_value
FROM
(
SELECT COUNT(*) cnt
FROM (
	SELECT DISTINCT person_id
	FROM (
    SELECT
      m.person_id
    FROM
      omop.cmd.measurement m
    JOIN
      omop.cmd.observation_period op
    ON
      m.person_id = op.person_id
    AND
      m.measurement_date >= op.observation_period_start_date
    AND
      m.measurement_date <= op.observation_period_end_date
    ) a
	INTERSECT
	SELECT DISTINCT person_id
	FROM (
    SELECT
      co.person_id
    FROM
      omop.cmd.condition_occurrence co
    JOIN
      omop.cmd.observation_period op
    ON
      co.person_id = op.person_id
    AND
      co.condition_start_date >= op.observation_period_start_date
    AND
      co.condition_start_date <= op.observation_period_end_date
    ) b
	INTERSECT
	SELECT DISTINCT person_id
	FROM (
    SELECT
      de.person_id
    FROM
      omop.cmd.drug_exposure de
    JOIN
      omop.cmd.observation_period op
    ON
      de.person_id = op.person_id
    AND
      de.drug_exposure_start_date >= op.observation_period_start_date
    AND
      de.drug_exposure_start_date <= op.observation_period_end_date
    ) c
	) d
) e;
ANALYZE s_tmpach_2002
;


-- 2003	Patients with at least one visit
-- this analysis is in fact redundant, since it is possible to get it via
-- dist analysis 203 and query select count_value from achilles_results_dist where analysis_id = 203;
CREATE TEMP TABLE s_tmpach_2003
AS
SELECT
2003 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT vo.person_id) AS count_value
FROM
omop.cmd.visit_occurrence vo
JOIN 
	omop.cmd.observation_period op 
ON 
	vo.person_id = op.person_id
AND 
	vo.visit_start_date >= op.observation_period_start_date
AND 
	vo.visit_start_date <= op.observation_period_end_date;
ANALYZE s_tmpach_2003
;


-- Analysis 2004: Number of distinct patients that overlap between specific domains
-- Bit String Breakdown:   1) Condition Occurrence 2) Drug Exposure 3) Device Exposure 4) Measurement 5) Death 6) Procedure Occurrence 7) Observation
CREATE TEMP TABLE conoc 
AS
SELECT
distinct person_id 
FROM
omop.cmd.condition_occurrence;
ANALYZE conoc ;
CREATE TEMP TABLE drexp 
AS
SELECT
distinct person_id 
FROM
omop.cmd.drug_exposure;
ANALYZE drexp ;
CREATE TEMP TABLE dvexp 
AS
SELECT
distinct person_id 
FROM
omop.cmd.device_exposure;
ANALYZE dvexp ;
CREATE TEMP TABLE msmt 
AS
SELECT
distinct person_id 
FROM
omop.cmd.measurement;
ANALYZE msmt ;
CREATE TEMP TABLE death 
AS
SELECT
distinct person_id 
FROM
omop.cmd.death;
ANALYZE death ;
CREATE TEMP TABLE prococ 
AS
SELECT
distinct person_id 
FROM
omop.cmd.procedure_occurrence;
ANALYZE prococ ;
CREATE TEMP TABLE obs 
AS
SELECT
distinct person_id 
FROM
omop.cmd.observation;
ANALYZE obs ;
CREATE TEMP TABLE s_tmpach_2004 
AS
WITH rawData  AS (
select 2004 as analysis_id,
       CAST('0000001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0000010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0000011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0000100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0000101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0000110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0000111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0001111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0010111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0011111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0100111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0101111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0110111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('0111111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1000111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1001111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1010111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1011111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1100111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1101111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1110111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111000' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111001' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111010' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111011' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111100' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111101' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111110' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb UNION ALL
select 2004 as analysis_id,
       CAST('1111111' AS VARCHAR(255)) as stratum_1,
       cast((1.0 * personIntersection.count_value / totalPersonsDb.totalPersons) as varchar(255)) as stratum_2,
       CAST(NULL AS VARCHAR(255)) as stratum_3,
       CAST(NULL AS VARCHAR(255)) as stratum_4,
       CAST(NULL AS VARCHAR(255)) as stratum_5,
       personIntersection.count_value
      from
      (select count(*) as count_value from(select person_id from conoc intersect select person_id from drexp intersect select person_id from dvexp intersect select person_id from msmt intersect select person_id from death intersect select person_id from prococ intersect select person_id from obs) as subquery) as personIntersection,
  (select count(distinct(person_id)) as totalPersons from omop.cmd.person) as totalPersonsDb)  SELECT
* 
FROM
rawData;
ANALYZE s_tmpach_2004 ;


-- 2100	Number of persons with at least one device exposure , by device_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2100
AS
SELECT
2100 AS analysis_id,
	CAST(de.device_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(DISTINCT de.person_id) AS count_value
FROM
omop.cmd.device_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
GROUP BY 
	de.device_concept_id;
ANALYZE s_tmpach_2100
;


-- 2101	Number of device exposure  records, by device_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2101
AS
SELECT
2101 AS analysis_id,
	CAST(de.device_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(de.person_id) AS count_value
FROM
omop.cmd.device_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
GROUP BY 
	de.device_concept_id;
ANALYZE s_tmpach_2101
;


-- 2102	Number of persons by device by  start month, by device_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2102
AS
WITH rawData  AS (
SELECT 
	de.device_concept_id AS stratum_1,
	EXTRACT(YEAR FROM de.device_exposure_start_date) * 100 + EXTRACT(MONTH FROM de.device_exposure_start_date) AS stratum_2,
	COUNT(DISTINCT de.person_id) AS count_value
FROM 
	omop.cmd.device_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
GROUP BY 
	de.device_concept_id,
	EXTRACT(YEAR FROM de.device_exposure_start_date) * 100 + EXTRACT(MONTH FROM de.device_exposure_start_date)
)
 SELECT
2102 AS analysis_id,
  CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
  CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
  CAST(NULL AS VARCHAR(255)) AS stratum_3,
  CAST(NULL AS VARCHAR(255)) AS stratum_4,
  CAST(NULL AS VARCHAR(255)) AS stratum_5,
  count_value
FROM
rawData;
ANALYZE s_tmpach_2102
;


-- 2104	Number of persons with at least one device occurrence, by device_concept_id by calendar year by gender by age decile
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2104
AS
WITH rawData  AS (
SELECT 
	de.device_concept_id AS stratum_1,
	EXTRACT(YEAR FROM de.device_exposure_start_date) AS stratum_2,
	p.gender_concept_id AS stratum_3,
	FLOOR((EXTRACT(YEAR FROM de.device_exposure_start_date) - p.year_of_birth) / 10) AS stratum_4,
	COUNT(DISTINCT p.person_id) AS count_value
FROM 
	omop.cmd.person p
JOIN 
	omop.cmd.device_exposure de 
ON 
	p.person_id = de.person_id
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
GROUP BY 
	de.device_concept_id,
	EXTRACT(YEAR FROM de.device_exposure_start_date),
	p.gender_concept_id,
	FLOOR((EXTRACT(YEAR FROM de.device_exposure_start_date) - p.year_of_birth) / 10)
)
 SELECT
2104 AS analysis_id,
	CAST(stratum_1 AS VARCHAR(255)) AS stratum_1,
	CAST(stratum_2 AS VARCHAR(255)) AS stratum_2,
	CAST(stratum_3 AS VARCHAR(255)) AS stratum_3,
	CAST(stratum_4 AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	count_value
FROM
rawData;
ANALYZE s_tmpach_2104
;


-- 2105	Number of exposure records by device_concept_id by device_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2105
AS
SELECT
2105 AS analysis_id,
	CAST(de.device_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(de.device_type_concept_id AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(de.person_id) AS count_value
FROM
omop.cmd.device_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
GROUP BY 
	de.device_concept_id,
	de.device_type_concept_id;
ANALYZE s_tmpach_2105
;


-- 2106	Distribution of age by device_concept_id
--HINT DISTRIBUTE_ON_KEY(subject_id)
CREATE TEMP TABLE rawData_2106
AS
SELECT
o.device_concept_id AS subject_id,
	p.gender_concept_id,
	o.device_exposure_start_year - p.year_of_birth AS count_value
FROM
omop.cmd.person p
JOIN (
	SELECT 
		d.person_id,
		d.device_concept_id,
		MIN(EXTRACT(YEAR FROM d.device_exposure_start_date)) AS device_exposure_start_year
	FROM 
		omop.cmd.device_exposure d
	JOIN 
		omop.cmd.observation_period op 
	ON 
		d.person_id = op.person_id
	AND 
		d.device_exposure_start_date >= op.observation_period_start_date
	AND 
		d.device_exposure_start_date <= op.observation_period_end_date		
	GROUP BY 
		d.person_id,
		d.device_concept_id
	) o
ON 
	p.person_id = o.person_id
;
ANALYZE rawData_2106
;
--HINT DISTRIBUTE_ON_KEY(stratum1_id)
CREATE TEMP TABLE tempResults_2106
AS
WITH overallStats (stratum1_id, stratum2_id, avg_value, stdev_value, min_value, max_value, total)  AS (
  select subject_id as stratum1_id,
    gender_concept_id as stratum2_id,
    CAST(avg(1.0 * count_value) AS NUMERIC) as avg_value,
    CAST(STDDEV(count_value) AS NUMERIC) as stdev_value,
    min(count_value) as min_value,
    max(count_value) as max_value,
    COUNT(*) as total
  FROM rawData_2106
	group by subject_id, gender_concept_id
),
statsView (stratum1_id, stratum2_id, count_value, total, rn) as
(
  select subject_id as stratum1_id, gender_concept_id as stratum2_id, count_value, COUNT(*) as total, row_number() over (partition by subject_id, gender_concept_id order by count_value) as rn
  FROM rawData_2106
  group by subject_id, gender_concept_id, count_value
),
priorStats (stratum1_id, stratum2_id, count_value, total, accumulated) as
(
  select s.stratum1_id, s.stratum2_id, s.count_value, s.total, sum(p.total) as accumulated
  from statsView s
  join statsView p on s.stratum1_id = p.stratum1_id and s.stratum2_id = p.stratum2_id and p.rn <= s.rn
  group by s.stratum1_id, s.stratum2_id, s.count_value, s.total, s.rn
)
 SELECT
2106 as analysis_id,
  CAST(o.stratum1_id AS VARCHAR(255)) AS stratum1_id,
  CAST(o.stratum2_id AS VARCHAR(255)) AS stratum2_id,
  o.total as count_value,
  o.min_value,
	o.max_value,
	o.avg_value,
	o.stdev_value,
	MIN(case when p.accumulated >= .50 * o.total then count_value else o.max_value end) as median_value,
	MIN(case when p.accumulated >= .10 * o.total then count_value else o.max_value end) as p10_value,
	MIN(case when p.accumulated >= .25 * o.total then count_value else o.max_value end) as p25_value,
	MIN(case when p.accumulated >= .75 * o.total then count_value else o.max_value end) as p75_value,
	MIN(case when p.accumulated >= .90 * o.total then count_value else o.max_value end) as p90_value
FROM
priorStats p
join overallStats o on p.stratum1_id = o.stratum1_id and p.stratum2_id = o.stratum2_id 
GROUP BY o.stratum1_id, o.stratum2_id, o.total, o.min_value, o.max_value, o.avg_value, o.stdev_value
;
ANALYZE tempResults_2106
;
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_dist_2106
AS
SELECT
analysis_id, stratum1_id as stratum_1, stratum2_id as stratum_2, 
cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
tempResults_2106
;
ANALYZE s_tmpach_dist_2106
;
truncate table rawData_2106;
drop table rawData_2106;
truncate table tempResults_2106;
drop table tempResults_2106;


-- 2110	Number of device exposure records outside a valid observation period
CREATE TEMP TABLE s_tmpach_2110
AS
SELECT
2110 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.device_exposure de
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
WHERE
	op.person_id IS NULL
;
ANALYZE s_tmpach_2110
;


-- 2125	Number of device_exposure records, by device_source_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2125
AS
SELECT
2125 AS analysis_id,
	CAST(de.device_source_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.device_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
GROUP BY 
	de.device_source_concept_id;
ANALYZE s_tmpach_2125
;


-- 2130	Number of device exposure records inside a valid observation period
CREATE TEMP TABLE s_tmpach_2130
AS
SELECT
2130 AS analysis_id,
	CAST(NULL AS VARCHAR(255)) AS stratum_1,
	CAST(NULL AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	COUNT(*) AS count_value
FROM
omop.cmd.device_exposure de
JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date		
;
ANALYZE s_tmpach_2130
;


-- 2131	Proportion of people with at least one device_exposure record outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of people with a record outside a valid observation period (numerator)
-- stratum_3:   Number of people in device_exposure (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_2131
AS
WITH op_outside  AS (
SELECT 
	COUNT(DISTINCT de.person_id) AS person_count
FROM 
	omop.cmd.device_exposure de
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), de_total AS (
SELECT
	COUNT(DISTINCT person_id) person_count
FROM
	omop.cmd.device_exposure
)
 SELECT
2131 AS analysis_id,
	CASE WHEN det.person_count != 0 THEN
		CAST(CAST(1.0*op.person_count/det.person_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE
		CAST(NULL AS VARCHAR(255)) 
	END AS stratum_1, 
	CAST(op.person_count AS VARCHAR(255)) AS stratum_2,
	CAST(det.person_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.person_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	de_total det
;
ANALYZE s_tmpach_2131
;


-- 2132	Proportion of device_exposure records outside a valid observation period
--
-- stratum_1:   Proportion
-- stratum_2:   Number of device_exposure records outside a valid observation period (numerator)
-- stratum_3:   Number of device_exposure records (denominator)
-- count_value: Flag (0 or 1) indicating whether any such records exist
--
CREATE TEMP TABLE s_tmpach_2132
AS
WITH op_outside  AS (
SELECT 
	COUNT(*) AS record_count
FROM 
	omop.cmd.device_exposure de
LEFT JOIN 
	omop.cmd.observation_period op 
ON 
	de.person_id = op.person_id
AND 
	de.device_exposure_start_date >= op.observation_period_start_date
AND 
	de.device_exposure_start_date <= op.observation_period_end_date
WHERE
	op.person_id IS NULL
), de_total AS (
SELECT
	COUNT(*) record_count
FROM
	omop.cmd.device_exposure
)
 SELECT
2132 AS analysis_id,
	CASE WHEN det.record_count != 0 THEN
		CAST(CAST(1.0*op.record_count/det.record_count AS NUMERIC) AS VARCHAR(255)) 
	ELSE 
		CAST(NULL AS VARCHAR(255))
	END AS stratum_1, 
	CAST(op.record_count AS VARCHAR(255)) AS stratum_2,
	CAST(det.record_count AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SIGN(op.record_count) AS count_value
FROM
op_outside op
CROSS JOIN 
	de_total det
;
ANALYZE s_tmpach_2132
;


-- 2191	Number of total persons that have at least x measurements
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2191
AS
SELECT
2191 AS analysis_id,
	CAST(d.device_concept_id AS VARCHAR(255)) AS stratum_1,
	CAST(d.device_count AS VARCHAR(255)) AS stratum_2,
	CAST(NULL AS VARCHAR(255)) AS stratum_3,
	CAST(NULL AS VARCHAR(255)) AS stratum_4,
	CAST(NULL AS VARCHAR(255)) AS stratum_5,
	SUM(COUNT(d.person_id)) OVER (PARTITION BY d.device_concept_id ORDER BY d.device_count DESC) AS count_value
FROM
(
	SELECT 
		d.device_concept_id,
		COUNT(d.device_exposure_id) AS device_count,
		d.person_id
	FROM 
		omop.cmd.device_exposure d
	JOIN 
		omop.cmd.observation_period op 
	ON 
		d.person_id = op.person_id
	AND 
		d.device_exposure_start_date >= op.observation_period_start_date
	AND 
		d.device_exposure_start_date <= op.observation_period_end_date		
	GROUP BY 
		d.person_id,
		d.device_concept_id
	) d
GROUP BY 
	d.device_concept_id,
	d.device_count;
ANALYZE s_tmpach_2191
;


-- 2200	Number of persons with at least one note , by note_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2200
AS
SELECT
2200 as analysis_id, 
	CAST(m.note_type_CONCEPT_ID AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(distinct m.PERSON_ID) as count_value
FROM
omop.cmd.note m
group by m.note_type_CONCEPT_ID
;
ANALYZE s_tmpach_2200
;


-- 2201	Number of note records, by note_type_concept_id
--HINT DISTRIBUTE_ON_KEY(stratum_1)
CREATE TEMP TABLE s_tmpach_2201
AS
SELECT
2201 as analysis_id, 
    CAST(m.note_type_CONCEPT_ID AS VARCHAR(255)) as stratum_1,
	cast(null as varchar(255)) as stratum_2, cast(null as varchar(255)) as stratum_3, cast(null as varchar(255)) as stratum_4, cast(null as varchar(255)) as stratum_5,
	COUNT(m.PERSON_ID) as count_value
FROM
omop.cmd.note m
group by m.note_type_CONCEPT_ID
;
ANALYZE s_tmpach_2201
;


  DROP TABLE IF EXISTS omop.results.achilles_results;
--HINT DISTRIBUTE_ON_KEY(analysis_id)
CREATE TABLE omop.results.achilles_results
 AS
SELECT
analysis_id, stratum_1, stratum_2, stratum_3, stratum_4, stratum_5, count_value
FROM
(
  select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_0 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_3 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_4 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_5 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_7 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_8 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_9 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_10 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_11 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_12 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_101 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_102 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_108 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_109 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_110 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_111 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_112 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_113 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_114 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_115 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_116 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_117 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_118 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_119 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_200 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_201 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_202 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_204 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_207 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_209 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_210 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_211 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_212 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_220 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_221 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_225 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_226 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_230 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_231 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_232 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_300 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_301 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_303 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_325 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_400 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_401 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_402 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_404 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_405 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_409 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_410 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_411 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_412 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_413 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_414 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_415 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_416 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_420 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_425 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_430 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_431 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_432 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_500 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_501 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_502 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_504 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_505 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_509 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_510 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_525 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_530 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_531 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_532 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_600 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_601 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_602 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_604 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_605 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_609 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_610 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_612 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_613 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_620 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_625 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_630 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_631 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_632 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_691 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_700 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_701 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_702 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_704 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_705 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_709 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_710 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_711 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_712 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_713 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_720 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_725 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_730 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_731 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_732 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_791 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_800 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_801 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_802 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_804 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_805 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_807 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_809 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_810 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_812 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_813 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_814 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_820 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_822 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_823 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_825 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_826 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_827 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_830 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_831 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_832 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_891 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_900 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_901 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_902 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_904 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_908 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_910 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_911 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_920 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_930 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_931 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_932 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1000 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1001 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1002 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1004 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1008 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1010 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1011 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1020 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1030 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1031 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1032 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1100 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1101 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1102 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1103 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1200 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1201 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1202 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1203 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1300 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1301 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1302 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1304 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1307 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1309 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1310 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1311 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1312 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1320 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1321 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1325 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1326 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1330 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1331 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1332 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1408 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1409 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1410 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1411 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1412 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1413 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1414 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1415 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1425 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1800 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1801 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1802 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1804 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1805 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1807 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1809 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1810 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1811 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1812 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1813 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1814 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1818 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1819 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1820 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1821 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1822 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1823 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1825 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1826 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1827 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1830 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1831 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1832 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1833 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1891 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_1900 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2000 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2001 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2002 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2003 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2004 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2100 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2101 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2102 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2104 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2105 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2110 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2125 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2130 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2131 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2132 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2191 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2200 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value from
                   s_tmpach_2201
) Q
  where count_value > 5
;


  DROP TABLE IF EXISTS omop.results.achilles_results_dist;
--HINT DISTRIBUTE_ON_KEY(analysis_id)
CREATE TABLE omop.results.achilles_results_dist
 AS
SELECT
analysis_id, stratum_1, stratum_2, stratum_3, stratum_4, stratum_5, count_value, min_value, max_value, avg_value, stdev_value, median_value, p10_value, p25_value, p75_value, p90_value
FROM
(
  select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_0 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_103 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_104 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_105 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_106 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_107 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_203 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_206 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_213 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_403 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_406 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_506 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_511 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_512 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_513 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_514 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_515 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_603 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_606 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_703 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_706 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_715 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_716 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_717 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_803 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_806 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_815 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_903 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_906 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_907 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1003 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1006 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1007 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1303 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1306 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1313 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1406 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1407 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1803 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1806 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1815 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1816 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_1817 
union all
 select cast(analysis_id as int) as analysis_id, cast(stratum_1 as varchar(255)) as stratum_1, cast(stratum_2 as varchar(255)) as stratum_2, cast(stratum_3 as varchar(255)) as stratum_3, cast(stratum_4 as varchar(255)) as stratum_4, cast(stratum_5 as varchar(255)) as stratum_5, cast(count_value as bigint) as count_value, cast(min_value as NUMERIC) as min_value, cast(max_value as NUMERIC) as max_value, cast(avg_value as NUMERIC) as avg_value, cast(stdev_value as NUMERIC) as stdev_value, cast(median_value as NUMERIC) as median_value, cast(p10_value as NUMERIC) as p10_value, cast(p25_value as NUMERIC) as p25_value, cast(p75_value as NUMERIC) as p75_value, cast(p90_value as NUMERIC) as p90_value from
                   s_tmpach_dist_2106
) Q
  where count_value > 5
;


drop index omop.results.idx_ar_aid;

drop index omop.results.idx_ar_s1;

drop index omop.results.idx_ar_s2;

drop index omop.results.idx_ar_aid_s1;

drop index omop.results.idx_ar_aid_s1234;

drop index omop.results.idx_ard_aid;

drop index omop.results.idx_ard_s1;

drop index omop.results.idx_ard_s2;

create index idx_ar_aid on omop.results.achilles_results (analysis_id);

create index idx_ar_s1 on omop.results.achilles_results (stratum_1);

create index idx_ar_s2 on omop.results.achilles_results (stratum_2);

create index idx_ar_aid_s1 on omop.results.achilles_results (analysis_id,stratum_1);

create index idx_ar_aid_s1234 on omop.results.achilles_results (analysis_id,stratum_1,stratum_2,stratum_3,stratum_4);

create index idx_ard_aid on omop.results.achilles_results_dist (analysis_id);

create index idx_ard_s1 on omop.results.achilles_results_dist (stratum_1);

create index idx_ard_s2 on omop.results.achilles_results_dist (stratum_2);