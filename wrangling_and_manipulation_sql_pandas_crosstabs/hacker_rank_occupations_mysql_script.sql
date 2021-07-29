set @doctors_row_num = 0;
set @professors_row_num = 0;
set @singers_row_num = 0;
set @actors_row_num = 0;


select max(doctor), max(professor), max(singer), max(actor) from (
select union_table.row_num as row_num, 
case when occupation ='Doctor' then union_table.name
else NULL 
end as doctor,
case when occupation ='Professor' then union_table.name
else NULL 
end as professor,
case when occupation ='Singer' then union_table.name
else NULL 
end as singer,
case when occupation ='Actor' then union_table.name
else NULL 
end as actor

from 
( 
    select doctors.row_num, doctors.name, doctors.occupation
from  (select name, occupation, (@doctors_row_num:=@doctors_row_num+1) as row_num 
      from occupations 
      where occupation = 'Doctor' 
      order by name asc) as doctors
union

select professors.row_num, professors.name, professors.occupation 
from (select name, occupation, (@professors_row_num:=@professors_row_num+1) as row_num 
      from occupations 
      where Occupation = 'Professor' 
      order by name asc) as professors
union

select singers.row_num, singers.name, singers.occupation 
from (select name, occupation, (@singers_row_num:=@singers_row_num+1) as row_num 
      from occupations 
      where Occupation = 'Singer' 
      order by name asc) as singers
union

select actors.row_num, actors.name, actors.occupation 
from (select name, occupation, (@actors_row_num:=@actors_row_num+1) as row_num 
      from occupations 
      where Occupation = 'Actor' 
      order by name asc) as actors
    
) as union_table ) as end_table 
group by row_num;