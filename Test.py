str = """with 
expressions as
     (select 1 id, '2 3 4 + * 5 *' expr from dual
       union all
       select 2 id, '17 10 + 3 * 9 /' expr from dual
       union all
       select 3 id, '12.4 4 / 10 * 2 + 11 / 4 / 0.25 +' expr from dual
      ),
with_num as( select ' ' ||expr||' ' str, rownum num from expressions),
res(expr, num) AS(select str,num
                  from with_num union all
select REGEXP_REPLACE(REPLACE(expr,
                              REGEXP_SUBSTR(expr,'((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'), 
                              DECODE(REGEXP_SUBSTR(expr,'[-/\+\*] '), 
                                     '+ ',' '||TO_CHAR(TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                           '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                             '(^| )(-)?((\d+)|(\d*\.\d+)) ')))+ 
                                                       TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                          '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                             ' (-)?((\d+)|(\d*\.\d+)) ',2))))||' ',

                                     '* ',' '||TO_CHAR(TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                         '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                            '(^| )(-)?((\d+)|(\d*\.\d+)) ')))*
                                                      TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                          '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                            ' (-)?((\d+)|(\d*\.\d+)) ',2))))||' ',

                                    '/ ',' '||TO_CHAR(TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                          '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                            '(^| )(-)?((\d+)|(\d*\.\d+)) ')))/ 
                                                      TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                          '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                            ' (-)?((\d+)|(\d*\.\d+)) ',2))),
                                                      '999999.9999999')||' ',

                                    '- ',' '||TO_CHAR(TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                          '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                            '(^| )(-)?((\d+)|(\d*\.\d+)) ')))- 
                                                      TO_NUMBER(TRIM(' ' FROM REGEXP_SUBSTR(REGEXP_SUBSTR(expr,
                                                                                                          '((^| )(-)?((\d+)|(\d*\.\d+))){2}( [-/\+\*])( |$)'),
                                                                                            ' (-)?((\d+)|(\d*\.\d+)) ',2))))||' '
                                  )),'( )+',' ') expr, num
FROM res
WHERE REGEXP_INSTR(expr,'[-/\+\*] ') > 0)
SELECT r.num ID, cc.str expr, to_char(to_number(TRIM(' ' FROM r.expr))) result
FROM res r JOIN with_num cc
ON r.num=cc.num
WHERE 
regexp_instr(r.expr, '[-/\+\*]') = 0
ORDER BY cc.num;
"""

print(str.lower())