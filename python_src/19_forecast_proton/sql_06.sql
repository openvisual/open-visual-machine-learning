WITH b(x,y) AS 
(
    SELECT 1,2 
    UNION ALL 
    SELECT x+ 1, y + 1 
    FROM b 
    WHERE x < 20
) SELECT * FROM b;