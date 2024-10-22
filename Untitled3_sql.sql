-- 1. How would you write a query to find the average house price in the dataset?
SELECT avg(`House_Price`)
FROM realstate.`house_price_regression_dataset.csv1`;

-- 2. How do you select the top 5 most expensive houses using SQL?
SELECT House_Price
FROM realstate.`house_price_regression_dataset.csv1`
order by House_Price DESC
LIMIT 5;

-- 3. How can you filter the dataset to show only houses in a specific location?
-- LOCATION NOT AVAILABLE--

-- 4. How would you use SQL to find the total number of houses in the dataset?
SELECT COUNT(*) AS Total_BedAndBath
FROM realstate.`house_price_regression_dataset.csv1`

-- 5. Write a query to calculate the minimum and maximum house prices.

SELECT min(`House_Price`),max(`House_Price`)
FROM realstate.`house_price_regression_dataset.csv1`;

-- 6. How would you group houses by location and get the average price for each location?
-- location not available

-- 7. How would you write a query to find houses with more than 3 bedrooms?
SELECT Square_Footage, Num_Bedrooms as bedr from realstate.`house_price_regression_dataset.csv1`
where Num_Bedrooms>3;

-- 8. Explain how you can use JOINS to combine two tables (e.g., house prices and property taxes).
-- unfortunatly there no extra table

-- 9. Write a query to return houses where the price is above the average.
Select Square_Footage from realstate.`house_price_regression_dataset.csv1`
where House_Price>(select avg(House_Price) from realstate.`house_price_regression_dataset.csv1` )

-- 10. How do you use the COUNT function to find how many houses are listed in the dataset?

Select COUNT(*) from realstate.`house_price_regression_dataset.csv1`

-- 11. How would you update house prices by 5% in SQL?
UPDATE from realstate.`house_price_regression_dataset.csv1`
SET House_Price = House_Price * 1.05;

-- 12. How do you delete records of houses priced below a certain threshold?
DELETE FROM realstate.`house_price_regression_dataset.csv1`
WHERE House_Price < 68754;


-- 13. How do you find the total number of houses sold within a specific time range?
select `Year_Built`, avg(`House_Price`) as av
FROM realstate.`house_price_regression_dataset.csv1`
group by `Year_Built`
order by av ;

-- 14. Write a SQL query to list houses that have the word "Luxury" in their description.
-- no
-- 15. How can you find the second-highest house price in SQL?
-- done
-- 16. How do you perform a SQL query to find houses priced between two values (e.g., $100,000 and $300,000)?
select House_Price FROM realstate.`house_price_regression_dataset.csv1`
where House_Price between 1000 and 543332;
-- 17. How do you sort house prices in descending order using SQL?
-- done
-- 18. Explain how you would write a query to find houses with missing data.
select *  FROM realstate.`house_price_regression_dataset.csv1`
where  Square_Footage is null
or	   Num_Bedrooms is null
-- 19. How would you use GROUP BY and HAVING to filter groups based on a condition?
-- done
-- 20. How do you use SQL to find the total number of houses in each price range (e.g., 0-100K, 100K-200K)?
select Square_Footage, avg(House_Price) as h_p
FROM realstate.`house_price_regression_dataset.csv1`
group by Square_Footage
order by h_p

-- 21. How can you use a subquery to find houses that have prices above the average for their location?
-- no
-- 22. How would you write a query to calculate the price per square foot for each house?
-- done
-- 23. Explain how to use window functions (e.g., ROW_NUMBER()) to rank houses by price.
-- done

-- 24. How do you use a CASE statement to categorize houses as “Affordable,” “Mid-range,” and “Luxury”?

SELECT House_Price,
   CASE
       WHEN House_Price < 5000 THEN 'Affordable'
       WHEN House_Price < 214454 THEN 'Mid'
       WHEN House_Price < 2534838 THEN 'Luxury'
       ELSE 'Ultra Luxury'
   END AS Price_Category
FROM realstate.`house_price_regression_dataset.csv1`;

-- 25. How can you find duplicate records of houses in the dataset?
SELECT House_Price, Year_Built, COUNT(*) AS Duplicate_Count
FROM realstate.`house_price_regression_dataset.csv1`
GROUP BY House_Price, Year_Built
HAVING COUNT(*) > 1;

-- 26. How would you write a query to find houses with a price that is a multiple of 100,000?
SELECT *
FROM realstate.`house_price_regression_dataset.csv1`
WHERE House_Price % 100000 = 0;

-- 27. Explain how you can use a JOIN to combine house prices with a table of interest rates.
SELECT hp.*, ir.Interest_Rate
FROM realstate.`house_price_regression_dataset.csv1` AS hp
JOIN realstate.`interest_rates_table` AS ir
ON hp.City = ir.City;

-- 28. How would you calculate the average price of houses with more than 2 bathrooms?
select avg(House_Price) as price  FROM realstate.`house_price_regression_dataset.csv1`
where Num_Bathrooms>2;
-- 29. How do you write a query to get the sum of house prices for a specific city?
select * FROM realstate.`house_price_regression_dataset.csv1`
where city="rome"
-- 30. How would you create a view in SQL to show only the top 10 most expensive houses?
select House_Price  FROM realstate.`house_price_regression_dataset.csv1`
order by House_Price desc;