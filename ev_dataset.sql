-- 1. How would you write a query to find the total EV sales for the year 2022?
SELECT sum(`EV_Sales_Quantity`) as 2022_sale
from ev_dataset
where Year=2022;

-- 2. Write a query to retrieve the top 3 states with the highest EV sales.
SELECT State, sum(`EV_Sales_Quantity`) as top_ev_sale
from ev_dataset
Group by State
order by top_ev_sale DESC
LIMIT 3;

-- 3. How do you find the average EV sales quantity grouped by each Vehicle Category?
SELECT Vehicle_Category, avg(`EV_Sales_Quantity`) as avg_sale
from ev_dataset
group by Vehicle_Category
order by avg_sale;

-- 4. Write a query to list all records where the Vehicle Type is "Bus."
SELECT *
from ev_dataset
Where Vehicle_Type="Bus"
-- 5. How do you get the total number of records from the state "Delhi"?
SELECT *
from ev_dataset
Where State = "Delhi";
-- 6. Write a query to filter data for EV sales greater than 1000 in the year 2021.
SELECT *
FROM ev_dataset
WHERE Year = 2021 AND EV_Sales_Quantity > 1000;

-- 7. How would you find the maximum EV sales quantity in the dataset?
SELECT max(`EV_Sales_Quantity`)
FROM ev_dataset;

-- 8. Write a query to display EV sales by month for the year 2020.
SELECT Month_Name,avg(EV_Sales_Quantity) as average
FROM ev_dataset
where Year=2020
group by Month_Name
order by average;



-- 9. How can you find the total EV sales quantity by Vehicle Class?
Select Vehicle_Class,sum(EV_Sales_Quantity) as total
FROM ev_dataset
group by Vehicle_Class
order by total;


-- 10. Write a query to join this dataset with another table containing state population data.

-- not avalable
-- 11. How would you use a subquery to find states with above-average EV sales?
SELECT State, avg(EV_Sales_Quantity) average
from ev_dataset
where EV_Sales_Quantity>(select avg(EV_Sales_Quantity) from ev_dataset)
group by State
order by average;

-- 12. Write a query to find the yearly growth of EV sales from 2019 to 2023.
SELECT Year, AVG(EV_Sales_Quantity) AS sale
FROM ev_dataset
WHERE Year BETWEEN 2019 AND 2023
GROUP BY Year
ORDER BY sale;

-- 13. How can you display all records where the Vehicle Category is "Others"?
SELECT * 
from ev_dataset
where Vehicle_Category="Others"

-- 14. Write a query to find the lowest EV sales quantity in each state.
SELECT State, MIN(EV_Sales_Quantity) AS lowest_sale
FROM ev_dataset
GROUP BY State;

-- 15. How do you write a query to calculate the percentage of sales for each Vehicle Type?
SELECT Vehicle_Type,
       SUM(EV_Sales_Quantity) AS total_sales,
       (SUM(EV_Sales_Quantity) * 100.0 / (SELECT SUM(EV_Sales_Quantity) FROM ev_dataset)) AS percentage_of_sales
FROM ev_dataset
GROUP BY Vehicle_Type
ORDER BY percentage_of_sales DESC;

-- 16. Write a query to sort the data by EV sales in descending order.
select EV_Sales_Quantity
from ev_dataset
order by EV_Sales_Quantity DESC;

-- 17. How would you count the number of unique Vehicle Classes in the dataset?
SELECT COUNT(Vehicle_Class) AS total_count
FROM ev_dataset;

-- 18. Write a query to get EV sales quantity data for the month of "June" across all years.
select Month_Name, avg(EV_Sales_Quantity) as average
from ev_dataset
where Month_Name="jun"
group by Month_Name
order by average;

-- 19. How do you update a record in the table to change the EV sales quantity?
UPDATE ev_dataset
SET EV_Sales_Quantity = 1200
WHERE State = 'Delhi' AND Year = 2022;


-- 20. Write a query to delete records with EV sales quantity equal to zero?
DELETE FROM ev_dataset
WHERE EV_Sales_Quantity = 0;





