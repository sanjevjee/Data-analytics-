-- How many total trips are recorded in the dataset?
select count(`Passenger_Count`)
FROM old.taxi_trip_pricing;

-- Write a query to find the sum of trip distances grouped by the time of day.
select Time_of_Day , sum(Trip_Distance_km) as total_dis
FROM old.taxi_trip_pricing
group by Time_of_Day
order by total_dis;

-- Retrieve the top 5 trips with the highest trip price.
select * , Trip_Price as average
FROM old.taxi_trip_pricing
order by average desc
limit 5;
-- Find the average trip duration for trips on weekends.
select Day_of_Week , avg(Trip_Duration_Minutes) as average_du
FROM old.taxi_trip_pricing
where Day_of_Week= 'Weekday'

-- Count the number of trips with traffic conditions "Low".
select Count(`Passenger_Count`) as coun
FROM old.taxi_trip_pricing
where Traffic_Conditions = 'Low'

-- Write a query to filter trips where the trip price is greater than $50.
select * , Trip_Price FROM old.taxi_trip_pricing
where Trip_Price>50

-- Retrieve trips where the base fare is null.
SELECT * 
FROM old.taxi_trip_pricing
WHERE Base_Fare IS NULL;

-- Group trips by "Day_of_Week" and calculate the total trip price for each group.
select Day_of_Week , sum(Trip_Price) as total_price
FROM old.taxi_trip_pricing
group by Day_of_Week
order by total_price desc

-- Write a query to find the maximum "Per_Km_Rate" in the dataset.
select max(Per_Km_Rate) as max_rate
FROM old.taxi_trip_pricing

-- Retrieve all rows where the number of passengers is greater than 2.
select * FROM old.taxi_trip_pricing
where Passenger_Count>2

-- Write a query to find trips with a distance of more than 40 km.
select * FROM old.taxi_trip_pricing
where Trip_Distance_km > 40

-- Calculate the average trip price for trips with "High" traffic conditions.
select avg(Trip_Price) as t_p
FROM old.taxi_trip_pricing
where Traffic_Conditions = 'High'

-- Create a query to count the trips with missing weather data.
select count(*)
FROM old.taxi_trip_pricing
where Weather is null

-- Write a query to list trips with a trip duration less than 30 minutes.
select count(*)
FROM old.taxi_trip_pricing
where Trip_Duration_Minutes < 30

-- Find the total trip price for each traffic condition.
select Traffic_Conditions , sum(Trip_Price) as total
FROM old.taxi_trip_pricing
group by Traffic_Conditions
order by total;

-- Retrieve trips taken in "Morning" with more than 2 passengers.
select * FROM old.taxi_trip_pricing
where Time_of_Day = 'Morning' and  Passenger_Count > 2

-- Find the trip with the minimum trip distance.
SELECT min(`Trip_Distance_km`) as dis 
FROM old.taxi_trip_pricing

-- Write a query to calculate the total price using the formula Base_Fare + (Trip_Distance_km * Per_Km_Rate) + (Trip_Duration_Minutes * Per_Minute_Rate).
select * , Base_Fare + (Trip_Distance_km * Per_Km_Rate) + (Trip_Duration_Minutes * Per_Minute_Rate)
FROM old.taxi_trip_pricing

-- List all trips sorted by trip price in descending order.
SELECT * 
FROM old.taxi_trip_pricing
order by Trip_Price DESC;

-- Retrieve trips where the weather condition is "Clear" and the traffic condition is "Low".
SELECT * 
FROM old.taxi_trip_pricing
where Weather = 'Clear' and  Traffic_Conditions = 'Low'
