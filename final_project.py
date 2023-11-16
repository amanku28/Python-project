import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    # Load the dataset
    url = "hour.csv\\hour.csv"
    df = pd.read_csv(url)

    # Call the all functions here
    
    hourly(df)
    monthly(df)
    seasonal_counts(df)
    workingday_counts(df)
    weather_counts(df)
    casual_registered(df)
    temp1(df)
    temp2(df)
    temp3(df)
    temp5(df)
    temp6(df)
    temp7(df)
    temp8(df)
    temp9(df)
    temp10(df)
    temp11(df)
    temp12(df)
    temp13(df)
    temp14(df)
    temp16(df)
    temp17(df)
    temp18(df)
    temp19(df)
    temp20(df)
    temp21(df)
    temp23(df)
    temp24(df)
    temp25(df)
    temp26(df)
    temp27(df)
    temp28(df)
    temp29(df)
    temp30(df)
    temp31(df)
    temp32(df)
    temp33(df)
    temp34(df)
    temp35(df)
    temp36(df)
    temp37(df)
    temp38(df)
    temp39(df)
    temp40(df)

# Definition of all functions 

# Hourly distribution of total bike users

def hourly(df):
    
    hourly_counts = df.groupby('hr')['cnt'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_counts, marker='o', linestyle='-', color='b')
    plt.title('Hourly Distribution of Total Bike Users')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Total Bike Users')
    plt.grid(True)
    plt.show(block=True)
    
 
# Monthly distribution of total bike users over multiple years
    
def monthly(df):
    
    # Extracting year and month from the 'dteday' column
    df['year_month'] = pd.to_datetime(df['dteday']).dt.to_period('M')

    monthly_counts = df.groupby('year_month')['cnt'].sum()

    plt.figure(figsize=(12, 6))
    plt.bar(monthly_counts.index.astype(str), monthly_counts, color='skyblue')
    plt.title('Total Bike Usage Over the Months')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Bike Users')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.show()
    

#Seasonal Distribution of Total Bike Users:
        
def seasonal_counts(df):
    seasonal_counts = df.groupby('season')['cnt'].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(seasonal_counts.index, seasonal_counts, color='skyblue')
    plt.title('Seasonal Distribution of Total Bike Users')
    plt.xlabel('Season')
    plt.ylabel('Total Bike Users')
    plt.xticks(seasonal_counts.index, ['Spring', 'Summer', 'Fall', 'Winter'])
    plt.show()
    
    seasonal_counts1 = df.groupby('season')['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(seasonal_counts, labels=['Spring', 'Summer', 'Fall', 'Winter'], autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen', 'lightgray'])
    plt.title('Seasonal Distribution of Total Bike Users')
    plt.show()


# Working Day vs. Non-Working Day Bike Usage:

def workingday_counts(df):
    workingday_counts = df.groupby('workingday')['cnt'].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(workingday_counts.index, workingday_counts, color='lightgreen')
    plt.title('Working Day vs. Non-Working Day Bike Usage')
    plt.xlabel('Working Day (0: Non-Working Day, 1: Working Day)')
    plt.ylabel('Total Bike Users')
    plt.xticks(workingday_counts.index, ['Non-Working Day', 'Working Day'])
    plt.show()
    
    workingday_counts1 = df.groupby('workingday')['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(workingday_counts, labels=['Non-Working Day', 'Working Day'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Working Day vs. Non-Working Day Bike Usage')
    plt.show()


# Weather Situation Distribution with bike users:

def  weather_counts(df):
    weather_counts = df.groupby('weathersit')['cnt'].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(weather_counts.index, weather_counts, color='lightcoral')
    plt.title('Weather Situation Distribution of Total Bike Users')
    plt.xlabel('Weather Situation')
    plt.ylabel('Total Bike Users')
    plt.xticks(weather_counts.index, ['Clear', 'Mist', 'Light Rain/Snow', 'Heavy Rain/Snow'])
    plt.show()
    
    weather_counts1 = df.groupby('weathersit')['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(weather_counts, labels=['Clear', 'Mist', 'Light Rain/Snow', 'Heavy Rain/Snow'], autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen', 'lightgray'])
    plt.title('Weather Situation Distribution of Total Bike Users')
    plt.show()

# Hourly Distribution of Casual vs. Registered Users:
    
def casual_registered(df):
    hourly_casual = df.groupby('hr')['casual'].sum()
    hourly_registered = df.groupby('hr')['registered'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_casual, marker='o', linestyle='-', color='orange', label='Casual Users')
    plt.plot(hourly_registered, marker='o', linestyle='-', color='blue', label='Registered Users')
    plt.title('Hourly Distribution of Casual vs. Registered Users')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)
    plt.show()   
    
# Temperature vs. Total Bike Users:

def temp1(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['temp'], df['cnt'], alpha=0.5, color='green')
    plt.title('Temperature vs. Total Bike Users')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Total Bike Users')
    plt.grid(True)
    plt.show()   
    
# Humidity vs. Total Bike Users:  

def temp2(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['hum'], df['cnt'], alpha=0.5, color='purple')
    plt.title('Humidity vs. Total Bike Users')
    plt.xlabel('Normalized Humidity')
    plt.ylabel('Total Bike Users')
    plt.grid(True)
    plt.show()  
    
# Wind Speed vs. Total Bike Users:

def temp3(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['windspeed'], df['cnt'], alpha=0.5, color='brown')
    plt.title('Wind Speed vs. Total Bike Users')
    plt.xlabel('Normalized Wind Speed')
    plt.ylabel('Total Bike Users')
    plt.grid(True)
    plt.show()
    
    
# Holiday vs. Non-Holiday Bike Usage:

def temp5(df):
    holiday_counts = df.groupby('holiday')['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(holiday_counts, labels=['Non-Holiday', 'Holiday'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Holiday vs. Non-Holiday Bike Usage')
    plt.show()
    
# Daily Trends in Bike Usage (Casual vs. Registered): 

def temp6(df):
    daily_casual = df.groupby('dteday')['casual'].sum()
    daily_registered = df.groupby('dteday')['registered'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_casual, marker='o', linestyle='-', color='orange', label='Casual Users')
    plt.plot(daily_registered, marker='o', linestyle='-', color='blue', label='Registered Users')
    plt.title('Daily Trends in Bike Usage (Casual vs. Registered)')
    plt.xlabel('Date')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Box Plot of Total Bike Users by Season:

def temp7(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='season', y='cnt', data=df, palette='pastel')
    plt.title('Box Plot of Total Bike Users by Season')
    plt.xlabel('Season')
    plt.ylabel('Total Bike Users')
    plt.show()

# Count of bikes during weekdays and weekends


def temp8(df):
    # Assuming df is your DataFrame
    # Convert 'dteday' column to datetime format
    df['dteday'] = pd.to_datetime(df['dteday'])

    # Create a new column 'day_type' to indicate weekend or weekday
    df['day_type'] = df['dteday'].dt.day_name()
    df['day_type'] = df['day_type'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')

    # Count of bikes during weekdays and weekends
    day_type_counts = df.groupby('day_type')['cnt'].sum()

    # Plotting the counts
    plt.figure(figsize=(8, 6))
    plt.bar(day_type_counts.index, day_type_counts, color=['lightcoral', 'lightblue'])
    plt.title('Bike Counts on Weekdays vs. Weekends')
    plt.xlabel('Day Type')
    plt.ylabel('Total Bike Users')
    plt.show()


# Usage Patterns Over the Years:

def temp9(df):
    yearly_counts = df.groupby('yr')['cnt'].sum()

    plt.figure(figsize=(8, 6))
    plt.bar(yearly_counts.index, yearly_counts, color=['lightblue', 'lightgreen'])
    plt.title('Total Bike Usage Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Total Bike Users')
    plt.xticks(yearly_counts.index, ['2011', '2012'])
    plt.show()


# Impact of Temperature on Casual and Registered Users:

def temp10(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['temp'], df['casual'], alpha=0.5, color='orange', label='Casual Users')
    plt.scatter(df['temp'], df['registered'], alpha=0.5, color='blue', label='Registered Users')
    plt.title('Impact of Temperature on Casual and Registered Users')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)
    plt.show()


# Hourly Trends on Weekdays and Weekends:

def temp11(df):
    weekday_hourly = df[df['day_type'] == 'Weekday'].groupby('hr')['cnt'].sum()
    weekend_hourly = df[df['day_type'] == 'Weekend'].groupby('hr')['cnt'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(weekday_hourly, marker='o', linestyle='-', color='blue', label='Weekday')
    plt.plot(weekend_hourly, marker='o', linestyle='-', color='orange', label='Weekend')
    plt.title('Hourly Trends on Weekdays vs. Weekends')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Correlation Matrix:

def temp12(df):
    corr_matrix = df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()  


# Distribution of Bike Users on Holidays by Season:  

def temp13(df):
    holiday_season_counts = df.groupby(['holiday', 'season'])['cnt'].sum().unstack()

    plt.figure(figsize=(12, 6))
    holiday_season_counts.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Distribution of Bike Users on Holidays by Season')
    plt.xlabel('Season')
    plt.ylabel('Total Bike Users')
    plt.xticks([0, 1], ['Non-Holiday', 'Holiday'], rotation=0)
    plt.legend(title='Season', loc='upper right')
    plt.show()
    
# Box Plot of Bike Users by Weather Situation:   

def temp14(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weathersit', y='cnt', data=df, palette='pastel')
    plt.title('Box Plot of Bike Users by Weather Situation')
    plt.xlabel('Weather Situation')
    plt.ylabel('Total Bike Users')
    plt.show()
       
# Distribution of Bike Users by Month:  

def temp16(df):
    monthly_user_counts = df.groupby('mnth')['cnt'].sum()

    plt.figure(figsize=(12, 6))
    plt.bar(monthly_user_counts.index, monthly_user_counts, color='skyblue')
    plt.title('Distribution of Bike Users by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Bike Users')
    plt.show()
    
# Peak Hour Analysis for Casual and Registered Users: 

def temp17(df):
    peak_hour_casual = df.groupby('hr')['casual'].sum()
    peak_hour_registered = df.groupby('hr')['registered'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(peak_hour_casual, marker='o', linestyle='-', color='orange', label='Casual Users')
    plt.plot(peak_hour_registered, marker='o', linestyle='-', color='blue', label='Registered Users')
    plt.title('Peak Hour Analysis for Casual and Registered Users')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Impact of Humidity on Bike Usage During Different Seasons:

def temp18(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='hum', y='cnt', hue='season', data=df, palette='coolwarm')
    plt.title('Impact of Humidity on Bike Usage During Different Seasons')
    plt.xlabel('Normalized Humidity')
    plt.ylabel('Total Bike Users')
    plt.show()
   
# Daywise Distribution of Bike Users:

def temp19(df):
    daywise_counts = df.groupby('weekday')['cnt'].sum()

    plt.figure(figsize=(10, 6))
    plt.bar(daywise_counts.index, daywise_counts, color='lightgreen')
    plt.title('Daywise Distribution of Bike Users')
    plt.xlabel('Day of the Week')
    plt.ylabel('Total Bike Users')
    plt.xticks(daywise_counts.index, ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
    plt.show()
    
    daywise_counts1 = df.groupby('weekday')['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(daywise_counts, labels=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen', 'lightgray', 'lightpink', 'lightcyan', 'lightyellow'])
    plt.title('Daywise Distribution of Bike Users')
    plt.show()  
    
# Distribution of Bike Users by Season: pie chart

def temp20(df):
    season_counts = df.groupby('season')['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(season_counts, labels=['Spring', 'Summer', 'Fall', 'Winter'], autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen', 'lightgray'])
    plt.title('Distribution of Bike Users by Season')
    plt.show()   

# Distribution of Bike Users by Day of the Week: pie chart

def temp21(df):
    daywise_counts = df.groupby('weekday')['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(daywise_counts, labels=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen', 'lightgray', 'lightpink', 'lightyellow', 'lightcyan'])
    plt.title('Distribution of Bike Users by Day of the Week')
    plt.show()
    



# Comparison of Bike Users on Working Days vs. Non-Working Days:

def temp23(df):
    workingday_counts = df.groupby('workingday')['cnt'].sum()
    plt.figure(figsize=(8, 8))
    plt.pie(workingday_counts, labels=['Non-Working Day', 'Working Day'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Comparison of Bike Users on Working Days vs. Non-Working Days')
    plt.show()
    
# Average Bike Users per Hour:

def temp24(df):
    hourly_avg = df.groupby('hr')['cnt'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_avg, marker='o', linestyle='-', color='purple')
    plt.title('Average Bike Users per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Bike Users')
    plt.grid(True)
    plt.show()
    
# Pairwise Scatter Plot of Temperature, Humidity, and Bike Usage:

def temp25(df):
    sns.pairplot(df[['temp', 'hum', 'cnt']])
    plt.suptitle('Pairwise Scatter Plot of Temperature, Humidity, and Bike Usage', y=1.02)
    plt.show()
    

# Average Bike Users by Month:

def temp26(df):
    monthly_avg = df.groupby('mnth')['cnt'].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_avg, marker='o', linestyle='-', color='teal')
    plt.title('Average Bike Users by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Bike Users')
    plt.grid(True)
    plt.show()

# Comparison of Bike Usage on Weekdays and Weekends by Season:

def temp27(df):
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='weekday', y='cnt', hue='season', data=df, ci=None, palette='muted')
    plt.title('Comparison of Bike Usage on Weekdays and Weekends by Season')
    plt.xlabel('Day of the Week')
    plt.ylabel('Total Bike Users')
    plt.legend(title='Season', loc='upper right')
    plt.show()
    
# Bike Usage Trends Over the Years by Month:

def temp28(df):
    yearly_monthly_counts = df.groupby(['yr', 'mnth'])['cnt'].sum().unstack()

    plt.figure(figsize=(12, 8))
    yearly_monthly_counts.plot(kind='line', marker='o', colormap='viridis')
    plt.title('Bike Usage Trends Over the Years by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Bike Users')
    plt.legend(title='Year', loc='upper right')
    plt.show()
    
# Comparison of Registered and Casual Users by Temperature:

def temp29(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='temp', y='cnt', hue='registered', data=df, palette='coolwarm', alpha=0.8)
    plt.title('Comparison of Registered and Casual Users by Temperature')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Total Bike Users')
    plt.legend(title='Registered Users', loc='upper right')
    plt.show()
    
# Count of bikes during weekdays and weekends: (added)

def temp30(df):
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=df, x='hr', y='cnt', hue='weekday', ax=ax)
    ax.set_title('Count of bikes during weekdays and weekends')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Count of Bikes')
    plt.show()
 
# Count of bikes during weekdays and weekends: Unregistered users

def temp31(df):
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=df, x='hr', y='casual', hue='weekday', ax=ax)
    ax.set_title('Count of bikes during weekdays and weekends: Unregistered users')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Count of Bikes (Casual)')
    plt.show()
    
# Count of bikes during weekdays and weekends: Registered users

def temp32(df):
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=df, x='hr', y='registered', hue='weekday', ax=ax)
    ax.set_title('Count of bikes during weekdays and weekends: Registered users')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Count of Registered Bikes')
    plt.show()
    
# Count of bikes during different weathers

def temp33(df):
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=df, x='hr', y='cnt', hue='weathersit', ax=ax)
    ax.set_title('Count of bikes during different weathers')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Count of Bikes')
    plt.show()
  
# Count of bikes during different seasons

def temp34(df):
    # Assuming 'data' is your DataFrame
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.pointplot(data=df, x='hr', y='cnt', hue='season', ax=ax)
    ax.set_title('Count of bikes during different seasons')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Count of Bikes')
    plt.show()


# Correlation Matrix for Selected Features:

def temp35(df):
    corr_matrix_selected = df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_selected, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix for Selected Features')
    plt.show()
    
# Impact of Windspeed on Bike Usage During Different Weather Situations:

def temp36(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='windspeed', y='cnt', hue='weathersit', data=df, palette='coolwarm')
    plt.title('Impact of Windspeed on Bike Usage During Different Weather Situations')
    plt.xlabel('Normalized Windspeed')
    plt.ylabel('Total Bike Users')
    plt.show()
    

# Daily Trends of Casual and Registered Users:

def temp37(df):
    daily_casual = df.groupby('dteday')['casual'].sum()
    daily_registered = df.groupby('dteday')['registered'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_casual, marker='o', linestyle='-', color='orange', label='Casual Users')
    plt.plot(daily_registered, marker='o', linestyle='-', color='blue', label='Registered Users')
    plt.title('Daily Trends of Casual and Registered Users')
    plt.xlabel('Date')
    plt.ylabel('Number of Users')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Scatter Plot of Temperature vs. Total Bike Users:

def temp38(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['temp'], df['cnt'], alpha=0.5, color='green')
    plt.title('Scatter Plot of Temperature vs. Total Bike Users')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Total Bike Users')
    plt.grid(True)
    plt.show()
  
# Scatter Plot of Windspeed vs. Total Bike Users:

def temp39(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['windspeed'], df['cnt'], alpha=0.5, color='brown')
    plt.title('Scatter Plot of Windspeed vs. Total Bike Users')
    plt.xlabel('Normalized Windspeed')
    plt.ylabel('Total Bike Users')
    plt.grid(True)
    plt.show()

# Box Plot of Total Bike Users by Season

def temp40(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='season', y='cnt', data=df, palette='pastel')
    plt.title('Box Plot of Total Bike Users by Season')
    plt.xlabel('Season')
    plt.ylabel('Total Bike Users')
    plt.show()


if __name__ == "__main__":
    main()
