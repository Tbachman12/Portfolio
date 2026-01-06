# Portfolio
This project involves a comprehensive data cleaning process using MYSQL. The dataset I used in this project contained information about layoffs across multiple companies worldwide. The main goal of this project was to turn raw, messy data into a clean, structured format ready for exploratory data analysis

Steps I took in this cleaning process:

1. Staging the Data
I created a tale called layoffs_staging so that I could make sure that the origingal raw data stays untouched. I did this becuase it's the best way to prevent permanent data loss during the cleaning process.

2. Removing Duplicates
In this next step I used ROW_NUMBER() over partitions of all columns to help me find the duplicate entries. I then created a second staging table called layoffs_staging2 with an extra column row_num to quickly filter the data and delete duplicates where row_num > 1.

3. Standardize the Data
I used a technique called trimming to get rid of unnecessary white spaces from company names. I then found variations of the same indsutry in the data, for example "Cyrpto" and "CyptoCurrency". After I found these I standardized them into a single label: 'Cyrpto'. In the data I also found an error with punctuation in a country name. The country name United States had a period at the end so I had to update the data and get rid of the period. Lastly, I converted the date column from a text format into a DATE format using STR_TO_DATE to allow for time-series analysis.

4. Null & Blank Values
I found blank strings in the data, so I converted them into null for consistency. I then performed a Self-Join to populate the missing industry values. So if the company had one row with an industry and had another row without it, the script would immediately populate the null value based on the existing record for that same company.

5. Removing Rows & Columns that aren't needed
I realized in the data that there were rows and columns that I didn't need to have so I deleted records where both total_laid_off and percentage_laid_off were null becuase they did not show me actionable insights for analysis. I also got rid of the row_num column since it was no longer needed for duplicate removal.

MYSQL Techniques that I used:
CTEs(Common Table Expressions)
Window Functions(ROW_NUMBER(), PARTITION BY)
JOINS
DML Statements(INSERT INTO, UPDATE, DELETE)
DDL Statements(CREATE TABLE, ALTER TABLE)
Stirng & Date Functions(TRIM, LIKE, STR_TO_DATE)

Getting Started

1. Import the raw layoffs.csv in MYSQL
2. Run the scripts provided in the SQL file
3. The final result you will get is a cleaned table named layoffs_staging2
