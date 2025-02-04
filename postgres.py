import pandas as pd
import psycopg2

# Anslut till PostgreSQL-databasen
conn = psycopg2.connect(
    host="localhost",      # Byt ut med din värd
    database="postgres",   # Byt ut med din databas
    user="postgres",       # Byt ut med ditt användarnamn
    password="abc123",     # Byt ut med ditt lösenord
    port="5432"            # Byt ut med din port (oftast 5432 för PostgreSQL)
)

cursor = conn.cursor()
 
# Skapa schema om det inte finns
# cursor.execute("DROP SCHEMA IF EXISTS 'jag.sales';")
# conn.commit()
cursor.execute("CREATE SCHEMA IF NOT EXISTS jag;")
conn.commit()
 
# Skapa tabellen i schemat 'jag' om den inte finns
create_table_query = """
    CREATE TABLE IF NOT EXISTS jag.sales (          
        unemployment_insurance_fund_id SERIAL PRIMARY KEY,
        year INTEGER,
        unemployment_insurance_fund VARCHAR(50),
        gender VARCHAR(50),
        age_range_year VARCHAR(50),  
        days NUMERIC,
        amount_sek NUMERIC
    );
"""
cursor.execute(create_table_query)
conn.commit()
 
# Läs in CSV-filen
csv_file_path = r"C:\Users\tannaz.yadollahi\Desktop\Övningar python\Grupparbete\paid_unemployment_benefit_fund_year.csv"  # Sökvägen till din CSV-fil
df = pd.read_csv(csv_file_path)
 
# Ta bort mellanslag från kolumnnamnen i CSV
df.columns = df.columns.str.strip()
 
# Kontrollera kolumnnamnen i CSV
print("Kolumnnamn i CSV:", df.columns.tolist())
 
# Kontrollera att rätt kolumner finns
expected_columns = ["year", "unemployment_insurance_fund_id", "unemployment_insurance_fund", "gender", "age_range_year", "days", "amount_sek"]
missing_columns = [col for col in expected_columns if col not in df.columns]
 
if missing_columns:
    raise ValueError(f"Följande kolumner saknas i CSV-filen: {missing_columns}")
 
# Iterera genom DataFrame och lägg till data i tabellen
for index, row in df.iterrows():
    # Hantera åldersintervallet korrekt, t.ex. '25-29' -> 25
    age_str = str(row["age_range_year"]).strip()  # Ta bort extra mellanslag
 
    if age_str == "" or age_str.lower() == "nan":
        age_value = None  # Eller sätt 0 om du vill ha ett numeriskt standardvärde
    elif "-" or "+" in age_str:
        age_value = str(age_str)
    #     try:
    #         age_value = str(age_str.split("-")[1])
    #         print(age_str.split("-"))
    #         # age_value = str(age_str.split("+")[0])
    #     except ValueError:
    #         print(f"⚠️ Ogiltigt värde för åldersintervall i rad {index}: {age_str}. Sätt åldern till None.")
    #         age_value = None
    # elif "+" in age_str:
    #     try:
    #         print(age_value)
    #     except ValueError:
    #         print(f"⚠️ Ogiltigt värde för åldersintervall i rad {index}: {age_str}. Sätt åldern till None.")
    #         age_value = None
    else:
        try:
            age_value = int(age_str)
        except ValueError:
            print(f"⚠️ Ogiltigt värde för ålder i rad {index}: {age_str}. Sätt åldern till None.")
            age_value = None
 
    # Hantera felaktiga eller tomma värden för 'days' och 'amount_sek'
    try:
        days = float(row["days"]) if row["days"] != "" else None
    except ValueError:
        days = None
 
    try:
        amount_sek = float(row["amount_sek"]) if row["amount_sek"] != "" else None
    except ValueError:
        amount_sek = None
 
    # Försök att införa rad i tabellen
    insert_query = """
        INSERT INTO jag.sales (year, unemployment_insurance_fund, gender, age_range_year, days, amount_sek)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (row["year"], row["unemployment_insurance_fund"], row["gender"], str(age_value), days, amount_sek))
 
# Spara ändringar
conn.commit()
 
# Stäng anslutningen
cursor.close()
conn.close()
 
print(" CSV-filen har importerats till PostgreSQL!")
