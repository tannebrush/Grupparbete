
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

csv_file_path = r"C:\Users\tannaz.yadollahi\Desktop\Övningar python\Grupparbete\paid_unemployment_benefit_fund_year.csv"
df = pd.read_csv(csv_file_path, sep=',')


df.columns = df.columns.str.strip()

#Fyller saknade värden med 0
df.fillna(0, inplace=True)

#Summera 'amount_sek' per år och kön
df_total = df.groupby(['year', 'gender'], as_index=False).agg({'amount_sek': 'sum'})

#Beräkna procentuell förändring från föregående år
df_total['percent_change'] = df_total.groupby('gender')['amount_sek'].pct_change() * 100


output_csv_path = r"C:\Users\tannaz.yadollahi\Desktop\Övningar python\Grupparbete\processed_unemployment_data.csv"
df_total.to_csv(output_csv_path, index=False)
print(f"Bearbetad data har sparats till {output_csv_path}")

#VISUALISERINGAR 

#Histogram över totalt belopp för arbetslöshetsersättning
plt.figure(figsize=(10, 6))
sns.histplot(df_total['amount_sek'], bins=20, kde=True)
plt.title('Histogram för totalt belopp av arbetslöshetsersättning')
plt.xlabel('Belopp i SEK')
plt.ylabel('Frekvens')
plt.show()

#Linjediagram för att visa trender över tid för varje kön
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_total, x='year', y='amount_sek', hue='gender', marker='o')
plt.title('Linjediagram för arbetslöshetsersättning över tid för varje kön')
plt.xlabel('År')
plt.ylabel('Belopp i SEK')
plt.legend(title='Kön')
plt.show()

#Scatter plot för att visa relationen mellan olika åldersgrupper och belopp
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age_range_year', y='amount_sek', hue='gender', palette='Set2')
plt.title('Relation mellan åldersgrupper och arbetslöshetsersättning')
plt.xlabel('Åldersgrupp')
plt.ylabel('Belopp i SEK')
plt.legend(title='Kön')
plt.xticks(rotation=45)
plt.show()

#Ny graf: Procentuell förändring över tid
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_total, x='year', y='percent_change', hue='gender', marker='o')
plt.axhline(0, linestyle='--', color='gray')  # Nollinje för att visa när det minskar
plt.title('Procentuell förändring av arbetslöshetsersättning över tid')
plt.xlabel('År')
plt.ylabel('Förändring i %')
plt.legend(title='Kön')
plt.show()

#MASKININLÄRNING: REGRESSION

#Regression för män
df_regression_men = df_total[df_total['gender'] == 'men']
X_men = df_regression_men[['year']]
y_men = df_regression_men['amount_sek']

X_train_men, X_test_men, y_train_men, y_test_men = train_test_split(X_men, y_men, test_size=0.2, random_state=42)

#Träna modellen
model_men = LinearRegression()
model_men.fit(X_train_men, y_train_men)

#Prediktion för män
y_pred_men = model_men.predict(X_test_men)

#Förutsägelse för nästa år (2025) för män
pred_men_2025 = model_men.predict([[2026]])[0]
print(f"Förutsagd arbetslöshetsersättning för män 2025: {pred_men_2025:.2f} SEK")

#Resultatet av regressionen för män
plt.figure(figsize=(10, 6))
plt.scatter(X_test_men, y_test_men, label='Riktiga värden', color='blue')
plt.plot(X_test_men, y_pred_men, label='Prediktion', color='red')
plt.scatter(2026, pred_men_2025, color='green', label='Förutsägelse för 2025', zorder=5)
plt.title('Regression för att förutsäga arbetslöshetsersättning över tid (män)')
plt.xlabel('År')
plt.ylabel('Belopp i SEK')
plt.legend()
plt.show()

#Regression för kvinnor
df_regression_women = df_total[df_total['gender'] == 'women']
X_women = df_regression_women[['year']]
y_women = df_regression_women['amount_sek']

X_train_women, X_test_women, y_train_women, y_test_women = train_test_split(X_women, y_women, test_size=0.2, random_state=42)

#Träna modellen
model_women = LinearRegression()
model_women.fit(X_train_women, y_train_women)

#Prediktion för kvinnor
y_pred_women = model_women.predict(X_test_women)

#Förutsägelse för nästa år (2025) för kvinnor
pred_women_2025 = model_women.predict([[2025]])[0]
print(f"Förutsagd arbetslöshetsersättning för kvinnor 2025: {pred_women_2025:.2f} SEK")

#Resultatet av regressionen för kvinnor
plt.figure(figsize=(10, 6))
plt.scatter(X_test_women, y_test_women, label='Riktiga värden', color='blue')
plt.plot(X_test_women, y_pred_women, label='Prediktion', color='red')
plt.scatter(2025, pred_women_2025, color='green', label='Förutsägelse för 2025', zorder=5)
plt.title('Regression för att förutsäga arbetslöshetsersättning över tid (kvinnor)')
plt.xlabel('År')
plt.ylabel('Belopp i SEK')
plt.legend()
plt.show()


df_total = df.groupby('gender', as_index=False).agg({'amount_sek': 'sum'})

#Skapa regression för förutsägelse för nästa år
df_regression = df_total[['gender', 'amount_sek']]
X = np.array([0, 1]).reshape(-1, 1)  # 0 för män, 1 för kvinnor
y = df_regression['amount_sek'].values
model = LinearRegression().fit(X, y)

#Förutsägelse för 2026
pred_2026 = model.predict([[0], [1]])


#Tårtdiagram för fördelning mellan män och kvinnor
plt.figure(figsize=(6, 6))
sizes = df_total['amount_sek'].values
labels = df_total['gender']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'pink'])
plt.title('Fördelning av arbetslöshetsersättning mellan män och kvinnor')
plt.show()

print(f"Förutsägelse för män 2026: {pred_2026[0]:.2f} SEK")
print(f"Förutsägelse för kvinnor 2026: {pred_2026[1]:.2f} SEK")
