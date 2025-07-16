# ============================== #
#   📦 Import Libraries          #
# ============================== #
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# For ignoring warnings (cleaner output)
import warnings
warnings.filterwarnings("ignore")

print("\n✅ Libraries imported successfully!")


# ======================================== #
#   📂 Step 1: Load the Dataset            #
# ======================================== #
file_path = "air\\data.csv"
df = pd.read_csv(file_path, encoding='latin1')  # or 'ISO-8859-1'

print("\n📊 Dataset loaded!")
print("🔍 Shape of dataset:", df.shape)
print("🧾 Columns:\n", df.columns.tolist())


# ======================================== #
#   🧹 Step 2: Basic Cleanup               #
# ======================================== #
print("\n🔧 Checking for null values...")
print(df.isnull().sum())

# Optional: Rename confusing columns
df = df.rename(columns={
    'state': 'State',
    'location': 'City',
    'type': 'Area_Category',
    'so2': 'Sulphur_Dioxide',
    'no2': 'Nitrogen_Dioxide',
    'rspm': 'RSPM',
    'spm': 'SPM'
})

print("\n✅ Columns renamed for clarity!")


# ======================================== #
#   🧪 Step 3: Data Types + Basic Info     #
# ======================================== #
print("\n🧬 Dataset Info:")
print(df.info())

print("\n📈 Summary Statistics:")
print(df.describe(include='all'))


# ======================================== #
#   🧼 Step 4: Missing Data Treatment      #
# ======================================== #
print("\n🔍 Checking % of missing values per column:")
missing_percent = df.isnull().mean() * 100
print(missing_percent)

# If too much missing, decide what to do (you can use dropna, fillna etc.)
# For example, let's just fill NA pollution values with the column mean for now
for col in ['Sulphur_Dioxide', 'Nitrogen_Dioxide', 'RSPM', 'SPM']:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
        print(f"✅ Filled missing values in '{col}' with mean.")


# ======================================== #
#   📊 Step 5: Value Counts for Categories #
# ======================================== #
print("\n🧾 Unique value counts:")
for col in ['State', 'City', 'Area_Category']:
    print(f"\n{col}:\n", df[col].value_counts(normalize=True) * 100)


# # ======================================== #
# #   📈 Step 6: Pollution Trends Analysis   #
# # ======================================== #

# 👉 Average pollutants per Area_Category
print("\n📌 Average pollution levels by Area Category:")
area_pollution = df.groupby('Area_Category')[
    ['Sulphur_Dioxide', 'Nitrogen_Dioxide', 'RSPM', 'SPM']].mean()
print(area_pollution)

# 👉 Plot Area category vs pollution
area_pollution.plot(kind='bar', figsize=(10, 6))
plt.title("📊 Pollution Levels by Area Category")
plt.ylabel("Concentration (µg/m³)")
plt.xlabel("Area Category")
plt.grid(True)
plt.tight_layout()
plt.show()


# ======================================== #
#   📈 Step 7: State-wise Comparison       #
# ======================================== #
state_pollution = df.groupby('State')[
    ['Sulphur_Dioxide', 'Nitrogen_Dioxide']].mean().sort_values(by='Nitrogen_Dioxide', ascending=False)

print("\n🏞️ Top states with highest Nitrogen Dioxide levels:")
print(state_pollution.head())

state_pollution.plot(kind='bar', figsize=(14, 6))
plt.title("🌆 State-wise Average Pollution (SO2 & NO2)")
plt.ylabel("Concentration (µg/m³)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(True)
plt.show()



# # ======================================== #
# #   📊 Step 9: Correlation Matrix          #
# # ======================================== #
print("\n🧠 Correlation between pollutants:")
corr = df[['Sulphur_Dioxide', 'Nitrogen_Dioxide', 'RSPM', 'SPM']].corr()
print(corr)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("🔗 Correlation between Pollutants")
plt.tight_layout()
plt.show()
