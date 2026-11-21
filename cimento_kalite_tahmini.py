# %% [1] Kütüphaneler ve Klasör Hazırlığı
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

os.makedirs("outputs", exist_ok=True)

# %% [2] Veri Yükleme, Sütun Düzenleme ve Veri Temizliği (Faz 1)
df_raw = pd.read_excel("Concrete_Data.xls", sheet_name="Sheet1")
df_raw.columns = [
    "Cement", "Slag", "Fly_Ash", "Water",
    "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate",
    "Age", "Strength"
]

ham_satir = df_raw.shape[0]
eksik_deger = df_raw.isnull().sum().sum()
tam_kopya_sayisi = df_raw.duplicated().sum()

# Birebir aynı olan 25 kopya satırı temizle (Data Leakage önlemi)
df = df_raw.drop_duplicates().reset_index(drop=True)
ayni_girdi_farkli_olcum = df.duplicated(subset=df.columns[:-1]).sum()

print("=" * 75)
print("  FAZ 1: VERİ SETİ ÖZETİ, TEMİZLİK VE AYKIRI DEĞER (EDA) RAPORU")
print("=" * 75)
print(f"Ham Veri Boyutu                : {ham_satir} satır, {df_raw.shape[1]} sütun")
print(f"Eksik (Null) Değer Sayısı      : {eksik_deger}")
print(f"Temizlenen Tam Kopya (Duplicate): {tam_kopya_sayisi} satır")
print(f"Temizlenmiş Net Veri Boyutu    : {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"Aynı Reçete/Yaş Farklı Ölçüm   : {ayni_girdi_farkli_olcum} satır (Lab numune tekrarları)")

# Temel İstatistikler + Sıfır (0) Değer Sayısı + IQR Aykırı Değer Sayısı
istatistik_df = df.describe().T[["min", "mean", "50%", "max", "std"]].round(2)
istatistik_df.rename(columns={"50%": "median"}, inplace=True)
istatistik_df["sifir_sayisi"] = (df == 0).sum()

aykiri_sayilari = {}
for col in df.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr
    aykiri_sayilari[col] = int(((df[col] < alt_sinir) | (df[col] > ust_sinir)).sum())

istatistik_df["aykiri_deger_(IQR)"] = pd.Series(aykiri_sayilari)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
print("\n--- Temel İstatistikler, Sıfır (0) Oranları ve Aykırı Değer (IQR) Tablosu ---")
print(istatistik_df)

# %% [3] Keşifsel Veri Analizi (EDA) ve Mühendislik Grafikleri (Faz 1)
df_eda = df.copy()
df_eda["Water_Cement_Ratio"] = df_eda["Water"] / df_eda["Cement"]

# Türkçe Girdi ve Çıktı İsimlendirmeleri (Grafikler İçin)
TR_ISIMLER = {
    "Cement": "Çimento",
    "Slag": "Y.Fırın Cürufu (Slag)",
    "Fly_Ash": "Uçucu Kül (Fly Ash)",
    "Water": "Su",
    "Superplasticizer": "Akışkanlaştırıcı (SP)",
    "Coarse_Aggregate": "İri Agrega (Çakıl)",
    "Fine_Aggregate": "İnce Agrega (Kum)",
    "Age": "Kür Yaşı (Gün)",
    "Strength": "Basınç Dayanımı",
    "Water_Cement_Ratio": "Su/Çimento (W/C)",
}

# Grafik 0: Girdilerin ve Çıktının Dağılım / Aykırı Değer (Boxplot) Analizi
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Veri Seti Dağılımı ve Aykırı Değer (Boxplot) Analizi (n = 1005)", fontsize=14, fontweight="bold")
birimler = {
    "Cement": "kg/m³", "Slag": "kg/m³", "Fly_Ash": "kg/m³",
    "Water": "kg/m³", "Superplasticizer": "kg/m³", "Coarse_Aggregate": "kg/m³",
    "Fine_Aggregate": "kg/m³", "Age": "Gün", "Strength": "MPa (Hedef)"
}
for ax, col in zip(axes.flat, df.columns):
    renk = "#ff7f0e" if col == "Strength" else "#1f77b4"
    sns.boxplot(x=df[col], ax=ax, color=renk, width=0.45, fliersize=4)
    ax.set_title(f"{TR_ISIMLER[col]} [{birimler[col]}] | Aykırı: {aykiri_sayilari[col]}", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/0_eda_dagilim_ve_aykiri.png", dpi=300)

# Grafik 1: Korelasyon Matrisi (Türkçe Etiketli)
plt.figure(figsize=(10.5, 7.5))
corr_df = df_eda.rename(columns=TR_ISIMLER).corr()
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Girdiler, Su/Çimento Oranı ve Basınç Dayanımı Korelasyonu", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/1_korelasyon_matrisi.png", dpi=300)

# Grafik 1b: Mühendislik İlişkileri Saçılım (Scatter) Grafikleri (2x2 Panel)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Keşifsel Veri Analizi (EDA): Temel Mühendislik İlişkileri (Scatter)", fontsize=14, fontweight="bold")

# Panel 1: Su/Çimento Oranı (W/C) vs Basınç Dayanımı (Abrams Yasası)
ax1 = axes[0, 0]
sns.regplot(
    data=df_eda, x="Water_Cement_Ratio", y="Strength", ax=ax1,
    order=2, scatter_kws={"alpha": 0.35, "color": "#1f77b4", "s": 25},
    line_kws={"color": "#d62728", "lw": 2.5, "label": "Polinom Trend (Abrams Yasası)"}
)
ax1.set_title("1) Su/Çimento Oranı (W/C) vs Dayanım (r = -0.50)", fontweight="bold")
ax1.set_xlabel("Su / Çimento Oranı (W/C)")
ax1.set_ylabel("Basınç Dayanımı (MPa)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Çimento Miktarı vs Basınç Dayanımı
ax2 = axes[0, 1]
sns.regplot(
    data=df_eda, x="Cement", y="Strength", ax=ax2,
    scatter_kws={"alpha": 0.35, "color": "#2ca02c", "s": 25},
    line_kws={"color": "#d62728", "lw": 2.5, "label": "Doğrusal Trend"}
)
ax2.set_title("2) Çimento Miktarı vs Dayanım (r = +0.49)", fontweight="bold")
ax2.set_xlabel("Çimento Miktarı (kg/m³)")
ax2.set_ylabel("Basınç Dayanımı (MPa)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Kür Süresi (Age) vs Basınç Dayanımı (Logaritmik Hidratasyon Eğrisi)
ax3 = axes[1, 0]
ax3.scatter(df_eda["Age"], df_eda["Strength"], alpha=0.35, color="#9467bd", s=25, label="Gözlemler")
age_grid = np.linspace(1, 365, 200)
log_katsayi = np.polyfit(np.log(df_eda["Age"]), df_eda["Strength"], 1)
ax3.plot(
    age_grid, log_katsayi[0] * np.log(age_grid) + log_katsayi[1],
    color="#d62728", lw=2.5, label="Logaritmik Hidratasyon Trendi (ln(Age))"
)
ax3.axvline(28, color="black", linestyle="--", lw=1.3, label="28 Gün (Standart Kür)")
ax3.set_title("3) Kür Süresi (Yaş) vs Dayanım (Logaritmik Artış)", fontweight="bold")
ax3.set_xlabel("Kür Süresi / Yaş (Gün)")
ax3.set_ylabel("Basınç Dayanımı (MPa)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Süper Akışkanlaştırıcı vs Su Miktarı (Renk: Basınç Dayanımı)
ax4 = axes[1, 1]
sc = ax4.scatter(
    df_eda["Superplasticizer"], df_eda["Water"],
    c=df_eda["Strength"], cmap="viridis", alpha=0.7, s=30, edgecolors="none"
)
cbar = plt.colorbar(sc, ax=ax4)
cbar.set_label("Basınç Dayanımı (MPa)")
sp_grid = np.linspace(df_eda["Superplasticizer"].min(), df_eda["Superplasticizer"].max(), 100)
sp_katsayi = np.polyfit(df_eda["Superplasticizer"], df_eda["Water"], 1)
ax4.plot(
    sp_grid, sp_katsayi[0] * sp_grid + sp_katsayi[1],
    color="#d62728", lw=2.5, label="Su Azaltım Trendi (r = -0.65)"
)
ax4.set_title("4) Süper Akışkanlaştırıcı vs Su İhtiyacı ve Dayanım Etkisi", fontweight="bold")
ax4.set_xlabel("Süper Akışkanlaştırıcı (kg/m³)")
ax4.set_ylabel("Su Miktarı (kg/m³)")
ax4.legend(loc="upper right")
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/1b_muhendislik_scatter.png", dpi=300)

# %% [4] Eğitim / Test Ayrımı (%80 - %20) ve 4 Aşamalı Model Eğitimi (Faz 2 & Fiziksel Kısıtlı Model)
X = df.drop(columns=["Strength"])
y = df["Strength"]


def muhendislik_ozellikleri_uret(df_in):
    """
    Doğrusal modelde uç değer taşmalarını (multicollinearity overshoot) önleyen
    düzenlileştirilmiş fiziksel/kimyasal özellik dönüşümü:
    1) W_C_Ratio : Su / Çimento Oranı (Abrams Yasası, fiziksel aralıkta sınırlandırılmış)
    2) W_B_Ratio : Su / Toplam Bağlayıcı (Çimento + Cüruf + Uçucu Kül) Oranı
    3) Log_Age   : Logaritmik Kür Süresi ln(Age + 1) (Hidratasyon hızı fiziği)
    4) Age       : 120 günde doyuma ulaşan kür yaşı (ln(Age) ile ters işaret çarpışmasını önler)
    """
    out = df_in.copy()
    out["Age"] = np.minimum(df_in["Age"], 120.0)
    out["W_C_Ratio"] = np.clip(df_in["Water"] / df_in["Cement"], 0.25, 1.35)
    out["W_B_Ratio"] = np.clip(
        df_in["Water"] / (df_in["Cement"] + df_in["Slag"] + df_in["Fly_Ash"]), 0.20, 1.15
    )
    out["Log_Age"] = np.log1p(df_in["Age"])
    return out


X_fe = muhendislik_ozellikleri_uret(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
Xfe_train, Xfe_test, _, _ = train_test_split(
    X_fe, y, test_size=0.20, random_state=42
)

# Model 1: Yorumlanabilir Baz Model (Linear Regression - 8 Ham Girdi)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 2: Mühendislik Destekli Ridge Regression (+ W/C, W/B, ln(Age) - Katsayı Çarpışmasız)
lr_fe_model = Ridge(alpha=3.0)
lr_fe_model.fit(Xfe_train, y_train)
lr_fe_pred = lr_fe_model.predict(Xfe_test)

# Model 3: Random Forest Regressor (Brief'te Önerilen Ağaç Tabanlı Karşılaştırma Modeli)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Model 4: Fiziksel Monotonik Kısıtlı Gradient Boosting (Sıfır Zikzaklı Ana Simülasyon Modeli)
# Kısıtlar: Çimento(+1), Cüruf(+1), Uçucu Kül(+1), Su(-1), Akışkanlaştırıcı(+1), İri Agrega(0), İnce Agrega(0), Yaş(+1)
hgb_model = HistGradientBoostingRegressor(
    monotonic_cst=[1, 1, 1, -1, 1, 0, 0, 1],
    max_iter=150,
    learning_rate=0.07,
    max_leaf_nodes=25,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=42,
)
hgb_model.fit(X_train, y_train)
hgb_pred = hgb_model.predict(X_test)

print("\n" + "=" * 82)
print("  FAZ 2: MODEL PERFORMANS KARŞILAŞTIRMASI (DOĞRUSAL vs AĞAÇ vs FİZİKSEL KISITLI)")
print("=" * 82)
performans_df = pd.DataFrame([
    {
        "Model": "1. Baz Linear Regression (8 Ham Girdi)",
        "Train_R2": round(r2_score(y_train, lr_model.predict(X_train)), 4),
        "Test_R2": round(r2_score(y_test, lr_pred), 4),
        "MAE_(MPa)": round(mean_absolute_error(y_test, lr_pred), 2),
        "RMSE_(MPa)": round(root_mean_squared_error(y_test, lr_pred), 2),
    },
    {
        "Model": "2. Mühendislik Destekli LR (+W/C, W/B, ln(Age))",
        "Train_R2": round(r2_score(y_train, lr_fe_model.predict(Xfe_train)), 4),
        "Test_R2": round(r2_score(y_test, lr_fe_pred), 4),
        "MAE_(MPa)": round(mean_absolute_error(y_test, lr_fe_pred), 2),
        "RMSE_(MPa)": round(root_mean_squared_error(y_test, lr_fe_pred), 2),
    },
    {
        "Model": "3. Random Forest Regressor (Non-Linear)",
        "Train_R2": round(r2_score(y_train, rf_model.predict(X_train)), 4),
        "Test_R2": round(r2_score(y_test, rf_pred), 4),
        "MAE_(MPa)": round(mean_absolute_error(y_test, rf_pred), 2),
        "RMSE_(MPa)": round(root_mean_squared_error(y_test, rf_pred), 2),
    },
    {
        "Model": "4. Fiziksel Kısıtlı Gradient Boosting (Monotonic)",
        "Train_R2": round(r2_score(y_train, hgb_model.predict(X_train)), 4),
        "Test_R2": round(r2_score(y_test, hgb_pred), 4),
        "MAE_(MPa)": round(mean_absolute_error(y_test, hgb_pred), 2),
        "RMSE_(MPa)": round(root_mean_squared_error(y_test, hgb_pred), 2),
    },
])
print(performans_df.to_string(index=False))

# Standartlaştırılmış (StandardScaler) vs Ham Linear Regression Katsayıları
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
lr_std_model = LinearRegression().fit(X_train_std, y_train)

katsayi_karsilastirma = pd.DataFrame({
    "Ham_Katsayi_(1_Birim_MPa)": pd.Series(lr_model.coef_, index=X.columns).round(4),
    "Standart_Katsayi_(1_Std_MPa)": pd.Series(lr_std_model.coef_, index=X.columns).round(2),
    "RF_Onem_Skoru_(%)": pd.Series(rf_model.feature_importances_ * 100, index=X.columns).round(1),
}).sort_values(by="Standart_Katsayi_(1_Std_MPa)", ascending=False)

print("\n--- Linear Regression (Ham vs Standartlaştırılmış) ve RF Önem Karşılaştırması ---")
print(katsayi_karsilastirma)

# %% [5] Sunum Grafikleri: Tahmin vs Gerçek, Feature Importance ve LR Katsayıları (Faz 2)
# Grafik 2: 3 Model Yan Yana Tahmin vs Gerçek Değer Karşılaştırması
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
fig.suptitle("Tahmin vs Gerçek Basınç Dayanımı: 3 Aşamalı Model Karşılaştırması", fontsize=13.5, fontweight="bold")

model_cizimleri = [
    (axes[0], lr_pred, "1) Baz Linear Regression (Ham)", "#ff7f0e"),
    (axes[1], lr_fe_pred, "2) Mühendislik Destekli LR (+W/C, ln(Age))", "#9467bd"),
    (axes[2], hgb_pred, "3) Fiziksel Kısıtlı Ağaç Modeli (RF/HGB)", "#1f77b4"),
]

for ax, preds, baslik, renk in model_cizimleri:
    r2_val = r2_score(y_test, preds)
    mae_val = mean_absolute_error(y_test, preds)
    ax.scatter(y_test, preds, alpha=0.6, color=renk, edgecolors="k", s=35)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="İdeal Tahmin (y = x)")
    ax.set_title(f"{baslik}\nR² = {r2_val:.2f} | MAE = {mae_val:.2f} MPa", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Gerçek Basınç Dayanımı (MPa)")
    if ax == axes[0]:
        ax.set_ylabel("Tahmin Edilen Basınç Dayanımı (MPa)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("outputs/2_tahmin_vs_gercek.png", dpi=300)

# Grafik 3: Random Forest Feature Importance (Türkçe & Yüzde Etiketli)
importances = pd.Series(rf_model.feature_importances_, index=[TR_ISIMLER[c] for c in X.columns]).sort_values()
plt.figure(figsize=(9, 5.2))
bars = plt.barh(importances.index, importances.values, color="#2ca02c", edgecolor="black", height=0.65)
plt.title("Basınç Dayanımını En Çok Etkileyen Girdiler (RF Feature Importance)", fontsize=11.5, fontweight="bold")
plt.xlabel("Göreceli Önem Skoru")
plt.xlim(0, importances.max() * 1.18)
for bar, val in zip(bars, importances.values):
    plt.text(
        val + 0.005, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f} (%{val * 100:.1f})",
        va="center", ha="left", fontsize=9.5, fontweight="bold"
    )
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/3_feature_importance.png", dpi=300)

# Grafik 4: Linear Regression Katsayıları (Standartlaştırılmış vs Ham Etki Yönü - Türkçe Etiketli)
std_coef_series = pd.Series(lr_std_model.coef_, index=[TR_ISIMLER[c] for c in X.columns]).sort_values()
raw_coef_series = pd.Series(lr_model.coef_, index=[TR_ISIMLER[c] for c in X.columns]).loc[std_coef_series.index]

fig, (ax_std, ax_raw) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
fig.suptitle("Linear Regression Katsayı Analizi: Neden Ölçeklendirme (StandardScaler) Önemli?", fontsize=13, fontweight="bold")

# Sol Panel: Standartlaştırılmış Katsayılar (Gerçek Göreceli Önem ve Yön)
renkler_std = ["#d62728" if c < 0 else "#1f77b4" for c in std_coef_series]
bars_std = ax_std.barh(std_coef_series.index, std_coef_series.values, color=renkler_std, edgecolor="black", height=0.65)
ax_std.axvline(0, color="black", lw=1)
ax_std.set_title("Standartlaştırılmış Katsayılar (1 Std Sapma Artışın Etkisi)\n[Girdiler Arası Doğru Önem Sıralaması]", fontsize=10.5, fontweight="bold")
ax_std.set_xlabel("Dayanım Değişimi (MPa / 1 Std Sapma)")
ax_std.set_xlim(-6.5, std_coef_series.max() * 1.25)
ax_std.grid(True, axis="x", alpha=0.3)
for bar, val in zip(bars_std, std_coef_series.values):
    offset = 0.3 if val >= 0 else -0.3
    ha = "left" if val >= 0 else "right"
    ax_std.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.2f} MPa", va="center", ha=ha, fontsize=9, fontweight="bold")

# Sağ Panel: Ham Katsayılar (1 kg/m³ veya 1 Gün Artışın Etkisi)
renkler_raw = ["#d62728" if c < 0 else "#7f7f7f" for c in raw_coef_series]
bars_raw = ax_raw.barh(raw_coef_series.index, raw_coef_series.values, color=renkler_raw, edgecolor="black", height=0.65)
ax_raw.axvline(0, color="black", lw=1)
ax_raw.set_title("Ham Katsayılar (1 kg/m³ veya 1 Gün Artışın Etkisi)\n[Ölçek Farkı Nedeniyle Akışkanlaştırıcı Büyük Görünür]", fontsize=10.5, fontweight="bold")
ax_raw.set_xlabel("Dayanım Değişimi (MPa / 1 Birim)")
ax_raw.set_xlim(-0.23, raw_coef_series.max() * 1.28)
ax_raw.grid(True, axis="x", alpha=0.3)
for bar, val in zip(bars_raw, raw_coef_series.values):
    offset = 0.01 if val >= 0 else -0.01
    ha = "left" if val >= 0 else "right"
    ax_raw.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.4f}", va="center", ha=ha, fontsize=9, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("outputs/4_lr_katsayilar.png", dpi=300)
plt.savefig("outputs/5_lr_katsayilar.png", dpi=300)

# %% [6] Eğitilmiş Modelleri ve Referans Değerleri Simülasyon İçin Kaydet
model_paketi = {
    "hgb_model": hgb_model,
    "rf_model": rf_model,
    "lr_model": lr_model,
    "lr_fe_model": lr_fe_model,
    "feature_names": list(X.columns),
    "fe_feature_names": list(X_fe.columns),
    "medians": X.median().to_dict(),
    "mins": X.min().to_dict(),
    "maxs": X.max().to_dict(),
}
joblib.dump(model_paketi, "outputs/cimento_modelleri.joblib")

print("\nTüm grafikler ve eğitilmiş modeller 'outputs/' klasörüne kaydedildi.")
plt.show()