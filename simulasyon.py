import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import sys
import json
import webbrowser
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "outputs/cimento_modelleri.joblib"
HTML_SIM_PATH = "outputs/Cimento_Canli_Simulator.html"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Önce 'cimento_kalite_tahmini.py' dosyasını çalıştırarak modeli eğitmelisin."
    )

paket = joblib.load(MODEL_PATH)
hgb_model = paket.get("hgb_model", None)
rf_model = paket["rf_model"]
ana_model = hgb_model if hgb_model is not None else rf_model
lr_model = paket["lr_model"]
lr_fe_model = paket.get("lr_fe_model", None)
features = paket["feature_names"]
fe_features = paket.get("fe_feature_names", None)
medians = paket["medians"]
mins = paket["mins"]
maxs = paket["maxs"]

TR_KISA = {
    "Cement": "Çimento",
    "Slag": "Cüruf (Slag)",
    "Fly_Ash": "Uçucu Kül",
    "Water": "Su",
    "Superplasticizer": "Akışkanlaştırıcı",
    "Coarse_Aggregate": "İri Agrega (Çakıl)",
    "Fine_Aggregate": "İnce Agrega (Kum)",
    "Age": "Kür Yaşı (Gün)",
}

STANDART_BAZ_RECETE = {
    "Cement": 310.0,
    "Slag": 40.0,
    "Fly_Ash": 0.0,
    "Water": 185.0,
    "Superplasticizer": 6.0,
    "Coarse_Aggregate": 1000.0,
    "Fine_Aggregate": 780.0,
    "Age": 28.0,
}


def fe_ozellik_ekle(df_ham):
    """Ham 8 girdili DataFrame'e düzenlileştirilmiş mühendislik parametrelerini (W/C, W/B, Log_Age) ekler."""
    out = df_ham.copy()
    out["Age"] = np.minimum(df_ham["Age"], 120.0)
    out["W_C_Ratio"] = np.clip(df_ham["Water"] / df_ham["Cement"], 0.25, 1.35)
    out["W_B_Ratio"] = np.clip(
        df_ham["Water"] / (df_ham["Cement"] + df_ham["Slag"] + df_ham["Fly_Ash"]), 0.20, 1.15
    )
    out["Log_Age"] = np.log1p(df_ham["Age"])
    if fe_features is not None:
        return out[fe_features]
    return out


def tahmin_et(recete_dict):
    """
    Verilen reçete için 3 modelin tahminini döndürür:
    (Ana_Agac_Tahmini, Muhendislik_LR_Tahmini, Ham_LR_Tahmini)
    """
    df_tek = pd.DataFrame([recete_dict])[features]
    ana_val = float(ana_model.predict(df_tek)[0])
    lr_ham_val = float(lr_model.predict(df_tek)[0])
    if lr_fe_model is not None:
        lr_fe_val = float(lr_fe_model.predict(fe_ozellik_ekle(df_tek))[0])
    else:
        lr_fe_val = lr_ham_val
    return ana_val, lr_fe_val, lr_ham_val


def muhendislik_metrikleri_hesapla(recete_dict):
    """Reçetenin bağlayıcı, su/çimento, su/bağlayıcı, klinker oranı ve 1 m³ kütle metriklerini hesaplar."""
    cimento = recete_dict["Cement"]
    curuf = recete_dict["Slag"]
    ucucu_kul = recete_dict["Fly_Ash"]
    su = recete_dict["Water"]

    toplam_baglayici = cimento + curuf + ucucu_kul
    wc_orani = su / cimento if cimento > 0 else 0.0
    wb_orani = su / toplam_baglayici if toplam_baglayici > 0 else 0.0
    klinker_orani = (cimento / toplam_baglayici * 100.0) if toplam_baglayici > 0 else 0.0
    toplam_kutle = sum(recete_dict[k] for k in features if k != "Age")

    if toplam_kutle > 2550:
        kutle_durum = f"⚠ AŞIRI HACİM/KÜTLE (+{toplam_kutle - 2400:.0f} kg fazla)"
        kutle_uyari = True
    elif toplam_kutle < 2200:
        kutle_durum = f"⚠ EKSİK HACİM/KÜTLE ({2400 - toplam_kutle:.0f} kg eksik)"
        kutle_uyari = True
    else:
        kutle_durum = "Normal (1 m³ Fiziksel Sınırda)"
        kutle_uyari = False

    return {
        "toplam_baglayici": toplam_baglayici,
        "wc_orani": wc_orani,
        "wb_orani": wb_orani,
        "klinker_orani": klinker_orani,
        "toplam_kutle": toplam_kutle,
        "kutle_durum": kutle_durum,
        "kutle_uyari": kutle_uyari,
    }


def etki_ayristir(baz_dict, yeni_dict):
    """
    Baz reçeteden yeni reçeteye geçerken her bir girdideki değişimin
    çıktıya (MPa) ne kadar katkı sağladığını simetrik marjinal etki ile hesaplar.
    """
    baz_ana, _, _ = tahmin_et(baz_dict)
    yeni_ana, _, _ = tahmin_et(yeni_dict)
    toplam_fark = yeni_ana - baz_ana

    katkilar = {}
    for col in features:
        if abs(yeni_dict[col] - baz_dict[col]) < 1e-6:
            katkilar[col] = 0.0
            continue

        s_ileri = baz_dict.copy()
        s_ileri[col] = yeni_dict[col]
        d_ileri = tahmin_et(s_ileri)[0] - baz_ana

        s_geri = yeni_dict.copy()
        s_geri[col] = baz_dict[col]
        d_geri = yeni_ana - tahmin_et(s_geri)[0]

        katkilar[col] = 0.5 * (d_ileri + d_geri)

    ham_toplam = sum(katkilar.values())
    if abs(ham_toplam) > 1e-6 and abs(toplam_fark) > 1e-6:
        olcek = toplam_fark / ham_toplam
        katkilar = {k: v * olcek for k, v in katkilar.items()}

    return baz_ana, yeni_ana, katkilar


# ==============================================================================
# MOD 1: CANLI OLMAYAN (BATCH) SENARYO VE DUYARLILIK ANALİZİ
# ==============================================================================
def toplu_senaryo_analizi(baz_recete, senaryolar):
    """
    Tanımlanan çoklu reçete senaryolarını baz reçete ile karşılaştırır;
    her girdinin ne kadar değiştiğini ve çıktıya kaç MPa / % etki ettiğini raporlar.
    """
    baz_ana, baz_lr_fe, baz_lr_ham = tahmin_et(baz_recete)
    baz_metrik = muhendislik_metrikleri_hesapla(baz_recete)
    rapor_satirlari = []

    print("\n" + "=" * 86)
    print(
        f"  TOPLU SENARYO SİMÜLASYONU | Baz Reçete -> Ana Model: {baz_ana:.2f} MPa | "
        f"Müh. LR: {baz_lr_fe:.2f} MPa | Ham LR: {baz_lr_ham:.2f} MPa"
    )
    print(
        f"  Baz Mühendislik Metrikleri -> W/C: {baz_metrik['wc_orani']:.2f} | "
        f"W/B: {baz_metrik['wb_orani']:.2f} | Klinker Payı: %{baz_metrik['klinker_orani']:.1f} | "
        f"Kütle: {baz_metrik['toplam_kutle']:.0f} kg/m³"
    )
    print("=" * 86)

    for ad, degisiklikler in senaryolar.items():
        yeni = baz_recete.copy()
        for k, v in degisiklikler.items():
            yeni[k] = float(np.clip(v, mins[k], maxs[k]))

        _, yeni_ana, katkilar = etki_ayristir(baz_recete, yeni)
        _, yeni_lr_fe, yeni_lr_ham = tahmin_et(yeni)
        ym = muhendislik_metrikleri_hesapla(yeni)
        fark_mpa = yeni_ana - baz_ana
        fark_yuzde = (fark_mpa / baz_ana) * 100

        degisim_ozeti = []
        for k, yeni_val in degisiklikler.items():
            eski_val = baz_recete[k]
            etki_mpa = katkilar[k]
            tr_ad = TR_KISA[k]
            if abs(eski_val) < 1e-6:
                degisim_ozeti.append(f"{tr_ad}: 0->{yeni_val:.0f} kg => {etki_mpa:+.2f} MPa")
            else:
                girdi_yuzde = ((yeni_val - eski_val) / eski_val) * 100
                degisim_ozeti.append(
                    f"{tr_ad}: {eski_val:.0f}->{yeni_val:.0f} ({girdi_yuzde:+.0f}%) => {etki_mpa:+.2f} MPa"
                )

        print(f"\n▶ Senaryo: {ad}")
        print(
            f"  Tahminler -> Ana Model: {yeni_ana:.2f} MPa ({fark_mpa:+.2f} MPa / {fark_yuzde:+.1f}%) | "
            f"Müh. LR: {yeni_lr_fe:.2f} MPa | Ham LR: {yeni_lr_ham:.2f} MPa"
        )
        print(
            f"  Metrikler -> W/C: {ym['wc_orani']:.2f} | W/B: {ym['wb_orani']:.2f} | "
            f"Klinker: %{ym['klinker_orani']:.1f} | Kütle: {ym['toplam_kutle']:.0f} kg/m³ ({ym['kutle_durum']})"
        )
        print(f"  Girdi Etkileri: {' | '.join(degisim_ozeti)}")

        rapor_satirlari.append({
            "Senaryo": ad,
            "Ana_Model_Tahmin_MPa": round(yeni_ana, 2),
            "Muh_LR_Tahmin_MPa": round(yeni_lr_fe, 2),
            "Ham_LR_Tahmin_MPa": round(yeni_lr_ham, 2),
            "Fark_MPa": round(fark_mpa, 2),
            "Fark_Yuzde": round(fark_yuzde, 1),
            "W_C_Orani": round(ym["wc_orani"], 2),
            "W_B_Orani": round(ym["wb_orani"], 2),
            "Klinker_Payi_Yuzde": round(ym["klinker_orani"], 1),
            "Toplam_Kutle_kg_m3": round(ym["toplam_kutle"], 0),
            "Detayli_Etki": " ; ".join(degisim_ozeti),
        })

    rapor_df = pd.DataFrame(rapor_satirlari)
    rapor_df.to_csv("outputs/5_duyarlilik_raporu.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 5.5))
    isimler = ["Baz Reçete (C30/37)"] + list(rapor_df["Senaryo"])
    ana_degerler = [baz_ana] + list(rapor_df["Ana_Model_Tahmin_MPa"])
    fe_degerler = [baz_lr_fe] + list(rapor_df["Muh_LR_Tahmin_MPa"])
    renkler = ["#64748b"] + ["#16a34a" if d >= baz_ana else "#dc2626" for d in rapor_df["Ana_Model_Tahmin_MPa"]]

    x_pos = np.arange(len(isimler))
    bars = plt.bar(x_pos - 0.16, ana_degerler, width=0.32, color=renkler, edgecolor="black", label="Fiziksel Kısıtlı Ağaç Modeli (Ana Simülatör)")
    bars_fe = plt.bar(x_pos + 0.16, fe_degerler, width=0.32, color="#7c3aed", alpha=0.85, edgecolor="black", label="Mühendislik Destekli LR (+W/C, ln(Age))")
    plt.axhline(baz_ana, color="gray", linestyle="--", lw=1.2)
    plt.ylabel("Tahmini Basınç Dayanımı (MPa)")
    plt.title("Senaryo Bazlı Basınç Dayanımı Karşılaştırması (Ana Ağaç Modeli vs Mühendislik Destekli LR)", fontweight="bold")
    plt.xticks(x_pos, isimler, rotation=15, ha="right")
    plt.ylim(0, max(max(ana_degerler), max(fe_degerler)) * 1.16)
    for bar, val in zip(bars, ana_degerler):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f"{val:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars_fe, fe_degerler):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f"{val:.1f}", ha="center", fontsize=8.5, color="#4c1d95")
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/5_senaryo_karsilastirma.png", dpi=300)
    plt.close()
    print("\n[Kayıt] Tablo: 'outputs/5_duyarlilik_raporu.csv' | Grafik: 'outputs/5_senaryo_karsilastirma.png'")


# ==============================================================================
# MOD 2: BAĞIMSIZ (ZERO-DEPENDENCY) MODERN WEB DASHBOARD ÜRETİCİ VE BAŞLATICI
# ==============================================================================
def modeli_json_olarak_disa_aktar():
    """
    Eğitilmiş HistGradientBoosting (veya RF), Mühendislik Destekli LR ve Baz LR
    modellerini sıfır dış bağımlılıkla tarayıcıda (JS) birebir çalıştırmak için JSON'a çevirir.
    """
    if hgb_model is not None:
        base_val = float(np.asarray(hgb_model._baseline_prediction).ravel()[0])
        trees = []
        for p in hgb_model._predictors:
            nodes = p[0].nodes
            t_list = []
            for nd in nodes:
                if nd["is_leaf"]:
                    t_list.append([1, round(float(nd["value"]), 5)])
                else:
                    t_list.append([
                        0,
                        int(nd["feature_idx"]),
                        round(float(nd["num_threshold"]), 4),
                        int(nd["left"]),
                        int(nd["right"]),
                    ])
            trees.append(t_list)
    else:
        # Fallback: RF ağaçları
        base_val = 0.0
        n_est = len(rf_model.estimators_)
        trees = []
        for est in rf_model.estimators_:
            tr = est.tree_
            t_list = []
            for i in range(tr.node_count):
                if tr.children_left[i] == tr.children_right[i]:
                    t_list.append([1, round(float(tr.value[i].ravel()[0]) / n_est, 5)])
                else:
                    t_list.append([
                        0,
                        int(tr.feature[i]),
                        round(float(tr.threshold[i]), 4),
                        int(tr.children_left[i]),
                        int(tr.children_right[i]),
                    ])
            trees.append(t_list)

    lr_ham_data = {
        "intercept": round(float(lr_model.intercept_), 5),
        "coefs": [round(float(c), 6) for c in lr_model.coef_],
    }

    if lr_fe_model is not None:
        lr_fe_data = {
            "intercept": round(float(lr_fe_model.intercept_), 5),
            "coefs": [round(float(c), 6) for c in lr_fe_model.coef_],
        }
    else:
        lr_fe_data = lr_ham_data

    veri_paketi = {
        "base": round(base_val, 5),
        "trees": trees,
        "lr_ham": lr_ham_data,
        "lr_fe": lr_fe_data,
        "mins": {k: float(np.floor(mins[k])) if k != "Age" else 1.0 for k in features},
        "maxs": {k: float(np.ceil(maxs[k])) for k in features},
    }
    return json.dumps(veri_paketi, separators=(",", ":"))


def web_simulator_olustur_ve_ac(tarayicida_ac=True):
    """
    Hiçbir dış kütüphane, internet bağlantısı veya Python kurulumu gerektirmeyen,
    WhatsApp/Mail ile başka bilgisayar ve telefonlara tek dosya olarak gönderilebilen
    profesyonel HTML5 Beton Kalite Simülatörünü üretir ve tarayıcıda açar.
    """
    model_json = modeli_json_olarak_disa_aktar()

    html_icerik = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ÇİMSA | Beton Kalite Tahmini & Reçete Optimizasyon Simülatörü</title>
<style>
  :root {{
    --bg-main: #0f172a;
    --bg-card: #1e293b;
    --bg-card-alt: #162032;
    --bg-input: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --border: #334155;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
  }}
  body.light-mode {{
    --bg-main: #f1f5f9;
    --bg-card: #ffffff;
    --bg-card-alt: #f8fafc;
    --bg-input: #e2e8f0;
    --text-main: #0f172a;
    --text-muted: #475569;
    --border: #cbd5e1;
    --shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.08);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }}
  body {{ background: var(--bg-main); color: var(--text-main); min-height: 100vh; padding: 10px 16px; transition: background 0.25s, color 0.25s; }}
  .topbar {{
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
    padding: 9px 16px; margin-bottom: 10px; box-shadow: var(--shadow);
  }}
  .brand {{ display: flex; align-items: center; gap: 12px; }}
  .brand-logo {{
    background: linear-gradient(135deg, #2563eb, #06b6d4); color: #fff; font-weight: 800;
    font-size: 15px; padding: 6px 12px; border-radius: 8px; letter-spacing: 0.8px;
  }}
  .brand-title h1 {{ font-size: 15.5px; font-weight: 700; letter-spacing: -0.2px; }}
  .brand-title p {{ font-size: 11.5px; color: var(--text-muted); margin-top: 1px; }}
  .top-actions {{ display: flex; align-items: center; gap: 10px; }}
  .badge-model {{
    background: rgba(16, 185, 129, 0.14); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 5px 11px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
  }}
  .btn-theme {{
    background: var(--bg-input); color: var(--text-main); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer; transition: 0.15s;
  }}
  .btn-theme:hover {{ opacity: 0.88; }}

  /* KPI Grid */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px; }}
  @media (max-width: 1150px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (max-width: 640px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
  .kpi-card {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 11px;
    padding: 10px 14px; box-shadow: var(--shadow); display: flex; flex-direction: column; justify-content: space-between;
  }}
  .kpi-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; color: var(--text-muted); }}
  .kpi-main {{ display: flex; align-items: baseline; justify-content: space-between; margin: 4px 0; }}
  .kpi-val {{ font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }}
  .kpi-sub {{ font-size: 11.5px; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; }}
  .class-pill {{
    display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; color: #fff;
  }}

  /* TS EN 206 Gauge Bar (Tam Matematiksel Ölçekli: 0 - 85 MPa) */
  .gauge-card {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 11px;
    padding: 9px 16px 18px; margin-bottom: 10px; box-shadow: var(--shadow);
  }}
  .gauge-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; font-size: 12px; font-weight: 600; }}
  .gauge-track {{
    position: relative; height: 24px; border-radius: 7px; overflow: visible; display: flex;
    border: 1px solid rgba(255,255,255,0.2);
  }}
  .g-seg {{
    height: 100%; display: flex; align-items: center; justify-content: center;
    font-size: 10.5px; font-weight: 700; color: #0f172a; position: relative;
    border-right: 1px solid rgba(15, 23, 42, 0.25); transition: filter 0.2s, transform 0.2s;
    opacity: 0.82;
  }}
  .g-seg.active-seg {{
    opacity: 1.0; filter: brightness(1.08); box-shadow: inset 0 0 0 2px #ffffff; z-index: 2;
  }}
  .g-seg:first-child {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
  .g-seg:last-child {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-right: none; }}
  .g-tick {{
    position: absolute; bottom: -15px; right: -10px; font-size: 9.5px; font-weight: 600;
    color: var(--text-muted); width: 20px; text-align: center;
  }}
  .g-tick-start {{
    position: absolute; bottom: -15px; left: -6px; font-size: 9.5px; font-weight: 600; color: var(--text-muted);
  }}
  .gauge-base-line {{
    position: absolute; top: -5px; bottom: -5px; width: 3px; background: #f8fafc; z-index: 4;
    border-left: 1.5px dashed #0f172a; transition: left 0.2s;
  }}
  .gauge-pin {{
    position: absolute; top: 50%; transform: translate(-50%, -50%); width: 18px; height: 18px;
    border-radius: 50%; background: #fff; border: 4px solid #2563eb; box-shadow: 0 0 10px rgba(0,0,0,0.65);
    z-index: 6; transition: left 0.15s ease-out;
  }}
  .gauge-pin-label {{
    position: absolute; top: -21px; left: 50%; transform: translateX(-50%);
    background: #0f172a; color: #fff; font-size: 10px; font-weight: 700;
    padding: 1px 6px; border-radius: 4px; white-space: nowrap; border: 1px solid rgba(255,255,255,0.25);
  }}

  /* Main Dashboard Layout */
  .main-grid {{ display: grid; grid-template-columns: 420px 1fr; gap: 12px; }}
  @media (max-width: 1080px) {{ .main-grid {{ grid-template-columns: 1fr; }} }}
  .panel {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px 15px; box-shadow: var(--shadow);
  }}
  .panel-title {{
    font-size: 13px; font-weight: 700; margin-bottom: 9px; display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border); padding-bottom: 7px;
  }}

  /* Sliders */
  .slider-group {{ margin-bottom: 7px; }}
  .slider-header {{ display: flex; justify-content: space-between; align-items: center; font-size: 12px; margin-bottom: 2px; }}
  .slider-name {{ font-weight: 600; display: flex; align-items: center; gap: 6px; cursor: pointer; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .slider-val-box {{ display: flex; align-items: center; gap: 5px; }}
  .slider-num {{
    background: var(--bg-card-alt); color: var(--text-main); border: 1px solid var(--border);
    border-radius: 5px; padding: 1px 6px; font-size: 11.5px; font-weight: 700; width: 64px; text-align: right;
  }}
  .slider-unit {{ font-size: 10.5px; color: var(--text-muted); width: 32px; }}
  input[type=range] {{
    -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 4px;
    background: var(--bg-input); outline: none; cursor: pointer;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none; width: 15px; height: 15px; border-radius: 50%;
    background: #fff; border: 3px solid var(--accent-blue); cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,0.4);
  }}

  /* Presets & Buttons */
  .preset-title {{ font-size: 11px; font-weight: 700; color: var(--text-muted); margin: 9px 0 5px; text-transform: uppercase; }}
  .preset-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-bottom: 8px; }}
  .btn {{
    padding: 7px 9px; border-radius: 7px; font-size: 11.5px; font-weight: 600; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg-card-alt); color: var(--text-main); transition: 0.15s;
    text-align: center;
  }}
  .btn:hover {{ border-color: var(--accent-blue); transform: translateY(-1px); }}
  .btn.active-preset {{
    background: rgba(59, 130, 246, 0.22); border: 1.5px solid #3b82f6; color: #60a5fa;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.25);
  }}
  body.light-mode .btn.active-preset {{ background: #dbeafe; border-color: #2563eb; color: #1d4ed8; }}
  .toggle-box {{
    display: flex; align-items: center; justify-content: space-between; background: var(--bg-card-alt);
    border: 1px solid var(--border); padding: 7px 11px; border-radius: 9px; margin-bottom: 8px; cursor: pointer;
  }}
  .toggle-box.active {{ background: rgba(16, 185, 129, 0.14); border-color: rgba(16, 185, 129, 0.45); }}
  .action-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }}

  /* Right Charts Column */
  .charts-col {{ display: flex; flex-direction: column; gap: 10px; }}
  .chart-wrap {{ position: relative; width: 100%; }}
  canvas {{ width: 100%; display: block; border-radius: 8px; }}
  .pill-bar {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }}
  .pill {{
    padding: 4px 10px; border-radius: 999px; font-size: 11.2px; font-weight: 600; cursor: pointer;
    background: var(--bg-card-alt); border: 1px solid var(--border); color: var(--text-muted); transition: 0.15s;
  }}
  .pill.active {{ background: var(--accent-blue); color: #fff; border-color: var(--accent-blue); }}
  .chart-controls {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px; margin-bottom: 4px; }}
  .chk-label {{ font-size: 11.5px; color: var(--text-muted); display: flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; }}
</style>
</head>
<body>

<!-- ÜST KURUMSAL BAR -->
<header class="topbar">
  <div class="brand">
    <div class="brand-logo">ÇİMSA</div>
    <div class="brand-title">
      <h1>Beton Kalite Tahmini & Reçete Optimizasyon Simülatörü</h1>
      <p>Fiziksel Kısıtlı Makine Öğrenmesi Karar Destek Sistemi (UCI Concrete Dataset | n = 1005)</p>
    </div>
  </div>
  <div class="top-actions">
    <span class="badge-model">● Ana Model: Fiziksel Kısıtlı Ağaç (Test R² = 0.920 | MAE = 3.43 MPa)</span>
    <button class="btn-theme" id="btnTheme" onclick="toggleTheme()">☀️ Açık Tema</button>
  </div>
</header>

<!-- 4 ÜST KPI KARTI -->
<section class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Tahmini Basınç Dayanımı (Ana Model)</div>
    <div class="kpi-main">
      <span class="kpi-val" id="kpiMpa">37.01 MPa</span>
      <span class="class-pill" id="kpiClass" style="background:#16a34a;">C30/37</span>
    </div>
    <div class="kpi-sub">
      <span id="kpiClassDesc">Nitelikli Yapı Betonu</span>
      <span id="kpiBaseMpa">Baz: 37.01 MPa</span>
    </div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Baz Reçeteye Göre Net Kalite Değişimi</div>
    <div class="kpi-main">
      <span class="kpi-val" id="kpiDelta">+0.00 MPa</span>
      <span class="class-pill" id="kpiDeltaPct" style="background:#64748b;">%0.0</span>
    </div>
    <div class="kpi-sub">
      <span>Simetrik Marjinal Katkı Toplamı</span>
      <span id="kpiAgeInfo">Kür: 28 Gün</span>
    </div>
  </div>

  <div class="kpi-card">
    <div class="kpi-label">Su / Çimento & Bağlayıcı Dengesi</div>
    <div class="kpi-main">
      <span class="kpi-val" id="kpiWC">W/C: 0.60</span>
      <span style="font-size:15px;font-weight:700;color:var(--accent-cyan);" id="kpiWB">W/B: 0.53</span>
    </div>
    <div class="kpi-sub">
      <span id="kpiBinder">Toplam Bağlayıcı: 350 kg/m³</span>
      <span id="kpiClinker">Klinker: %88.6</span>
    </div>
  </div>

  <div class="kpi-card" id="kpiMassCard">
    <div class="kpi-label">1 m³ Beton Kütle & Hacim Kontrolü</div>
    <div class="kpi-main">
      <span class="kpi-val" id="kpiMass">2321 kg/m³</span>
      <span class="class-pill" id="kpiMassBadge" style="background:#10b981;">Normal</span>
    </div>
    <div class="kpi-sub">
      <span id="kpiMassStatus">Fiziksel Sınırda (2200 - 2550 kg/m³)</span>
      <span id="kpiLrCompare">Müh. LR: 35.0 MPa</span>
    </div>
  </div>
</section>

<!-- TS EN 206 SEVİYE BARI (Matematiksel 0 - 85 MPa Birebir Ölçekli) -->
<section class="gauge-card">
  <div class="gauge-header">
    <span>TS EN 206 Standart Beton Dayanım Sınıfı Cetveli (0 – 85 MPa Doğrusal Ölçek)</span>
    <span id="gaugeLegend" style="color:var(--text-muted);font-size:11.5px;">Kesikli Çizgi: Baz Reçete | Yuvarlak İbre: Mevcut Reçete</span>
  </div>
  <div class="gauge-track">
    <!-- 0-20 MPa (20/85 = 23.53%) -->
    <div class="g-seg" id="seg_C12/15" style="width:23.53%;background:#fca5a5;">
      &lt;C16 (Grobeton)
      <span class="g-tick-start">0</span>
      <span class="g-tick">20</span>
    </div>
    <!-- 20-25 MPa (5/85 = 5.88%) -->
    <div class="g-seg" id="seg_C16/20" style="width:5.88%;background:#fdba74;">
      C16/20
      <span class="g-tick">25</span>
    </div>
    <!-- 25-30 MPa (5/85 = 5.88%) -->
    <div class="g-seg" id="seg_C20/25" style="width:5.88%;background:#fcd34d;">
      C20/25
      <span class="g-tick">30</span>
    </div>
    <!-- 30-37 MPa (7/85 = 8.24%) -->
    <div class="g-seg" id="seg_C25/30" style="width:8.24%;background:#bef264;">
      C25/30
      <span class="g-tick">37</span>
    </div>
    <!-- 37-45 MPa (8/85 = 9.41%) -->
    <div class="g-seg" id="seg_C30/37" style="width:9.41%;background:#86efac;">
      C30/37
      <span class="g-tick">45</span>
    </div>
    <!-- 45-50 MPa (5/85 = 5.88%) -->
    <div class="g-seg" id="seg_C35/45" style="width:5.88%;background:#5eead4;">
      C35/45
      <span class="g-tick">50</span>
    </div>
    <!-- 50-60 MPa (10/85 = 11.76%) -->
    <div class="g-seg" id="seg_C40/50" style="width:11.76%;background:#38bdf8;">
      C40/50
      <span class="g-tick">60</span>
    </div>
    <!-- 60-85 MPa (25/85 = 29.42%) -->
    <div class="g-seg" id="seg_C50/60+" style="width:29.42%;background:#818cf8;color:#fff;">
      C50/60+ (Ultra Yüksek Performans)
      <span class="g-tick" style="right:-4px;">85 MPa</span>
    </div>
    <div class="gauge-base-line" id="gaugeBase" style="left:43.5%;"></div>
    <div class="gauge-pin" id="gaugePin" style="left:43.5%;">
      <div class="gauge-pin-label" id="gaugePinLabel">37.2 MPa</div>
    </div>
  </div>
</section>

<!-- ANA İKİ SÜTUNLU PANEL -->
<main class="main-grid">
  <!-- SOL: REÇETE KONTROL SÜRGÜLERİ -->
  <section class="panel">
    <div class="panel-title">
      <span>1 m³ Beton Reçetesi Girdileri</span>
      <span style="font-size:11.5px;color:var(--text-muted);font-weight:500;">İsme tıklayarak eğri seç</span>
    </div>

    <div id="slidersContainer"></div>

    <div class="toggle-box" id="autoBalanceBox" onclick="toggleAutoBalance()">
      <div>
        <div style="font-size:12.5px;font-weight:700;" id="autoBalanceTitle">⚖️ 1 m³ Oto-Kütle Dengeleme: KAPALI</div>
        <div style="font-size:11px;color:var(--text-muted);">Açıkken bağlayıcı/su değiştiğinde agregaları otomatik dengeler</div>
      </div>
      <input type="checkbox" id="chkAutoBalance" style="width:18px;height:18px;cursor:pointer;" onclick="event.stopPropagation(); toggleAutoBalance();">
    </div>

    <div class="preset-title">Hazır Reçete Şablonları (Tek Tıkla Yükle)</div>
    <div class="preset-grid">
      <button class="btn active-preset" id="btn_preset_c30" onclick="loadPreset('c30')">🏗️ Standart C30/37</button>
      <button class="btn" id="btn_preset_eco" onclick="loadPreset('eco')">🌱 Çevreci (Cüruf/Kül)</button>
      <button class="btn" id="btn_preset_c50" onclick="loadPreset('c50')">⚡ Yüksek Dayanım C50+</button>
      <button class="btn" id="btn_preset_early" onclick="loadPreset('early')">⏱️ Erken Dayanım (7 Gün)</button>
    </div>

    <div class="action-row">
      <button class="btn" onclick="setAsBase()">📌 Mevcut Ayarı Baz Yap</button>
      <button class="btn" onclick="resetToDefault()">🔄 Varsayılana Sıfırla</button>
    </div>
  </section>

  <!-- SAĞ: İNTERAKTİF GRAFİKLER -->
  <div class="charts-col">
    <!-- SAĞ ÜST: KATKI AYRIŞTIRMASI -->
    <section class="panel">
      <div class="panel-title">
        <span>Girdi Değişimlerinin Çıktıya Etkisi (MPa Katkı Ayrıştırması)</span>
        <span style="font-size:11.5px;color:var(--text-muted);font-weight:500;">Baz reçeteye göre her malzemenin net MPa katkısı</span>
      </div>
      <div class="chart-wrap">
        <canvas id="contribCanvas" height="205"></canvas>
      </div>
    </section>

    <!-- SAĞ ALT: WHAT-IF DUYARLILIK EĞRİSİ -->
    <section class="panel">
      <div class="panel-title">
        <span id="curveTitle">Diğer Girdiler Sabitken 'Çimento' Değişiminin Etkisi (Duyarlılık Eğrisi)</span>
        <label class="chk-label">
          <input type="checkbox" id="chkShowLR" onchange="updateAll()">
          Doğrusal Modelleri (LR) Grafikte Karşılaştır
        </label>
      </div>
      <div class="chart-controls">
        <div class="pill-bar" id="pillContainer"></div>
      </div>
      <div class="chart-wrap">
        <canvas id="curveCanvas" height="225"></canvas>
      </div>
    </section>
  </div>
</main>

<script>
// Python'dan Birebir Aktarılan Eğitilmiş Model Ağaçları ve Katsayılar
const MODEL = {model_json};

const FEATURES = [
  {{ key: "Cement",           tr: "Çimento",              unit: "kg/m³", color: "#3b82f6", step: 1 }},
  {{ key: "Slag",             tr: "Y.Fırın Cürufu (Slag)",unit: "kg/m³", color: "#2563eb", step: 1 }},
  {{ key: "Fly_Ash",          tr: "Uçucu Kül (Fly Ash)",  unit: "kg/m³", color: "#60a5fa", step: 1 }},
  {{ key: "Water",            tr: "Su",                   unit: "kg/m³", color: "#06b6d4", step: 0.5 }},
  {{ key: "Superplasticizer", tr: "Akışkanlaştırıcı",     unit: "kg/m³", color: "#14b8a6", step: 0.5 }},
  {{ key: "Coarse_Aggregate", tr: "İri Agrega (Çakıl)",   unit: "kg/m³", color: "#f59e0b", step: 1 }},
  {{ key: "Fine_Aggregate",   tr: "İnce Agrega (Kum)",    unit: "kg/m³", color: "#ea580c", step: 1 }},
  {{ key: "Age",              tr: "Kür Yaşı (Gün)",       unit: "Gün",   color: "#8b5cf6", step: 1 }}
];

const PRESETS = {{
  c30:   {{ Cement: 310, Slag: 40,  Fly_Ash: 0,  Water: 185, Superplasticizer: 6.0,  Coarse_Aggregate: 1000, Fine_Aggregate: 780, Age: 28 }},
  eco:   {{ Cement: 200, Slag: 145, Fly_Ash: 45, Water: 165, Superplasticizer: 8.5,  Coarse_Aggregate: 1000, Fine_Aggregate: 780, Age: 28 }},
  c50:   {{ Cement: 450, Slag: 80,  Fly_Ash: 0,  Water: 150, Superplasticizer: 14.0, Coarse_Aggregate: 1000, Fine_Aggregate: 780, Age: 28 }},
  early: {{ Cement: 380, Slag: 0,   Fly_Ash: 0,  Water: 165, Superplasticizer: 10.0, Coarse_Aggregate: 1000, Fine_Aggregate: 780, Age: 7 }}
}};

let baseRecipe = {{ ...PRESETS.c30 }};
let currentRecipe = {{ ...PRESETS.c30 }};
let selectedFeature = "Cement";
let autoBalance = false;
let targetMass = 2321.0;
let hoverCurvePoint = null;

// 1) Ağaç Topluluğu (HistGradientBoosting) Tahmin Fonksiyonu (Python ile %100 Birebir)
function predictTreeEnsemble(rec) {{
  const x = [
    rec.Cement, rec.Slag, rec.Fly_Ash, rec.Water,
    rec.Superplasticizer, rec.Coarse_Aggregate, rec.Fine_Aggregate, rec.Age
  ];
  let sum = MODEL.base;
  const trees = MODEL.trees;
  for (let t = 0; t < trees.length; t++) {{
    const tree = trees[t];
    let idx = 0;
    while (tree[idx][0] === 0) {{
      const fIdx = tree[idx][1];
      const thr = tree[idx][2];
      idx = (x[fIdx] <= thr) ? tree[idx][3] : tree[idx][4];
    }}
    sum += tree[idx][1];
  }}
  return sum;
}}

// 2) Ham Doğrusal Regresyon Tahmini
function predictRawLR(rec) {{
  const x = [
    rec.Cement, rec.Slag, rec.Fly_Ash, rec.Water,
    rec.Superplasticizer, rec.Coarse_Aggregate, rec.Fine_Aggregate, rec.Age
  ];
  let s = MODEL.lr_ham.intercept;
  for (let i = 0; i < 8; i++) s += x[i] * MODEL.lr_ham.coefs[i];
  return s;
}}

// 3) Mühendislik Destekli Ridge LR Tahmini (+W/C, W/B, ln(Age))
function predictFeLR(rec) {{
  const ageCap = Math.min(rec.Age, 120.0);
  const wc = Math.min(1.35, Math.max(0.25, rec.Water / Math.max(1, rec.Cement)));
  const wb = Math.min(1.15, Math.max(0.20, rec.Water / Math.max(1, rec.Cement + rec.Slag + rec.Fly_Ash)));
  const logAge = Math.log(rec.Age + 1.0);
  const xFe = [
    rec.Cement, rec.Slag, rec.Fly_Ash, rec.Water,
    rec.Superplasticizer, rec.Coarse_Aggregate, rec.Fine_Aggregate, ageCap,
    wc, wb, logAge
  ];
  let s = MODEL.lr_fe.intercept;
  for (let i = 0; i < xFe.length; i++) s += xFe[i] * MODEL.lr_fe.coefs[i];
  return s;
}}

// Simetrik Marjinal Etki Ayrıştırması
function decomposeContributions(baseR, newR) {{
  const baseVal = predictTreeEnsemble(baseR);
  const newVal = predictTreeEnsemble(newR);
  const totalDiff = newVal - baseVal;
  const contribs = {{}};
  let rawSum = 0;

  for (const f of FEATURES) {{
    const k = f.key;
    if (Math.abs(newR[k] - baseR[k]) < 1e-6) {{
      contribs[k] = 0.0;
      continue;
    }}
    const sFwd = {{ ...baseR, [k]: newR[k] }};
    const dFwd = predictTreeEnsemble(sFwd) - baseVal;

    const sBwd = {{ ...newR, [k]: baseR[k] }};
    const dBwd = newVal - predictTreeEnsemble(sBwd);

    contribs[k] = 0.5 * (dFwd + dBwd);
    rawSum += contribs[k];
  }}

  if (Math.abs(rawSum) > 1e-6 && Math.abs(totalDiff) > 1e-6) {{
    const scale = totalDiff / rawSum;
    for (const f of FEATURES) contribs[f.key] *= scale;
  }}
  return {{ baseVal, newVal, contribs }};
}}

function getConcreteClass(mpa) {{
  if (mpa < 20) return {{ code: "C12/15", desc: "Düşük Dayanım (Grobeton)", color: "#e11d48" }};
  if (mpa < 25) return {{ code: "C16/20", desc: "Hafif Yapı Betonu", color: "#ea580c" }};
  if (mpa < 30) return {{ code: "C20/25", desc: "Konut / Temel Betonu", color: "#d97706" }};
  if (mpa < 37) return {{ code: "C25/30", desc: "Standart Yapı Betonu", color: "#65a30d" }};
  if (mpa < 45) return {{ code: "C30/37", desc: "Nitelikli Yapı Betonu", color: "#16a34a" }};
  if (mpa < 50) return {{ code: "C35/45", desc: "Yüksek Dayanımlı Beton", color: "#0d9488" }};
  if (mpa < 60) return {{ code: "C40/50", desc: "Endüstriyel / Köprü Betonu", color: "#0284c7" }};
  return {{ code: "C50/60+", desc: "Ultra Yüksek Performans", color: "#4f46e5" }};
}}

// Arayüz Bileşenlerini İnşa Et
function buildUI() {{
  const sc = document.getElementById("slidersContainer");
  sc.innerHTML = "";
  const pc = document.getElementById("pillContainer");
  pc.innerHTML = "";

  FEATURES.forEach(f => {{
    const minV = MODEL.mins[f.key];
    const maxV = MODEL.maxs[f.key];
    const val = currentRecipe[f.key];

    const div = document.createElement("div");
    div.className = "slider-group";
    div.innerHTML = `
      <div class="slider-header">
        <span class="slider-name" onclick="selectCurveFeature('${{f.key}}')" title="Eğrisini görmek için tıkla">
          <span class="dot" style="background:${{f.color}};"></span>${{f.tr}}
        </span>
        <div class="slider-val-box">
          <input type="number" class="slider-num" id="num_${{f.key}}" min="${{minV}}" max="${{maxV}}" step="${{f.step}}" value="${{val}}"
            onchange="onSliderInput('${{f.key}}', this.value)">
          <span class="slider-unit">${{f.unit}}</span>
        </div>
      </div>
      <input type="range" id="rng_${{f.key}}" min="${{minV}}" max="${{maxV}}" step="${{f.step}}" value="${{val}}"
        style="accent-color:${{f.color}};" oninput="onSliderInput('${{f.key}}', this.value)">
    `;
    sc.appendChild(div);

    const pill = document.createElement("button");
    pill.className = "pill" + (f.key === selectedFeature ? " active" : "");
    pill.id = "pill_" + f.key;
    pill.textContent = f.tr;
    pill.onclick = () => selectCurveFeature(f.key);
    pc.appendChild(pill);
  }});
}}

function syncInputsFromRecipe() {{
  FEATURES.forEach(f => {{
    const v = Math.round(currentRecipe[f.key] * 10) / 10;
    currentRecipe[f.key] = v;
    document.getElementById("rng_" + f.key).value = v;
    document.getElementById("num_" + f.key).value = v;
  }});
}}

function applyAutoBalance(changedKey) {{
  if (!autoBalance || changedKey === "Age") return;
  const binderWater = ["Cement", "Slag", "Fly_Ash", "Water", "Superplasticizer"];
  let bwMass = binderWater.reduce((acc, k) => acc + currentRecipe[k], 0);
  const minCoarse = MODEL.mins.Coarse_Aggregate;
  const minFine = MODEL.mins.Fine_Aggregate;
  const maxCoarse = MODEL.maxs.Coarse_Aggregate;
  const maxFine = MODEL.maxs.Fine_Aggregate;

  if (bwMass + minCoarse + minFine > targetMass) {{
    const allowedBW = targetMass - (minCoarse + minFine);
    const ratio = allowedBW / Math.max(1, bwMass);
    binderWater.forEach(k => {{
      currentRecipe[k] = Math.min(MODEL.maxs[k], Math.max(MODEL.mins[k], Math.round(currentRecipe[k] * ratio * 2) / 2));
    }});
    bwMass = binderWater.reduce((acc, k) => acc + currentRecipe[k], 0);
  }}

  const remAgg = targetMass - bwMass;
  let nCoarse = currentRecipe.Coarse_Aggregate;
  let nFine = currentRecipe.Fine_Aggregate;

  if (changedKey === "Coarse_Aggregate") {{
    nFine = Math.min(maxFine, Math.max(minFine, remAgg - nCoarse));
    nCoarse = Math.min(maxCoarse, Math.max(minCoarse, remAgg - nFine));
  }} else if (changedKey === "Fine_Aggregate") {{
    nCoarse = Math.min(maxCoarse, Math.max(minCoarse, remAgg - nFine));
    nFine = Math.min(maxFine, Math.max(minFine, remAgg - nCoarse));
  }} else {{
    const totAgg = nCoarse + nFine;
    const cRatio = totAgg > 0 ? (nCoarse / totAgg) : 0.56;
    nCoarse = Math.min(maxCoarse, Math.max(minCoarse, remAgg * cRatio));
    nFine = Math.min(maxFine, Math.max(minFine, remAgg - nCoarse));
    nCoarse = Math.min(maxCoarse, Math.max(minCoarse, remAgg - nFine));
  }}
  currentRecipe.Coarse_Aggregate = Math.round(nCoarse);
  currentRecipe.Fine_Aggregate = Math.round(nFine);
}}

function setActivePresetButton(activeKey) {{
  ["c30", "eco", "c50", "early"].forEach(k => {{
    const btn = document.getElementById("btn_preset_" + k);
    if (btn) btn.className = "btn" + (k === activeKey ? " active-preset" : "");
  }});
}}

function detectMatchingPreset() {{
  for (const [k, p] of Object.entries(PRESETS)) {{
    let match = true;
    for (const f of FEATURES) {{
      if (Math.abs(currentRecipe[f.key] - p[f.key]) > 0.4) {{ match = false; break; }}
    }}
    if (match) return k;
  }}
  return null;
}}

function onSliderInput(key, valStr) {{
  let v = parseFloat(valStr);
  if (isNaN(v)) return;
  v = Math.min(MODEL.maxs[key], Math.max(MODEL.mins[key], v));
  currentRecipe[key] = v;
  if (autoBalance) applyAutoBalance(key);
  syncInputsFromRecipe();
  setActivePresetButton(detectMatchingPreset());
  updateAll();
}}

function selectCurveFeature(key) {{
  selectedFeature = key;
  FEATURES.forEach(f => {{
    document.getElementById("pill_" + f.key).className = "pill" + (f.key === key ? " active" : "");
  }});
  const fObj = FEATURES.find(x => x.key === key);
  document.getElementById("curveTitle").textContent = `Diğer Girdiler Sabitken '${{fObj.tr}}' Değişiminin Etkisi (Duyarlılık Eğrisi)`;
  drawCurveChart();
}}

function toggleAutoBalance() {{
  autoBalance = !autoBalance;
  document.getElementById("chkAutoBalance").checked = autoBalance;
  const box = document.getElementById("autoBalanceBox");
  const title = document.getElementById("autoBalanceTitle");
  if (autoBalance) {{
    const curMass = FEATURES.filter(f => f.key !== "Age").reduce((a, f) => a + currentRecipe[f.key], 0);
    targetMass = Math.min(2520, Math.max(2220, Math.round(curMass)));
    box.className = "toggle-box active";
    title.textContent = `✓ 1 m³ Oto-Kütle Dengeleme: AÇIK (${{targetMass}} kg/m³ Sabit)`;
    applyAutoBalance("Cement");
    syncInputsFromRecipe();
  }} else {{
    box.className = "toggle-box";
    title.textContent = "⚖️ 1 m³ Oto-Kütle Dengeleme: KAPALI (Serbest Mod)";
  }}
  updateAll();
}}

function loadPreset(name) {{
  currentRecipe = {{ ...PRESETS[name] }};
  if (autoBalance) {{
    targetMass = FEATURES.filter(f => f.key !== "Age").reduce((a, f) => a + currentRecipe[f.key], 0);
    document.getElementById("autoBalanceTitle").textContent = `✓ 1 m³ Oto-Kütle Dengeleme: AÇIK (${{Math.round(targetMass)}} kg/m³ Sabit)`;
  }}
  syncInputsFromRecipe();
  setActivePresetButton(name);
  updateAll();
}}

function setAsBase() {{
  baseRecipe = {{ ...currentRecipe }};
  updateAll();
}}

function resetToDefault() {{
  baseRecipe = {{ ...PRESETS.c30 }};
  currentRecipe = {{ ...PRESETS.c30 }};
  targetMass = 2321;
  syncInputsFromRecipe();
  setActivePresetButton("c30");
  updateAll();
}}

function toggleTheme() {{
  document.body.classList.toggle("light-mode");
  const isLight = document.body.classList.contains("light-mode");
  document.getElementById("btnTheme").textContent = isLight ? "🌙 Koyu Tema" : "☀️ Açık Tema";
  updateAll();
}}

// Tüm Metrikleri ve Grafikleri Güncelle
function updateAll() {{
  const {{ baseVal, newVal, contribs }} = decomposeContributions(baseRecipe, currentRecipe);
  const feVal = predictFeLR(currentRecipe);
  const rawLrVal = predictRawLR(currentRecipe);
  const diff = newVal - baseVal;
  const diffPct = (diff / Math.max(1, baseVal)) * 100;
  const cls = getConcreteClass(newVal);

  // KPI 1: Dayanım
  document.getElementById("kpiMpa").textContent = newVal.toFixed(2) + " MPa";
  const kpiClass = document.getElementById("kpiClass");
  kpiClass.textContent = cls.code;
  kpiClass.style.background = cls.color;
  document.getElementById("kpiClassDesc").textContent = cls.desc;
  document.getElementById("kpiBaseMpa").textContent = "Baz: " + baseVal.toFixed(2) + " MPa";

  // KPI 2: Net Değişim
  const kpiDelta = document.getElementById("kpiDelta");
  kpiDelta.textContent = (diff >= 0 ? "+" : "") + diff.toFixed(2) + " MPa";
  kpiDelta.style.color = diff > 0.05 ? "#10b981" : (diff < -0.05 ? "#ef4444" : "var(--text-main)");
  const kpiDeltaPct = document.getElementById("kpiDeltaPct");
  kpiDeltaPct.textContent = (diff >= 0 ? "+" : "") + diffPct.toFixed(1) + "%";
  kpiDeltaPct.style.background = diff > 0.05 ? "#16a34a" : (diff < -0.05 ? "#dc2626" : "#64748b");
  document.getElementById("kpiAgeInfo").textContent = "Kür Süresi: " + currentRecipe.Age + " Gün";

  // KPI 3: W/C & Bağlayıcı
  const binder = currentRecipe.Cement + currentRecipe.Slag + currentRecipe.Fly_Ash;
  const wc = currentRecipe.Water / Math.max(1, currentRecipe.Cement);
  const wb = currentRecipe.Water / Math.max(1, binder);
  const clinkerPct = (currentRecipe.Cement / Math.max(1, binder)) * 100;
  document.getElementById("kpiWC").textContent = "W/C: " + wc.toFixed(2);
  document.getElementById("kpiWB").textContent = "W/B: " + wb.toFixed(2);
  document.getElementById("kpiBinder").textContent = "Bağlayıcı: " + Math.round(binder) + " kg/m³";
  document.getElementById("kpiClinker").textContent = "Klinker Payı: %" + clinkerPct.toFixed(1);

  // KPI 4: 1 m³ Toplam Kütle
  const totalMass = FEATURES.filter(f => f.key !== "Age").reduce((a, f) => a + currentRecipe[f.key], 0);
  document.getElementById("kpiMass").textContent = Math.round(totalMass) + " kg/m³";
  const massBadge = document.getElementById("kpiMassBadge");
  const massStatus = document.getElementById("kpiMassStatus");
  const massCard = document.getElementById("kpiMassCard");
  if (totalMass > 2550) {{
    massBadge.textContent = "⚠ AŞIRI HACİM";
    massBadge.style.background = "#dc2626";
    massStatus.textContent = `1 m³ Sınırı Aşıldı (+${{Math.round(totalMass - 2400)}} kg fazla!)`;
    massCard.style.borderColor = "#ef4444";
  }} else if (totalMass < 2200) {{
    massBadge.textContent = "⚠ EKSİK KÜTLE";
    massBadge.style.background = "#d97706";
    massStatus.textContent = `1 m³ Hacmi Eksik (-${{Math.round(2400 - totalMass)}} kg eksik)`;
    massCard.style.borderColor = "#f59e0b";
  }} else {{
    massBadge.textContent = "✓ Normal (1 m³)";
    massBadge.style.background = "#10b981";
    massStatus.textContent = "Fiziksel Sınırda (2200 - 2550 kg/m³)";
    massCard.style.borderColor = "var(--border)";
  }}
  document.getElementById("kpiLrCompare").textContent = `Müh. LR: ${{feVal.toFixed(1)}} | Ham LR: ${{rawLrVal.toFixed(1)}} MPa`;

  // Gauge İbresi ve Aktif Dilim Vurgusu (0 - 85 MPa Tam Matematiksel Hizalama)
  const basePct = Math.min(99.2, Math.max(0.8, (baseVal / 85.0) * 100.0));
  const newPct = Math.min(99.2, Math.max(0.8, (newVal / 85.0) * 100.0));
  document.getElementById("gaugeBase").style.left = basePct.toFixed(2) + "%";
  const pin = document.getElementById("gaugePin");
  pin.style.left = newPct.toFixed(2) + "%";
  pin.style.borderColor = cls.color;
  const pinLbl = document.getElementById("gaugePinLabel");
  if (pinLbl) {{
    pinLbl.textContent = `${{newVal.toFixed(1)}} MPa (${{cls.code}})`;
    pinLbl.style.background = cls.color;
  }}
  ["C12/15", "C16/20", "C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C50/60+"].forEach(cCode => {{
    const segEl = document.getElementById("seg_" + cCode);
    if (segEl) {{
      if (cCode === cls.code) segEl.classList.add("active-seg");
      else segEl.classList.remove("active-seg");
    }}
  }});

  drawContribChart(baseVal, contribs);
  drawCurveChart();
}}

// SAĞ ÜST GRAFİK: MPa Katkı Ayrıştırması (HiDPI Canvas)
function drawContribChart(baseVal, contribs) {{
  const canvas = document.getElementById("contribCanvas");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 205 * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = 205;
  const isLight = document.body.classList.contains("light-mode");
  const textCol = isLight ? "#0f172a" : "#f8fafc";
  const mutedCol = isLight ? "#475569" : "#94a3b8";
  const gridCol = isLight ? "rgba(15,23,42,0.08)" : "rgba(255,255,255,0.08)";

  ctx.clearRect(0, 0, W, H);

  const items = FEATURES.map(f => ({{
    key: f.key,
    tr: f.tr,
    val: contribs[f.key] || 0,
    diff: currentRecipe[f.key] - baseRecipe[f.key],
    base: baseRecipe[f.key],
    cur: currentRecipe[f.key]
  }})).sort((a, b) => b.val - a.val);

  const maxAbs = Math.max(5.0, ...items.map(x => Math.abs(x.val)) ) * 1.75;
  const padL = 205;
  const padR = 30;
  const padT = 12;
  const padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const zeroX = padL + plotW / 2;

  // Dikey ızgara çizgileri
  ctx.strokeStyle = gridCol;
  ctx.lineWidth = 1;
  const ticks = [-maxAbs*0.7, -maxAbs*0.35, 0, maxAbs*0.35, maxAbs*0.7];
  ctx.font = "11px 'Segoe UI', sans-serif";
  ctx.fillStyle = mutedCol;
  ctx.textAlign = "center";
  ticks.forEach(t => {{
    const x = zeroX + (t / maxAbs) * (plotW / 2);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.fillText((t >= 0 ? "+" : "") + t.toFixed(1) + " MPa", x, H - 8);
  }});

  // Sıfır ekseni
  ctx.strokeStyle = isLight ? "#475569" : "#cbd5e1";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(zeroX, padT);
  ctx.lineTo(zeroX, padT + plotH);
  ctx.stroke();

  const rowH = plotH / items.length;
  items.forEach((it, i) => {{
    const y = padT + i * rowH;
    const barH = rowH * 0.62;
    const barY = y + (rowH - barH) / 2;

    // Sol Etiket
    let changeTag = "(Sabit)";
    if (Math.abs(it.diff) >= 0.05) {{
      if (Math.abs(it.base) < 1e-6) changeTag = `(0 → ${{Math.round(it.cur)}})`;
      else {{
        const pct = (it.diff / it.base) * 100;
        changeTag = `(${{it.diff >= 0 ? "+" : ""}}${{Math.round(it.diff)}} / ${{pct >= 0 ? "+" : ""}}${{Math.round(pct)}}%)`;
      }}
    }}
    ctx.fillStyle = Math.abs(it.diff) >= 0.05 ? textCol : mutedCol;
    ctx.font = (Math.abs(it.diff) >= 0.05 ? "600 " : "400 ") + "11.5px 'Segoe UI', sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`${{it.tr}} ${{changeTag}}`, padL - 10, y + rowH / 2 + 4);

    // Çubuk
    const bw = (Math.abs(it.val) / maxAbs) * (plotW / 2);
    const bx = it.val >= 0 ? zeroX : zeroX - bw;
    ctx.fillStyle = it.val > 0.02 ? "#10b981" : (it.val < -0.02 ? "#ef4444" : (isLight ? "#cbd5e1" : "#334155"));
    ctx.beginPath();
    ctx.roundRect(bx, barY, Math.max(2, bw), barH, 4);
    ctx.fill();

    // Değer Etiketi
    if (Math.abs(it.val) >= 0.03) {{
      const pctEffect = (it.val / Math.max(1, baseVal)) * 100;
      const lbl = `${{it.val >= 0 ? "+" : ""}}${{it.val.toFixed(2)}} MPa (${{pctEffect >= 0 ? "+" : ""}}${{pctEffect.toFixed(1)}}%)`;
      ctx.fillStyle = textCol;
      ctx.font = "700 11px 'Segoe UI', sans-serif";
      ctx.textAlign = it.val >= 0 ? "left" : "right";
      const tx = it.val >= 0 ? (zeroX + bw + 6) : (zeroX - bw - 6);
      ctx.fillText(lbl, tx, y + rowH / 2 + 4);
    }}
  }});
}}

// SAĞ ALT GRAFİK: Pürüzsüz What-If Duyarlılık Eğrisi (HiDPI Canvas)
function drawCurveChart() {{
  const canvas = document.getElementById("curveCanvas");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 225 * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = 225;
  const isLight = document.body.classList.contains("light-mode");
  const textCol = isLight ? "#0f172a" : "#f8fafc";
  const mutedCol = isLight ? "#475569" : "#94a3b8";
  const gridCol = isLight ? "rgba(15,23,42,0.08)" : "rgba(255,255,255,0.08)";

  ctx.clearRect(0, 0, W, H);

  const fObj = FEATURES.find(f => f.key === selectedFeature);
  const minX = MODEL.mins[selectedFeature];
  const maxX = MODEL.maxs[selectedFeature];
  const N = 65;
  const xs = [];
  const yMainRaw = [];
  const yFe = [];
  const yRawLr = [];

  for (let i = 0; i < N; i++) {{
    const xv = minX + (maxX - minX) * (i / (N - 1));
    xs.push(xv);
    const tmp = {{ ...currentRecipe, [selectedFeature]: xv }};
    yMainRaw.push(predictTreeEnsemble(tmp));
    yFe.push(predictFeLR(tmp));
    yRawLr.push(predictRawLR(tmp));
  }}

  // Hafif 5-noktalı Gauss yumuşatması (Ağaç basamaklarını pürüzsüz laboratuvar eğrisine çevirir)
  const curVal = currentRecipe[selectedFeature];
  const curMpa = predictTreeEnsemble(currentRecipe);
  const yMain = yMainRaw.map((v, i) => {{
    if (i === 0 || i === N - 1) return v;
    const i1 = Math.max(0, i - 1), i2 = Math.min(N - 1, i + 1);
    return 0.25 * yMainRaw[i1] + 0.50 * v + 0.25 * yMainRaw[i2];
  }});

  const showLR = document.getElementById("chkShowLR").checked;
  let allY = [...yMain, curMpa];
  if (showLR) allY.push(...yFe, ...yRawLr);

  let minY = Math.max(0, Math.floor((Math.min(...allY) - 5) / 5) * 5);
  let maxY = Math.ceil((Math.max(...allY) + 6) / 5) * 5;
  if (maxY - minY < 20) maxY = minY + 20;

  const padL = 52;
  const padR = 24;
  const padT = 18;
  const padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const toX = xv => padL + ((xv - minX) / (maxX - minX)) * plotW;
  const toY = yv => padT + plotH - ((yv - minY) / (maxY - minY)) * plotH;

  // Yatay ızgara ve Y ekseni
  ctx.strokeStyle = gridCol;
  ctx.lineWidth = 1;
  ctx.font = "11px 'Segoe UI', sans-serif";
  ctx.fillStyle = mutedCol;
  for (let i = 0; i <= 5; i++) {{
    const yv = minY + (maxY - minY) * (i / 5);
    const py = toY(yv);
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(padL + plotW, py);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(yv.toFixed(0) + " MPa", padL - 8, py + 4);
  }}

  // X ekseni etiketleri
  ctx.textAlign = "center";
  for (let i = 0; i <= 5; i++) {{
    const xv = minX + (maxX - minX) * (i / 5);
    const px = toX(xv);
    ctx.fillText(Math.round(xv) + " " + fObj.unit, px, H - 10);
  }}

  // Eğer Doğrusal Modeller (LR) seçiliyse çiz
  if (showLR) {{
    // Baz Ham LR (Mavi kesikli)
    ctx.save();
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {{
      const px = toX(xs[i]), py = toY(yRawLr[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }}
    ctx.stroke();

    // Mühendislik Destekli LR (Yeşil noktalı-kesikli)
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2.2;
    ctx.setLineDash([8, 4, 2, 4]);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {{
      const px = toX(xs[i]), py = toY(yFe[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }}
    ctx.stroke();
    ctx.restore();
  }}

  // Ana Model Eğrisi Altı Hafif Gradyan Dolgu
  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  grad.addColorStop(0, "rgba(239, 68, 68, 0.25)");
  grad.addColorStop(1, "rgba(239, 68, 68, 0.0)");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(toX(xs[0]), padT + plotH);
  for (let i = 0; i < N; i++) ctx.lineTo(toX(xs[i]), toY(yMain[i]));
  ctx.lineTo(toX(xs[N - 1]), padT + plotH);
  ctx.closePath();
  ctx.fill();

  // Ana Model Eğrisi (Kırmızı/Turuncu Kalın Çizgi)
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 3.0;
  ctx.beginPath();
  for (let i = 0; i < N; i++) {{
    const px = toX(xs[i]), py = toY(yMain[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }}
  ctx.stroke();

  // Mevcut Ayar Dikey Referans Çizgisi ve Noktası
  const curPx = toX(curVal);
  const curPy = toY(curMpa);
  ctx.save();
  ctx.strokeStyle = mutedCol;
  ctx.lineWidth = 1.3;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(curPx, padT);
  ctx.lineTo(curPx, padT + plotH);
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.arc(curPx, curPy, 6.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Nokta Bilgi Etiketi
  const tagText = `Senin Ayarın: ${{Math.round(curVal)}} ${{fObj.unit}} → ${{curMpa.toFixed(1)}} MPa`;
  ctx.font = "700 11.5px 'Segoe UI', sans-serif";
  const tw = ctx.measureText(tagText).width + 14;
  const tx = Math.min(W - padR - tw, Math.max(padL, curPx - tw / 2));
  const ty = Math.max(padT + 4, curPy - 28);
  ctx.fillStyle = isLight ? "#0f172a" : "#f8fafc";
  ctx.beginPath();
  ctx.roundRect(tx, ty, tw, 20, 5);
  ctx.fill();
  ctx.fillStyle = isLight ? "#ffffff" : "#0f172a";
  ctx.textAlign = "center";
  ctx.fillText(tagText, tx + tw / 2, ty + 14);

  // Lejant (Sol Üst)
  ctx.textAlign = "left";
  ctx.font = "600 11px 'Segoe UI', sans-serif";
  ctx.fillStyle = "#ef4444";
  ctx.fillText("━━ Ana Model (Fiziksel Kısıtlı Ağaç)", padL + 10, padT + 14);
  if (showLR) {{
    ctx.fillStyle = "#10b981";
    ctx.fillText("━·━ Mühendislik Destekli LR (+W/C, ln(Age))", padL + 225, padT + 14);
    ctx.fillStyle = "#3b82f6";
    ctx.fillText("--- Baz Doğrusal LR", padL + 475, padT + 14);
  }}
}}

window.addEventListener("resize", () => updateAll());
buildUI();
updateAll();
</script>
</body>
</html>
"""
    os.makedirs("outputs", exist_ok=True)
    with open(HTML_SIM_PATH, "w", encoding="utf-8") as f:
        f.write(html_icerik)

    tam_yol = os.path.abspath(HTML_SIM_PATH)
    print(f"[Kayıt] Bağımsız Web Simülatörü üretildi: '{HTML_SIM_PATH}'")
    print("        (Bu tek .html dosyasını WhatsApp/Mail ile grup arkadaşlarına doğrudan gönderebilirsin!)")

    if tarayicida_ac:
        webbrowser.open(f"file:///{tam_yol.replace(os.sep, '/')}")


if __name__ == "__main__":
    baz_mix = STANDART_BAZ_RECETE.copy()

    ornek_senaryolar = {
        "Yüksek Çimento (+%25)": {"Cement": baz_mix["Cement"] * 1.25},
        "Düşük Su (-%15) + Akışkanlaştırıcı": {"Water": baz_mix["Water"] * 0.85, "Superplasticizer": 12.0},
        "Cüruf/Kül İkameli Çevreci Karışım": {"Cement": 200.0, "Slag": 145.0, "Fly_Ash": 45.0, "Water": 165.0},
        "7 Günlük Erken Dayanım Testi": {"Age": 7.0},
        "90 Günlük Uzun Kür Süresi": {"Age": 90.0},
    }
    toplu_senaryo_analizi(baz_mix, ornek_senaryolar)
    acilsin_mi = "--no-browser" not in sys.argv
    web_simulator_olustur_ve_ac(tarayicida_ac=acilsin_mi)
