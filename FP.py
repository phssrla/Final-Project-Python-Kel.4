import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Pemetaan Prioritas Kesehatan Indonesia", layout="wide")
st.title("📊 Pemetaan Wilayah Prioritas Pembangunan Kesehatan")

# ============================================================
# FUNCTION LOAD EXCEL BPS
# ============================================================

def load_clean_excel(path, columns):
    df = pd.read_excel(path, skiprows=4, header=None)
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    df = df.dropna(subset=["provinsi"])

    for col in columns:
        if col != "provinsi":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ============================================================
# LOAD DATA
# ============================================================

df_sanitasi = load_clean_excel("sanitasi.xlsx", ["provinsi", "sanitasi"])
df_keluhan = load_clean_excel("keluhan.xlsx", ["provinsi", "keluhan"])
df_ahh = load_clean_excel("ahh.xlsx", ["provinsi", "laki", "perempuan"])

df_ahh["AHH_rata2"] = (df_ahh["laki"] + df_ahh["perempuan"]) / 2
df_ahh = df_ahh[["provinsi", "AHH_rata2"]]

# DATASET MURNI
df_dataset = (
    df_sanitasi
    .merge(df_ahh, on="provinsi")
    .merge(df_keluhan, on="provinsi")
)

df_dataset.index = df_dataset.index + 1

df_dataset = df_dataset.rename(columns={
    "provinsi": "Provinsi",
    "sanitasi": "Persentase Akses Sanitasi Layak (%)",
    "AHH_rata2": "Angka Harapan Hidup Rata-rata (Tahun)",
    "keluhan": "Persentase Keluhan Kesehatan (%)"
})

df_dataset = df_dataset[df_dataset["Provinsi"].str.upper() != "INDONESIA"]
df_dataset = df_dataset.reset_index(drop=True)
df_dataset.index = df_dataset.index + 1

# ============================================================
# PROSES CLUSTERING
# ============================================================

X = df_dataset[
    [
        "Persentase Akses Sanitasi Layak (%)",
        "Angka Harapan Hidup Rata-rata (Tahun)",
        "Persentase Keluhan Kesehatan (%)"
    ]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df_cluster = df_dataset.copy()
df_cluster["Cluster"] = cluster_labels

# ============================================================
# PEMBERIAN LABEL ZONA PRIORITAS
# ============================================================

# Hitung rata-rata indikator tiap cluster
cluster_profile = (
    df_cluster
    .groupby("Cluster")[
        [
            "Persentase Akses Sanitasi Layak (%)",
            "Angka Harapan Hidup Rata-rata (Tahun)",
            "Persentase Keluhan Kesehatan (%)"
        ]
    ]
    .mean()
)

# Urutkan cluster dari kondisi TERBURUK ke TERBAIK
cluster_profile_sorted = cluster_profile.sort_values(
    by=[
        "Angka Harapan Hidup Rata-rata (Tahun)",
        "Persentase Akses Sanitasi Layak (%)",
        "Persentase Keluhan Kesehatan (%)"
    ],
    ascending=[True, True, False]
)

# Mapping label zona
zona_labels = {
    cluster_profile_sorted.index[0]: "Zona Prioritas Tinggi",
    cluster_profile_sorted.index[1]: "Zona Prioritas Sedang",
    cluster_profile_sorted.index[2]: "Zona Prioritas Rendah"
}

# Tambahkan kolom Zona ke dataset clustering
df_cluster["Zona Prioritas"] = df_cluster["Cluster"].map(zona_labels)


# ============================================================
# BUTTON MENU
# ============================================================

col1, col2, col3, col4 = st.columns(4)
show_dataset = col1.button("📁 Dataset")
show_cluster = col2.button("🧩 Clustering")
show_visual = col3.button("📈 Visualisasi")
show_insight = col4.button("🧠 Insight")

# ============================================================
# DATASET
# ============================================================

if show_dataset:
    st.subheader("📁 Dataset Kesehatan")
    st.dataframe(df_dataset, use_container_width=True)

# ============================================================
# CLUSTERING
# ============================================================

if show_cluster:
    st.subheader("🧩 Hasil K-Means Clustering")
    st.dataframe(df_cluster, use_container_width=True)

# ============================================================
# VISUALISASI CLUSTER (RAPI + TANPA INDONESIA)
# ============================================================

if show_visual:
    st.subheader("📈 Pemetaan Wilayah Prioritas Pembangunan Kesehatan")

    fig, ax = plt.subplots(figsize=(13, 9))

    sns.scatterplot(
        data=df_cluster,
        x="Persentase Akses Sanitasi Layak (%)",
        y="Angka Harapan Hidup Rata-rata (Tahun)",
        hue="Zona Prioritas",
        palette={
            "Zona Prioritas Tinggi": "#d62728",   # merah
            "Zona Prioritas Sedang": "#ff7f0e",   # oranye
            "Zona Prioritas Rendah": "#2ca02c"    # hijau
        },
        s=120,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    # Label Provinsi
    for _, row in df_cluster.iterrows():
        ax.annotate(
            row["Provinsi"],
            (row["Persentase Akses Sanitasi Layak (%)"],
             row["Angka Harapan Hidup Rata-rata (Tahun)"]),
            textcoords="offset points",
            xytext=(4, 3),
            ha="left",
            fontsize=8,
            alpha=0.9
        )

    ax.set_xlabel("Persentase Akses Sanitasi Layak (%)", fontsize=11)
    ax.set_ylabel("Angka Harapan Hidup Rata-rata (Tahun)", fontsize=11)
    ax.set_title(
        "Pemetaan Klaster Kesehatan Provinsi di Indonesia\n"
        "(Berdasarkan Sanitasi Layak dan Angka Harapan Hidup)",
        fontsize=13
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Zona Prioritas", fontsize=9, title_fontsize=10)

    st.pyplot(fig)

# ============================================================
# INSIGHT
# ============================================================

if show_insight:
    st.subheader("🧠 Insight & Interpretasi")

    cluster_mean = (
        df_cluster
        .groupby("Cluster")[
            [
                "Persentase Akses Sanitasi Layak (%)",
                "Angka Harapan Hidup Rata-rata (Tahun)",
                "Persentase Keluhan Kesehatan (%)"
            ]
        ]
        .mean()
        .round(2)
    )

    st.write("Rata-rata indikator kesehatan per cluster:")
    st.dataframe(cluster_mean)

    st.write("""
    **Interpretasi Umum:**
    - Cluster dengan akses sanitasi rendah, angka harapan hidup rendah,
      dan persentase keluhan kesehatan tinggi merupakan wilayah
      **prioritas utama pembangunan kesehatan**.
    - Cluster menengah menunjukkan wilayah dengan kondisi kesehatan sedang.
    - Cluster dengan indikator kesehatan tinggi menunjukkan wilayah relatif lebih baik.
    """)
