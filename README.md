# In Silico VNS Modulation of Epileptic Hippocampal Dynamics (BIOMED 2026)

Bu çalışma, medial temporal lob epilepsisindeki (mTLE) nöropatolojik hipokampal ağ dinamiklerinin (EC–DG–CA3–CA1), 30 farklı Vagus Sinir Stimülasyonu (VNS) parametre konfigürasyonuna verdiği elektrofizyolojik yanıtları ve nöbet baskılama oranını (SSR) çok ölçekli biyofiziksel hesaplamalı modelleme ile incelemektedir.

---

## 1. Çalışma Kapsamı ve Biyofiziksel Mimari

Model çok ölçekli (multiscale) hiyerarşik bir zincir üzerine kurulmuştur:

```
[Point-Source Cuff Elektrot] 
            │ (Ekstrasellüler alan: 1-3 mA, 1 Hz - 10 kHz, DC & Biphasic)
            ▼
[NEURON: MRG Akson Modeli] (15 Ranvier nodu, miyelinli lif, dt = 0.005 ms)
            │ (Distal aksiyon potansiyelleri)
            ▼
[Brian2: Jittered Fiber Bundle] (20 lifli refrakter akson demeti)
            │ (Depresan Tsodyks-Markram sinapsları)
            ▼
[Brian2: NTS Röle Çekirdeği] (30 LIF nöronu, Miles 1986 / Beaumont 2017 kalibrasyonu)
            │ (Fasilitatör Tsodyks-Markram projeksiyonları)
            ▼
[Epileptik Hipokampal Mikrodevre] (~480 nöron: EC - DG - CA3 - CA1)
  ├── 1. Hipokampal Skleroz (scl): Piramidal hücre kaybı, internöron korunumu
  ├── 2. Mossy Fiber Sprouting (spr): DG granül hücreleri arası anormal rekürrent eksitasyon
  ├── 3. Bozulmuş Klorür Dinamiği (tau_Cl): KCC2 yetmezliği -> depolarizan GABA
  └── 4. Tri-sinaptik döngü ve rekürrent CA3 kolateralleri
            │
            ▼
[LFP Simülasyonu & Spektral Biyobelirteçler]
  ├── Dipolar transmembran sinaptik akım toplamı (Aussel et al. 2022)
  ├── Butterworth bant geçiren filtreleme (0.15 - 480 Hz, 1024 Hz yeniden örnekleme)
  └── Welch PSD, Teta Bandı (4-8 Hz) Gücü, Senkronizasyon İndeksi ve SSR
```

---

## 2. Kurulum ve Ortam Hazırlığı

Proje bağımlılık ve ortam yönetimi için `uv` kullanır:

```bash
# 1. Sanal ortamı oluştur (Python 3.11)
uv venv --python 3.11

# 2. Bağımlılıkları yükle (NEURON uyumu için numpy < 2.0 zorunludur)
uv pip install --python .venv/bin/python "numpy>=1.24,<2.0" pandas matplotlib scipy brian2 neuron pytest

# 3. Derlenmiş NEURON mekanizmalarını kontrol et
uv run python -c "import neuron; from src.models._neuron_mechanisms import load_project_mechanisms; load_project_mechanisms(); print('NEURON mechanisms ready!')"
```

---

## 3. Testlerin Koşturulması

Tüm modüller, biyofiziksel denklemler ve entegrasyon seviyeleri otomatik birim testlerle kapsanmıştır:

```bash
uv run pytest tests/ -v
```

---

## 4. Kullanım ve CLI Komutları

### 4.1. Tekil Simülasyon Denemesi (Tek Koşul)

```bash
# Epileptik ağ üzerinde 20 Hz, 2.0 mA bifazik kare dalga uyarımı ve LFP/Raster grafiği üretimi
uv run python main.py --condition epileptic --freq 20.0 --amp 2.0 --waveform rectangular --duration 500 --plot

# Sağlıklı kontrol ağında uyarım
uv run python main.py --condition healthy --freq 20.0 --amp 2.0 --duration 500 --plot
```

### 4.2. 30 Parametre Konfigürasyonlu Sistematik Sweep

30 VNS koşulu (2 dalga formu $\times$ 3 genlik $\times$ 5 frekans) $\times$ 5 tekrar = 150 deneme + baseline kontrolü:

```bash
# Tam sweep çalıştır ve sonuç tablosu ile yayın grafiklerini otomatik üret
uv run python main.py --run-sweep --repeats 5 --duration 500 --output-dir output
```

Üretilen Çıktılar:
- `output/results/vns_epilepsy_sweep.csv`: Tüm denemelere ait bölgesel ateşleme hızları, teta gücü, senkronizasyon ve SSR metrikleri.
- `output/figures/trial_epileptic_rectangular_20.0Hz.png`: Bipolar LFP ve hipokampal alt bölgeler raster plot grafiği.
- `output/figures/frequency_response_biphasic_2ma.png`: Frekansa bağlı bölgesel ateşleme hızı ve teta gücü yanıt eğrileri.
- `output/figures/seizure_suppression_heatmap.png`: Genlik vs Frekans vs SSR (Nöbet Bastırma Oranı) ısı haritası.

---

## 5. Referanslar

1. **Aussel A. et al.** (2022). *Cell to network computational model of the epileptic human hippocampus suggests specific roles of network and channel dysfunctions in the ictal and interictal oscillations.* J Comput Neurosci, 50, 519–535.
2. **Ahn S. et al.** (2016). *A computational model of the human hippocampal formation.* J Comput Neurosci, 41, 207–223.
3. **Dereli M. et al.** (2026). *Frequency-dependent forebrain recruitment via vagus nerve stimulation.* 24th National Neuroscience Congress (USK 2026), p. 326.
4. **Miles R.** (1986). *Frequency dependence of synaptic transmission.* J Neurophysiol, 55(5), 1076–1090.
5. **Beaumont E. et al.** (2017). *Vagus nerve stimulation parameter titration.* Am J Physiol Heart Circ Physiol, 313(2), H354–H367.
