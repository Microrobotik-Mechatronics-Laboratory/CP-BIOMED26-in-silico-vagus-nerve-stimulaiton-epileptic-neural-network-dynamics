# Nerve Frequency Study — Periferik Sinir Stimülasyonu Frekans Çalışması

## Study Scope

### Basal Ganglians

### Neuron Model

### Plasticity Model

--

## Project Structure



--

## How to start

### uv Kurulumu

Bu proje bağımlılık ve sanal ortam yönetimi için [uv](https://docs.astral.sh/uv/)
kullanır. Aşağıdaki adımlarla işletim sisteminize göre kurabilirsiniz
(detaylar: [uv kurulum dokümantasyonu](https://docs.astral.sh/uv/getting-started/installation/)).

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Kurulumdan sonra terminali yeniden başlatın ve doğrulayın:
```bash
uv --version
```

---

### Dependices install

```bash
# 1. uv ile sanal ortam oluştur
uv venv --python 3.11

# 2. Bağımlılıkları yükle (numpy<2.0 kısıtı kritik — NEURON numpy 2.x ile uyumsuz)
uv pip install --python .venv/bin/python "numpy>=1.24,<2.0" pandas matplotlib brian2 neuron

# 3. Geliştirme bağımlılıkları (test + tip kontrolü, isteğe bağlı)
uv pip install --python .venv/bin/python "pytest>=7.0" "mypy>=1.8"

# 4. MRG / Sundt NEURON mekanizmalarını derle
cd mechanisms && ../.venv/bin/nrnivmodl && cd ..
```
--

### Test ve Tip Kontrolü

```bash
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m mypy config src visualization main.py tests
```

---

## Kullanım

```bash
# Varsayılan parametrelerle hızlı test
python main.py

# Özel frekans listesi
python main.py --frequencies 1 10 50 100 500

# Farklı genlik ve fiber çapı
python main.py --frequencies 20 100 200 --amp 5.0 --fiber-diam 12.8

# Sonuçları CSV'ye kaydet
python main.py --output results.csv

# Grafikleri dosyaya kaydet (ekranda göstermeden)
python main.py --save-plots outputs/ --no-plot

# Tam frekans aralığı (yavaş — ~15-30 dakika)
python main.py --full-range --amp 3.0

# Tüm seçenekler
python main.py --help
```

### CLI Argümanları

| Argüman | Varsayılan | Açıklama |
|:---|:---:|:---|

--

## Contrubution and Feedback

### Commit Formatı

```
tip(kapsam): kısa açıklama

Örnekler:
  fix(models): fizyolojik sıcaklık ayarı — h.celsius = 37°C
  feat(pipeline): paralel frekans sweep — multiprocessing
  docs(readme): kurulum adımları güncellendi
  refactor(config): magic number'lar defaults.py'e taşındı
  test(analysis): point_source_potential sayısal doğrulama
  chore(gitignore): derleme çıktıları eklendi
```

| Tip | Kullanım |
|:---|:---|
| `fix` | Hata düzeltme |
| `feat` | Yeni özellik |
| `docs` | Yalnızca dokümantasyon |
| `refactor` | Davranış değiştirmeyen yeniden yapılandırma |
| `test` | Test ekleme/güncelleme |
| `chore` | Araç/yapılandırma değişiklikleri |

---

## Python Kodlama Standartları

### Stil
- **PEP 8** — 4 boşluk girinti, maks 100 karakter satır
- Değişken/fonksiyon: `snake_case` | Sınıf: `PascalCase` | Sabit: `UPPER_SNAKE_CASE`
- Magic number yok — tüm sabitler `config/defaults.py`'de

### Docstring Formatı

```python
def fonksiyon(param: float, opsiyonel: int = 10) -> list:
    """
    Tek cümle özet.

    Parametreler
    ------------
    param : float
        Açıklama.
    opsiyonel : int, optional
        Açıklama. Varsayılan 10.

    Döndürür
    --------
    list
        Açıklama.
    """
```

### Push Request Standart




info mail: (ege.mrml@gmail.com)[mailto:ege.mrml@gmail.com] or (musaadereli@gmail.com)[mailto:musaadereli@gmail.com]