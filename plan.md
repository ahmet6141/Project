# UCAV 3D Model Kalite Yükseltme Planı

## Mevcut Durum Analizi

Mevcut model **gövde-tüp + düz kanat** mimarisinde. Gerçek UCAV'lar (nEUROn, X-47B, Taranis) ise
**Blended Wing Body (BWB)** — gövde ve kanat tek sürekli yüzey. Temel sorunlar:

1. **Gövde dar bir tüp** (max rh=0.66m), kanat ayrı bir parça olarak yapıştırılmış
2. **Wing-body junction** sadece bir "fairing" mesh ile kapatılıyor — keskin kenar görünür
3. **Düz planform** — gerçek UCAV'larda kıvrımlı, cranked delta planform var
4. **İntake gövdeden çıkıntılı** — flush dorsal entegrasyon yok
5. **Egzoz yuvarlak boru** — slot nozzle (yassı) olmalı
6. **Sawtooth TE çok ince şerit** — gerçek serrated edge kanat kalınlığına entegre
7. **Yüzey sürekliliği C0** — C1/C2 olmalı (tangent/curvature continuous)

## Plan (13 adım)

### Adım 1: BWB Fuselage-Wing Entegrasyon Sistemi
**Dosya:** `src/fightercad/geometry/fuselage.py`

Gövde genişliğini kanat kökü ile eşleştir. Mevcut `max_diameter_m=1.2` → gövde yarı-genişliği
yalnızca 0.85m. Gerçek BWB'de gövde yarı-genişliği kanat kökü chord'unun ~%40'ı olmalı.

- `_compute_radius()` fonksiyonuna **lateral extension** ekle: kanat istasyonunda
  (x = wing_station_pct * L) gövde genişliği, inner panel span kadar genişlesin
- Yeni parametre: `body_wing_blend_ratio` (0.0=ayrık, 1.0=tam BWB)
- Gövde kesiti kanat istasyonunda elipsten airfoil profiline geçiş yapsın
- Cross-section aspect ratio kanat bölgesinde otomatik artır (geniş+yassı)

### Adım 2: Sürekli Planform Kenar Tanımı
**Dosya:** `src/fightercad/geometry/wing.py`

Mevcut planform basit bir trapez. Gerçek UCAV planformu:
- **Cranked delta**: iç panel 53° sweep, dış panel 45° sweep, kırılma noktası net
- **Sürekli LE eğrisi**: spline ile tanımlı leading edge (düz çizgi yerine)
- **Rounded wingtip**: keskin uç yerine hafif yuvarlama

- `_compute_planform()` fonksiyonuna **cranked kite** planform modu ekle
- LE sweep'i inner/outer panel için ayrı hesapla
- Wingtip chord minimum %3 taper (mevcut %8, uygun)
- Yeni parametre: `outer_panel_sweep_deg` (mevcut sadece tek sweep var)

### Adım 3: Airfoil Profil Kalitesi
**Dosya:** `src/fightercad/geometry/primitives.py`

Mevcut biconvex/diamond profiller çok basit (parabolik/lineer). Gerçek UCAV'lar
supercritical veya reflex-camber profilleri kullanır.

- `naca_4digit_symmetric` fonksiyonuna **reflex camber** desteği ekle
- Yeni airfoil tipi: `supercritical` — düz üst yüzey, kavisli alt yüzey
- Leading edge radius'u daha smooth geçiş (mevcut keskin radius uygulaması)
- TE closure: mevcut TE kalınlığı uygulaması iyileştirilsin (cusp yerine blunt)

### Adım 4: Smooth BWB Blending (C2 Süreklilik)
**Dosya:** `src/fightercad/geometry/blending.py`

Mevcut fairing basit lineer interpolasyon + sinüs bump. Gerçek BWB:
- **Cubic Hermite spline** ile gövde→kanat geçişi
- Tangent ve curvature sürekliliği
- Geniş blend zone (%20-30 semi-span)

- `build_wing_fuselage_fairing()` → **yeni sistem**: gövde kesitinden kanat
  kesitime cubic spline interpolasyon
- Her x-istasyonunda gövde yüzeyinden kanat alt/üst yüzeyine smooth geçiş
- Fillet radius yerine **blend zone width** parametresi (% of semi-span)
- Strake'i kaldır, yerine entegre BWB LE extension

### Adım 5: Flush Dorsal İntake
**Dosya:** `src/fightercad/geometry/intake.py`

Mevcut dorsal intake gövdeden çıkıntı yapıyor. Gerçek UCAV dorsal intake:
- Gövde üst yüzeyine flush entegre
- S-duct kesiti smooth geçişle daralıyor
- Lip gövde konturu ile birleşik

- `_build_dorsal()` fonksiyonunu yeniden yaz:
  - İntake açıklığı gövde üst yüzeyinde bir kesik olarak tanımla
  - S-duct kesitleri gövde konturu ile aynı başlasın
  - İç duct ayrı mesh (görünmez ama yapısal bütünlük için)
- İntake collar'ı kaldır, gövde yüzeyine entegre et

### Adım 6: Slot Nozzle (Yassı Egzoz)
**Dosya:** `src/fightercad/geometry/exhaust.py`

Mevcut egzoz yuvarlak kesit. Stealth UCAV'larda:
- **Slot nozzle**: yüksek aspect ratio dikdörtgen çıkış (W:H = 5:1 - 8:1)
- Gövde arka kesimine entegre
- Beaver tail (yassılaşan kuyruk) ile birleşik

- `build()` fonksiyonuna **slot nozzle modu** ekle
- Yeni parametre: `slot_aspect_ratio` (default 6.0)
- Giriş kesiti yuvarlak → çıkış kesiti slot (süper-elips geçiş)
- Çıkış kenarında serrated lip (opsiyonel)

### Adım 7: V-Tail Entegrasyonu
**Dosya:** `src/fightercad/geometry/stabilizer.py`

V-tail kök bölgesi gövdeye düzgün entegre olmalı:
- Mevcut fillet çok ince (0.11m genişlik)
- V-tail LE sweep gövde arka kısmı ile hizalı olmalı
- Kök kesiti gövde konturuna tangent olmalı

- Stabilizer kök chord'u gövde genişliğine göre otomatik ölçekle
- Kök fillet genişliğini artır (chord'un %15'i minimum)
- V-tail'in gövde üstünde başlama pozisyonunu gövde konturu ile hizala

### Adım 8: Mesh Kalite İyileştirme
**Dosya:** `src/fightercad/aircraft.py`, `src/fightercad/geometry/wing.py`

- Section sayısını artır (wing: 20→32 per panel, fuselage: 80→120)
- Airfoil nokta sayısını artır (150→200)
- Leading edge bölgesinde daha yoğun nokta dağılımı (cosine clustering)
- Trailing edge bölgesinde keskin mesh (thin triangles at TE → dedicated TE closure)

### Adım 9: Normal Tutarlılık & Yüzey Kalitesi
**Dosya:** `src/fightercad/aircraft.py`

Mevcut centroid-based normal fix yetersiz (~%40 hala ters):
- **Component-level winding fix**: her bileşeni ayrı ayrı düzelt
- Gövde, kanat, intake mesh'lerinin her birinin kendi centroid'ini kullan
- Edge-propagation ile tutarlı winding (BFS from seed face)

### Adım 10: UCAV Parametrelerini Güncelle
**Dosya:** `configs/ucav_delta.yaml`, `src/fightercad/parameters.py`

Yeni parametreleri ekle ve UCAV preset'i güncelle:
- `body_wing_blend_ratio: 0.85` (yüksek BWB entegrasyonu)
- `outer_panel_sweep_deg: 45.0`
- `slot_aspect_ratio: 6.0`
- `nozzle_shape: "slot"` (yeni)
- Inner panel span: %29 → planform ile uyumlu
- Section/airfoil resolution artır

### Adım 11: STL Export Kalite
**Dosya:** `src/fightercad/export/step_export.py`

- STL export'ta normal vektörleri doğru yaz (mevcut face normal hesaplama)
- Binary STL desteği (daha küçük dosya)
- Export log: face count, file size, bounding box

### Adım 12: Görsel Doğrulama Aracı
**Dosya:** `src/fightercad/visualization/plot2d.py`

- Planform görünümü (top view) çizimi
- Cross-section evolution çizimi (5-6 istasyonda kesit)
- Side view silhouette

### Adım 13: Testleri Güncelle
**Dosya:** `tests/`

- BWB blending testleri (yüzey sürekliliği kontrolü)
- Slot nozzle testleri
- Planform alan doğrulaması
- Mesh kalite threshold testleri (flipped normal < %5)
