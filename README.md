# Problemli İnternet Kullanımının Tahmini – Yapay Zeka Destekli Analiz

## 📌 Proje Özeti

Bu proje, çocuk ve gençlerde giderek artan **Problemli İnternet Kullanımı (PIU)** sorununu fiziksel, biyolojik ve demografik verilerle analiz ederek yapay zeka destekli bir sınıflandırma modeli geliştirmeyi amaçlamaktadır. Geleneksel yöntemlerin ötesine geçilerek, akıllı saat verileriyle zenginleştirilmiş kapsamlı veri seti üzerinden makine öğrenmesi algoritmaları eğitilmiş ve PIU tahminleri yapılmıştır.

---

## 🎯 Proje Amacı ve Yöntemi

Bu projede;

- Problemli internet kullanımını sadece psikolojik değil, fiziksel ve biyolojik açıdan da ele almak,
- Bu verilerle **makine öğrenmesi algoritmaları eğiterek** sınıflandırma yapmak,
- Bu sayede **erken teşhis** ve **önleyici analiz** modelleri geliştirmek amaçlanmıştır.

Kullanılan yöntemler:

-  **Özellik çıkarımı** için LSTM ve istatistiksel zaman serisi analizleri
-  **Modelleme**: Random Forest, LightGBM, XGBoost gibi makine öğrenmesi algoritmaları
-  **Hiperparametre optimizasyonu**
-  PCIAT toplam puanlarının sürekli değişkenden sınıfa dönüştürülmesi
-  **Ensemble yöntemleri** ile modellerin birleştirilmesi

---

## 📁 Kullanılan Veri Seti

Bu çalışmada, **Healthy Brain Network (HBN)** tarafından sağlanan ve **Kaggle** platformunda yayınlanan halka açık veri seti kullanılmıştır.

🔗 **Kaggle Veri Seti Linki:**  
[https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data]

Kaggle platformunda veriler, `Child Mind Institute` klasörü altında farklı alt klasör ve dosyalara ayrılmıştır.

### 📂 Dizin Yapısı

- `train.csv`: 5-22 yaş arası 3960 kişinin fiziksel, biyolojik ve anket verileri
- `series_train.parquet`: 996 kişinin akıllı saat zaman serisi verileri
- `test.csv`: 20 kişilik test verisi (saat verisi hariç)
- `series_test.parquet`: 2 kişinin saat verileri

Veri seti genel olarak yüksek oranda eksik (NaN) değer içermektedir.

---

## 🔍 Kullanılan Veri Setinin Özellikleri

### `train.csv`

- Toplam 82 sütun
- 22 sütun: **PCIAT anketi** (hedef değişken olan SII’yi etiketlemek için kullanılır)
- Diğer sütunlar:
  - Yaş, cinsiyet, sezon gibi demografik bilgiler
  - Fiziksel ölçümler, FitnessGram Vitals, Treadmill, Child testleri
  - Uyku bozukluğu puanı (Sleep Disturbance Scale - SDS)
  - Bio-electric Impedance Analysis (BIA) verileri
  - Ebeveyn-çocuk İnternet Bağımlılığı Testi gibi sağlık anketleri

### `series_train.parquet`

- 996 kişinin kimlik numarasına göre klasör yapısı
- Her klasör içinde `part-0.parquet` dosyası
- 12 özellik:
  - X, Y, Z koordinatları
  - ENMO (hareket), anglez (kol açısı), light (ışık seviyesi)
  - non_wear_flag (saat takılı mı?), battery_voltage
  - time_of_day, weekday, quarter
  - relative_date_PCIAT (PCIAT tarihine göre geçen gün sayısı)

### `test.csv`

- 20 kişiye ait 59 sütun
- `train.csv` ile aynı özellikleri içerir ancak PCIAT ve SII verileri yoktur

### `series_test.parquet`

- 2 kişiye ait akıllı saat verileri
- `series_train.parquet` ile aynı yapıda

---

## 🧮 Proje Başarım Metriği

Bu projede model başarımı, **Quadratic Weighted Kappa (QWK)** metriği ile değerlendirilmiştir. Bu metrik, gerçek ve tahmin edilen etiketler arasındaki uyumu ölçer.

### QWK Hesaplama Süreci

1. **O Matrisi (Gözlem):** Gerçek ve tahmin edilen sınıfların frekansı
2. **W Matrisi (Ağırlık):**  
   \[
   W_{i,j} = \frac{(i-j)^2}{(N-1)^2}
   \]
3. **E Matrisi (Beklenen):** Etiketlerin bağımsız olduğu varsayımıyla tahmini dağılım

Bu matrisler ile hesaplanan QWK, 0 (rastgele tahmin) ile 1 (tam uyum) arasında değer alabilir. Negatif skorlar, tahminin rastgele uyumdan daha kötü olduğunu gösterir.

---


## 📊 Elde Edilen Bulgular

- Uyku durumu ve internet kullanım saatleriyle ilgili değişkenler tahmin başarısını artırdı.
- Zaman serisi verilerinden çıkarılan özellikler model skorlarını anlamlı şekilde iyileştirdi.
- **PCIAT_Total** tahmininin ardından sınıflandırma yapılması başarıyı artıran bir strateji oldu.
- **En iyi QWK skoru: 0.486** olarak elde edildi. Yarışma liderinin skoru olan 0.482'nin üzerine çıkıldı.

---

##  Katkı ve Özgünlük

- Bu çalışma, yapay zeka ve sağlık alanlarını birleştiren özgün bir örnektir.
- Literatürde genellikle sadece istatistiksel analizlere odaklanılmıştır.
- Projede ilk defa fiziksel aktivite ve biyolojik veriler ile **Yapay Zeka Tabanlı** PIU sınıflandırması yapılmıştır.




