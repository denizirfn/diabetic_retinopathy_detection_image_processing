Proje Hakkında
Bu proje, diyabetik retinopati (DR) hastalığının tespiti ve segmentasyonu üzerine yapılmıştır. Projenin ikinci aşamasında optik disk, kan damarları, mikroanevrizmalar ve eksüdaların segmentasyonu gerçekleştirilmiş, ardından bu segmentasyon sonuçlarından öznitelik çıkarımı yapılarak makine öğrenmesi modelleriyle hastalık sınıflandırması yapılmıştır.

Kullanılan Yöntemler ve İş Akışları
1. Optik Disk Segmentasyonu
Retina görüntüleri sabit boyuta yeniden boyutlandırılır.

Optik diskin daha belirgin olduğu kırmızı kanal seçilir.

Kırmızı kanal üzerinde CLAHE ile kontrast artırılır.

Piksel değerleri normalize edilir.

K-means algoritmasıyla en parlak küme optik disk olarak belirlenir.

Morfolojik açma ve kapama işlemleriyle gürültüler temizlenir, maskenin bütünlüğü sağlanır.

En büyük kontur çevresinde dairesel maske oluşturulur ve Gaussian bulanıklığı uygulanır.

Maske genişletilerek optik disk tamamen kapsanır ve görüntüden çıkarılır.

2. Kan Damarlarının Çıkarılması
Görüntüler sabit çözünürlüğe ölçeklendirilir.

Yeşil kanal seçilir ve CLAHE ile kontrast artırılır.

İki farklı işlem hattı ile damarlar belirginleştirilir (medyan bulanıklaştırma ve çıkarma).

İki işlem hattının sonuçları bit düzeyinde birleştirilir.

Sonuç orijinal çözünürlüğe yeniden boyutlandırılır.

3. Mikroanevrizma Segmentasyonu
Görüntü RGB kanallarına ayrılır, mikroanevrizmaların belirgin olduğu yeşil kanal ters çevrilir.

CLAHE ile lokal kontrast artırılır.

Gamma ayarlarıyla parlaklık ve kontrast iyileştirilir.

Gaussian filtresiyle gürültü azaltılır.

Otsu eşikleme ile ikili görüntü elde edilir.

Top Hat dönüşümü ve morfolojik açma ile mikroanevrizmalar izole edilir.

4. Eksüda Segmentasyonu
Görüntüler standart boyuta yeniden boyutlandırılır.

Yeşil kanal seçilir.

K-means algoritması ile büyük eksüdalar segmentlenir.

Canny kenar tespiti ve morfolojik işlemlerle küçük eksüdalar belirginleştirilir.

İki maske birleştirilerek nihai eksüda maskesi oluşturulur.

Optik disk maskesi kullanılarak karışıklık önlenir.

Öznitelik Çıkarımı
Segmentasyon sonuçlarından aşağıdaki öznitelikler çıkarılmıştır:

Alan (beyaz piksel sayısı)

Ortalama piksel değeri

Piksel yoğunluğunun standart sapması

Doluluk oranı (segmentin görüntüye oranı)

Ayrıca kan damarlarının uzunluğu ve kıvrımlılığı, mikroanevrizmaların kapladığı alan ve kenar yoğunluğu gibi özellikler hesaplanmıştır. Bu öznitelikler, hastalık tanısı için makine öğrenmesi modellerine girdi olarak verilmiştir.

Veri Seti
Eğitim için 100 hasta ve 100 normal retina görüntüsü,

Test için 50 hasta ve 50 normal retina görüntüsü kullanılmıştır.

Segmentasyon sonuçları kaydedilmiş ve öznitelikler çıkarılmıştır.

Sınıflandırma ve Değerlendirme
Öznitelikler Standard Scaler ile normalize edilmiştir.

Random Forest, Support Vector Machines (SVM) gibi çeşitli makine öğrenmesi algoritmaları kullanılarak sınıflandırma yapılmıştır.

En yüksek başarı gösteren model seçilmiştir.

Kullanım
Retina görüntülerini sabit boyuta yeniden boyutlandırın.

Optik disk segmentasyonunu gerçekleştirin.

Kan damarlarını çıkarın.

Mikroanevrizma ve eksüda segmentasyonlarını uygulayın.

Segmentasyon sonuçlarından öznitelikleri çıkarın.

Öznitelikleri modelinize vererek sınıflandırma yapın.

İletişim
Herhangi bir sorunuz veya geri bildiriminiz için lütfen iletişime geçin.
