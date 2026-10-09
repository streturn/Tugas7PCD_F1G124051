import cv2
import re
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

import pandas as pd
import numpy as np

from pathlib import Path


# 1. Pengaturan

FOLDER_NOMOR_IJAZAH = Path("Nomor_Ijazah")
FOLDER_HASIL = Path("hasil")

FOLDER_NOMOR_IJAZAH.mkdir(exist_ok=True)
FOLDER_HASIL.mkdir(exist_ok=True)

# Nomor ijazah acuan dari gambar yang diberikan
GROUND_TRUTH = "571012022000056"

# Area tanda tangan dalam persen setelah gambar diluruskan.
AREA_TTD = (35, 18, 96, 45)

# Ambang awal deteksi tanda tangan
AMBANG_TTD = 0.005

EKSTENSI_GAMBAR = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def luruskan_gambar(gambar):
    """
    Memutar gambar ijazah yang orientasinya menyamping.
    Untuk contoh gambar yang dikirim, gunakan rotasi 90 derajat.
    """

    # Putar 90 derajat searah jarum jam
    return cv2.rotate(gambar, cv2.ROTATE_90_CLOCKWISE)


def batas_area_persen(gambar, area):
    tinggi, lebar = gambar.shape[:2]
    x1, y1, x2, y2 = area

    return (
        max(0, min(int(lebar * x1 / 100), lebar - 1)),
        max(0, min(int(tinggi * y1 / 100), tinggi - 1)),
        max(1, min(int(lebar * x2 / 100), lebar)),
        max(1, min(int(tinggi * y2 / 100), tinggi))
    )


# 3. Enhancement gambar
def buat_metode(gambar):
    # Pastikan input berupa gambar grayscale
    if len(gambar.shape) == 3:
        abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    else:
        abu = gambar.copy()

    # Mengurangi bintik kecil
    denoise = cv2.fastNlMeansDenoising(abu, None, 10, 7, 21)

    # Meningkatkan kontras lokal
    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(denoise)

    # Mempertajam teks
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.5)
    sharpen = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)

    # Threshold dengan blok lebih besar untuk mengurangi
    # pengaruh bintik-bintik kecil pada latar
    adaptive = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15
    )

    # Otsu sebagai metode pembanding
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "Grayscale": abu,
        "Denoise + CLAHE": clahe,
        "Sharpen": sharpen,
        "Adaptive Threshold": adaptive,
        "Otsu Threshold": otsu
    }

# 4. OCR nomor ijazah
def baca_ocr(gambar):
    """
    Membaca nomor ijazah dari gambar.
    """

    # Memperbesar gambar agar teks lebih mudah dibaca
    gambar_besar = cv2.resize(
        gambar, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )
    konfigurasi = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-/"
    teks = pytesseract.image_to_string(gambar_besar, config=konfigurasi)

    # Menghapus karakter selain angka, tanda hubung, dan garis miring
    teks = re.sub(r"[^0-9\-/]", "", teks)

    return teks


# 5. Menghitung CER
def hitung_cer(acuan, hasil):
    """
    CER = (S + D + I) / N

    S = karakter salah
    D = karakter hilang
    I = karakter tambahan
    N = jumlah karakter nomor acuan
    """

    # Spasi diabaikan, tanda hubung tetap dihitung
    acuan = re.sub(r"\s+", "", str(acuan))
    hasil = re.sub(r"\s+", "", str(hasil))

    if len(acuan) == 0:
        return 0.0 if len(hasil) == 0 else 1.0

    n = len(acuan)
    m = len(hasil)

    # Matriks jarak edit
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            biaya = 0 if acuan[i - 1] == hasil[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + biaya
            )

    return dp[n][m] / n


# 6. Deteksi tanda tangan
def deteksi_ttd(gambar):
    # 1. Grayscale seluruh gambar
    if len(gambar.shape) == 3:
        abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    else:
        abu = gambar.copy()

    abu = cv2.GaussianBlur(abu, (3, 3), 0)

    # 3. Thresholding Global
    _, mask_global = cv2.threshold(abu, 150, 255, cv2.THRESH_BINARY_INV)

    # 4. Thresholding Otsu
    _, mask_otsu = cv2.threshold(
        abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5. Morphology pada hasil thresholding Global
    kernel = np.ones((2, 2), np.uint8)

    mask_morph = cv2.morphologyEx(mask_global, cv2.MORPH_OPEN, kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)

    # Hitung foreground hanya pada zona tanda tangan, tanpa crop gambar.
    x1, y1, x2, y2 = batas_area_persen(gambar, AREA_TTD)
    area_mask = np.zeros_like(mask_morph)
    cv2.rectangle(
        area_mask, (x1, y1), (x2 - 1, y2 - 1), 255, thickness=cv2.FILLED
    )
    piksel_foreground = cv2.countNonZero(
        cv2.bitwise_and(mask_morph, area_mask)
    )
    total_piksel = cv2.countNonZero(area_mask)

    persentase_foreground = piksel_foreground / max(total_piksel, 1) * 100

    # 7. Prediksi tanda tangan
    if persentase_foreground >= AMBANG_TTD * 100:
        status = "PRESENT"
    else:
        status = "ABSENT"

    return (
        status, persentase_foreground, mask_global, mask_otsu, mask_morph
    )

# 7. Simulasi tanpa tanda tangan
def buat_simulasi_tanpa_ttd(gambar):
    """
    Membuat salinan gambar dengan area tanda tangan diputihkan.
    Gambar asli tidak diubah.
    """

    simulasi = gambar.copy()

    x1, y1, x2, y2 = batas_area_persen(simulasi, AREA_TTD)

    simulasi[y1:y2, x1:x2] = 255

    return simulasi


# 8. Memproses satu gambar
def evaluasi_gambar(gambar, nama_file, jenis):
    # Menjalankan OCR pada seluruh gambar
    metode = buat_metode(gambar)
    hasil_metode = []

    for nama_metode, gambar_metode in metode.items():
        nomor = baca_ocr(gambar_metode)
        cer = hitung_cer(GROUND_TRUTH, nomor)

        hasil_metode.append({
            "Nama file": nama_file,
            "Jenis gambar": jenis,
            "Metode": nama_metode,
            "Hasil OCR": nomor,
            "CER (%)": round(cer * 100, 2)
        })

    # Memilih metode dengan CER terendah
    terbaik = min(hasil_metode, key=lambda item: item["CER (%)"])

    # Deteksi tanda tangan pada seluruh gambar
    (
        status_ttd, persentase_foreground, mask_global, mask_otsu, mask_morph
    ) = deteksi_ttd(gambar)

    # Folder output untuk setiap gambar
    folder_output = FOLDER_HASIL / Path(nama_file).stem
    folder_output.mkdir(parents=True, exist_ok=True)

    # Simpan hasil deteksi tanda tangan pada seluruh gambar
    cv2.imwrite(str(folder_output / "threshold_global.jpg"), mask_global)
    cv2.imwrite(str(folder_output / "threshold_otsu.jpg"), mask_otsu)
    cv2.imwrite(str(folder_output / "morphology.jpg"), mask_morph)

    # Simpan gambar lurus
    cv2.imwrite(str(folder_output / "gambar_lurus.jpg"), gambar)

    # Simpan semua hasil enhancement
    for nama_metode, gambar_metode in metode.items():
        nama_aman = re.sub(r"\W+", "_", nama_metode.lower())
        cv2.imwrite(str(folder_output / f"{nama_aman}.jpg"), gambar_metode)

    # Ringkasan untuk CSV
    ringkasan = {
        "Nama file": nama_file,
        "Jenis gambar": jenis,
        "Hasil OCR": terbaik["Hasil OCR"],
        "Status TTD": status_ttd,
        "Persentase foreground (%)": round(persentase_foreground, 2),
        "Nilai CER (%)": terbaik["CER (%)"],
        "Metode terbaik": terbaik["Metode"]
    }

    return ringkasan, hasil_metode

# 9. Memproses semua gambar
def main():
    daftar_gambar = sorted(
        file for file in FOLDER_NOMOR_IJAZAH.iterdir()
        if file.is_file()
        and file.suffix.lower() in EKSTENSI_GAMBAR
    )

    if not daftar_gambar:
        print(
            f"Folder {FOLDER_NOMOR_IJAZAH} kosong. "
            "Masukkan gambar ijazah terlebih dahulu."
        )
        return

    semua_ringkasan = []
    semua_perbandingan = []

    for file_gambar in daftar_gambar:
        gambar = cv2.imread(str(file_gambar))

        if gambar is None:
            print("Gagal membaca:", file_gambar.name)
            continue

        # LANGKAH PENTING: luruskan gambar terlebih dahulu
        gambar = luruskan_gambar(gambar)

        # Simpan hasil gambar yang sudah diluruskan
        cv2.imwrite(str(FOLDER_HASIL / f"{file_gambar.stem}_lurus.jpg"), gambar)

        # Uji gambar asli
        ringkasan, perbandingan = evaluasi_gambar(gambar, file_gambar.name, "ASLI")

        semua_ringkasan.append(ringkasan)
        semua_perbandingan.extend(perbandingan)

        # Buat gambar simulasi tanpa tanda tangan
        simulasi = buat_simulasi_tanpa_ttd(gambar)

        nama_simulasi = f"{file_gambar.stem}_tanpa_ttd.png"

        folder_simulasi = FOLDER_HASIL / "simulasi_tanpa_ttd"
        folder_simulasi.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(folder_simulasi / nama_simulasi), simulasi)

        # Uji gambar simulasi menggunakan pipeline yang sama
        ringkasan_sim, perbandingan_sim = evaluasi_gambar(
            simulasi, nama_simulasi, "SIMULASI TANPA TTD"
        )

        semua_ringkasan.append(ringkasan_sim)
        semua_perbandingan.extend(perbandingan_sim)

        print("Selesai memproses:", file_gambar.name)

    # Menyimpan tabel utama
    if semua_ringkasan:
        tabel = pd.DataFrame(semua_ringkasan)

        tabel.to_csv(FOLDER_HASIL / "hasil_verifikasi.csv", index=False, encoding="utf-8-sig")

        print("\n=== TABEL HASIL VERIFIKASI ===")
        print(tabel.to_string(index=False))

    # Menyimpan perbandingan seluruh metode OCR
    if semua_perbandingan:
        tabel_metode = pd.DataFrame(semua_perbandingan)

        tabel_metode.to_csv(
            FOLDER_HASIL / "perbandingan_metode.csv", index=False, encoding="utf-8-sig"
        )

        print("\n=== PERBANDINGAN METODE ===")
        print(tabel_metode.to_string(index=False))

    print("\nSemua hasil disimpan di folder hasil.")


if __name__ == "__main__":
    main()