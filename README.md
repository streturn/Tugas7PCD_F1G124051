# Tugas7PCD_F1G124051

Program ini memproses gambar ijazah, membaca nomor ijazah dengan OCR, dan
mendeteksi tanda tangan.

## Mengirim program ke GitHub

Jalankan perintah berikut di PowerShell pada folder proyek:

```powershell
git add README.md main.py
git commit -m "Add instructions to run the program"
git push origin main
```

Periksa `git status` sebelum commit. Jika ingin menyediakan gambar contoh agar
program bisa langsung dicoba setelah di-clone, tambahkan gambar yang aman
untuk dibagikan:

```powershell
git add Nomor_Ijazah/
git commit -m "Add sample diploma images"
git push origin main
```

Jangan unggah gambar yang berisi data pribadi atau tidak memiliki izin untuk
dibagikan. Pengguna lain dapat memasukkan gambar mereka sendiri ke folder
`Nomor_Ijazah`.

## Menyiapkan laptop lain (Windows)

### 1. Pasang aplikasi yang dibutuhkan

- Pasang Git.
- Pasang Python 3.10 atau lebih baru, lalu aktifkan opsi **Add Python to PATH**
  saat instalasi.
- Pasang Tesseract OCR untuk Windows. Program mengharapkan executable di
  `C:\Program Files\Tesseract-OCR\tesseract.exe`. Jika memasangnya di lokasi
  lain, ubah `pytesseract.pytesseract.tesseract_cmd` di `main.py` agar menunjuk
  ke `tesseract.exe` di laptop tersebut.

### 2. Unduh proyek

```powershell
git clone https://github.com/streturn/Tugas7PCD_F1G124051.git
cd Tugas7PCD_F1G124051
```

### 3. Buat lingkungan Python dan pasang pustaka

Jalankan perintah berikut dari folder proyek:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install opencv-python pytesseract pandas numpy
```

### 4. Siapkan gambar dan jalankan

Taruh gambar ijazah di folder `Nomor_Ijazah`. Jika folder belum tersedia
(misalnya gambar contoh tidak disertakan di GitHub), buat folder tersebut:

```powershell
New-Item -ItemType Directory -Force Nomor_Ijazah
```

Format gambar yang didukung: JPG, JPEG, PNG, BMP, TIF, dan TIFF. Jalankan
program dengan:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Hasil

Program membuat folder `hasil` secara otomatis. Ringkasan verifikasi disimpan
di `hasil/hasil_verifikasi.csv`, perbandingan metode OCR di
`hasil/perbandingan_metode.csv`, dan gambar hasil pemrosesan di subfolder
`hasil`.
