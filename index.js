require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { google } = require('googleapis');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const sharp = require('sharp');
const fs = require('fs');
const { createClient } = require('@supabase/supabase-js');

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Setup Multer for handling file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// --- INIT GOOGLE GEMINI AI (ROTATION SYSTEM) ---
// Baca semua API key yang dipisahkan koma
const geminiKeys = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
  .split(',')
  .map(k => k.trim())
  .filter(Boolean);

let currentKeyIndex = 0;

// --- SETUP SUPABASE ---
const WebSocket = require('ws'); // Fix untuk Node.js < 22
global.WebSocket = WebSocket;

const supabaseUrl = process.env.SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_KEY || '';
// Inisialisasi Supabase hanya jika URL dan Key tersedia di .env
const supabase = (supabaseUrl && supabaseKey) ? createClient(supabaseUrl, supabaseKey) : null;

// --- SETUP GMAIL OAUTH2 ---
const oauth2Client = new google.auth.OAuth2(
  process.env.GMAIL_CLIENT_ID,
  process.env.GMAIL_CLIENT_SECRET,
  process.env.GMAIL_REDIRECT_URI
);

if (process.env.GMAIL_REFRESH_TOKEN) {
  oauth2Client.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
}

const gmail = google.gmail({ version: 'v1', auth: oauth2Client });

// ==========================================
// ENDPOINTS
// ==========================================

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'STB Backend API is running with Gemini Vision' });
});

// 1. Endpoint to process STB image using Gemini Vision AI
app.post('/api/process-stb', upload.single('image'), async (req, res) => {
  console.log('\n--- New STB Process Request ---');
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image uploaded' });
    }

    console.log(`Original File: ${req.file.originalname || 'capture'} (${(req.file.size / 1024).toFixed(1)} KB)`);
    console.log('Compressing image for faster processing...');

    // KOMPRESI & PENAJAMAN GAMBAR (KHUSUS DOT MATRIX KERTAS PANJANG):
    // Kita naikkan resolusi ke 1600px agar huruf kecil tidak blur saat difoto dari jauh.
    const compressedBuffer = await sharp(req.file.buffer)
      .resize({ width: 1600, withoutEnlargement: true })
      .grayscale() // Buang warna
      .normalize() // Auto-contrast: kertas jadi putih bersih, tinta pudar jadi hitam pekat
      .sharpen()   // Penajaman ekstrem agar titik-titik dot matrix tidak menyatu/blur
      .jpeg({ quality: 85 }) // Kualitas sedikit dinaikkan
      .toBuffer();

    console.log(`Compressed File Size: ${(compressedBuffer.length / 1024).toFixed(1)} KB`);
    console.log('Sending optimized image to Gemini Vision AI...');

    // Konversi image buffer yang sudah dikompres ke format yang dimengerti Gemini
    const imageBase64 = compressedBuffer.toString('base64');
    const mimeType = 'image/jpeg';

    // Prompt yang sangat spesifik dan dilengkapi daftar validasi agar Gemini tidak salah baca
    const prompt = `Kamu adalah sistem OCR canggih untuk mengekstrak data dari dokumen STB (Serah Terima Barang) kapal.
Dokumen ini mungkin dicetak dengan PRINTER DOT MATRIX atau ditulis tangan.
- Waspadai angka pudar/putus: "5" sering terbaca "6", "8" SANGAT SERING terbaca "6" atau "B", "1" terbaca "7".

Tugasmu HANYA mencari TIGA informasi: "Nomor Permintaan", "Nama Kapal", dan "Nama Barang".

1. NAMA KAPAL:
Cari tulisan yang menunjukkan nama kapal. Biasanya kapal berakhiran dengan angka (misal XX). 
Daftar armada kami: Maju Daya, Mega Daya, Intan Daya, Terus Daya, Central Daya.

2. NOMOR PERMINTAAN (PENTING: BISA LEBIH DARI SATU!):
Cari nomor registrasi dokumen. Format standarnya WAJIB TERDIRI DARI 4 BAGIAN: [KODE KAPAL]/[Bulan]/[Tahun]/[Nomor Urut]
- AWAS! Jangan sampai melewatkan angka BULAN. Contoh Benar: ID298/4/2026/010. Contoh SALAH: ID296/2026/010 (Bulannya hilang).
- JIKA ADA LEBIH DARI SATU NOMOR PERMINTAAN di kertas (misalnya dipisahkan koma), EKSTRAK SEMUANYA dan gabungkan dengan koma. (Contoh keluaran: "ID298/4/2026/010, ID298/3/2026/009").

3. NAMA BARANG:
Cari daftar barang yang diserahterimakan di dalam dokumen. Gabungkan nama-nama barang tersebut menjadi satu teks yang dipisahkan dengan koma. (Contoh: "1. Starting Valve, 2. Exhaust Valve")

ATURAN WAJIB (CROSS-CHECK SANGAT KETAT):
KODE HURUF & ANGKA di awal "Nomor Permintaan" PASTI merupakan inisial/singkatan dari "Nama Kapal". Jika nama kapal diakhiri angka (kita sebut XX), berikut rumus baku singkatannya:
- "Maju Daya XX" -> Singkatan: MDXX -> (Nomor Permintaan berawalan "MDXX/")
- "Mega Daya XX" -> Singkatan: MEDXX -> (Nomor Permintaan berawalan "MEDXX/")
- "Intan Daya XX" -> Singkatan: IDXX -> (Nomor Permintaan berawalan "IDXX/")
- "Terus Daya XX" -> Singkatan: TDXX -> (Nomor Permintaan berawalan "TDXX/")
- "Central Daya XX" -> Singkatan: CDXX -> (Nomor Permintaan berawalan "CDXX/")

BACA BAIK-BAIK GAMBARNYA! Jika tinta pudar dan huruf/angka sulit dibaca, JADIKAN RUMUS SINGKATAN NAMA KAPAL DI ATAS SEBAGAI PATOKAN MUTLAK untuk mengoreksi kode awal pada Nomor Permintaan.

INSTRUKSI FINAL:
Jawab HANYA dengan format JSON valid, tanpa markdown, tanpa penjelasan tambahan:
{"noPermintaan": "SEMUA_NOMOR_HASIL_KOREKSI_DIPISAH_KOMA", "namaKapal": "NAMA_KAPAL", "namaBarang": "DAFTAR_BARANG_HASIL_EKSTRAKSI"}`;

    let result;
    let isSuccess = false;
    let attempts = 0;
    const maxAttempts = geminiKeys.length;

    if (maxAttempts === 0) throw new Error('Tidak ada Gemini API Key yang terkonfigurasi di .env');

    // ROTASI API KEY: Coba pakai kunci satu per satu sampai berhasil
    while (!isSuccess && attempts < maxAttempts) {
      try {
        const apiKey = geminiKeys[currentKeyIndex];
        const genAI = new GoogleGenerativeAI(apiKey);
        const geminiModel = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

        console.log(`[OCR] Using Gemini API Key slot: #${currentKeyIndex + 1}`);

        result = await geminiModel.generateContent({
          contents: [{
            role: "user",
            parts: [
              { text: prompt },
              { inlineData: { data: imageBase64, mimeType: mimeType } }
            ]
          }],
          // PEMBATASAN UNTUK KECEPATAN KILAT:
          // Kunci output murni ke JSON agar AI tidak bisa "yapping" atau menulis penjelasan
          generationConfig: {
            temperature: 0.1, // Suhu super rendah agar fokus
            responseMimeType: "application/json"
          }
        });

        isSuccess = true; // Berhasil! Keluar dari loop.
      } catch (geminiError) {
        console.error(`[OCR] Key #${currentKeyIndex + 1} Failed:`, geminiError.message);
        // Geser ke kunci berikutnya
        currentKeyIndex = (currentKeyIndex + 1) % geminiKeys.length;
        attempts++;

        if (attempts >= maxAttempts) {
          throw new Error('Gangguan koneksi ke AI atau kuota harian habis. Silakan coba lagi.');
        }
        console.log(`[OCR] Switching to next API Key...`);
      }
    }

    const responseText = result.response.text().trim();
    console.log('Gemini Response:', responseText);

    // Parse JSON dari response Gemini
    let parsedData = { noPermintaan: null, namaKapal: null, namaBarang: null };
    try {
      // Bersihkan response (kadang Gemini membungkusnya dengan ```json ... ```)
      const cleanJson = responseText.replace(/```json\n?|\n?```/g, '').trim();
      parsedData = JSON.parse(cleanJson);
    } catch (parseErr) {
      console.warn('Could not parse JSON, trying regex fallback...');
      // Fallback: coba cari dengan regex langsung dari response text
      const fallbackRegex = /([A-Z]{2}\d{2}[\s/\\|-]?\d{2,4}[\s/\\|-]?\d{2,4}[\s/\\|-]?\d{2,3})/i;
      const fallbackMatch = responseText.match(fallbackRegex);
      parsedData.noPermintaan = fallbackMatch ? fallbackMatch[0] : null;
    }

    console.log('Final Result:', parsedData);

    // --- SIMPAN KE DATABASE SUPABASE ---
    // DIPINDAHKAN KE FRONTEND: Data tidak lagi disimpan otomatis di sini.
    // User akan mereview hasil OCR terlebih dahulu, membalas email,
    // lalu baru menekan tombol "Simpan ke Arsip" yang akan menyimpan
    // data ke Supabase dan foto asli ke IndexedDB (local storage) di perangkat.
    let dbStatus = "Penyimpanan arsip dipindahkan ke sisi klien (frontend)";

    res.json({
      success: true,
      data: {
        noPermintaan: parsedData.noPermintaan,
        namaKapal: parsedData.namaKapal,
        namaBarang: parsedData.namaBarang,
        rawText: responseText
      },
      database: dbStatus
    });

  } catch (error) {
    console.error('ERROR in process-stb:', error.message);
    res.status(500).json({
      success: false,
      error: 'Terjadi gangguan jaringan atau sistem sedang sibuk. Silakan coba lagi.',
      details: error.message
    });
  }
});

// 2. Endpoint to search Gmail threads based on No Permintaan
app.get('/api/gmail/search', async (req, res) => {
  try {
    const { query } = req.query;
    if (!query) return res.status(400).json({ error: 'Query parameter is required' });

    if (!process.env.GMAIL_REFRESH_TOKEN) {
      return res.status(500).json({ success: false, error: 'Gmail OAuth not configured' });
    }

    console.log(`Searching Gmail for: "${query}"`);

    // Gunakan query lebih fleksibel: biarkan Gmail API mencari di subject/body/attachment sesuai query
    const response = await gmail.users.threads.list({
      userId: 'me',
      q: query,
      maxResults: 5,
    });

    const threads = response.data.threads || [];
    const threadDetails = [];

    for (const t of threads) {
      const detail = await gmail.users.threads.get({
        userId: 'me',
        id: t.id,
        format: 'metadata',
        metadataHeaders: ['Subject', 'From', 'Date']
      });

      const headers = detail.data.messages[0].payload.headers;
      const subject = headers.find(h => h.name.toLowerCase() === 'subject')?.value || 'No Subject';
      const from = headers.find(h => h.name.toLowerCase() === 'from')?.value || 'Unknown';
      const date = headers.find(h => h.name.toLowerCase() === 'date')?.value || '';

      threadDetails.push({
        id: t.id,
        messageId: detail.data.messages[detail.data.messages.length - 1].id,
        subject,
        sender: from,
        date,
        snippet: detail.data.snippet
      });
    }

    console.log(`Found ${threadDetails.length} threads.`);
    res.json({ success: true, threads: threadDetails });

  } catch (error) {
    console.error('Error searching Gmail:', error.message);
    res.status(500).json({ error: 'Failed to search Gmail' });
  }
});

// 3. Endpoint to send reply with attachment
app.post('/api/gmail/reply', upload.single('image'), async (req, res) => {
  try {
    const { threadId, messageId, subject, toEmail } = req.body;
    const file = req.file;

    if (!file) return res.status(400).json({ error: 'Image attachment is required' });

    console.log(`Sending reply to thread: ${threadId}`);

    // Meningkatkan Kualitas Foto sebelum dikirim ke Gmail
    console.log('Enhancing image quality for Gmail attachment...');
    const enhancedImageBuffer = await sharp(file.buffer)
      .normalize() // Menyeimbangkan kontras agar lebih jelas
      .sharpen()   // Mempertajam foto yang agak buram
      .jpeg({ quality: 100 }) // Kualitas maksimal 100%
      .toBuffer();

    // Construct MIME Message
    const boundary = 'STB_SCANNER_BOUNDARY';
    const nl = '\r\n';

    const str = [
      `To: ${toEmail}`,
      `Subject: Re: ${subject}`,
      `In-Reply-To: ${messageId}`,
      `References: ${messageId}`,
      `Content-Type: multipart/mixed; boundary="${boundary}"`,
      '',
      `--${boundary}`,
      'Content-Type: text/plain; charset="UTF-8"',
      'Content-Transfer-Encoding: 7bit',
      '',
      'Terlampir Foto STB (Serah Terima Barang) yang telah diproses oleh sistem STB Vision.',
      'Terima kasih.',
      '',
      `--${boundary}`,
      `Content-Type: image/jpeg; name="${file.originalname || 'stb_capture.jpg'}"`,
      `Content-Disposition: attachment; filename="${file.originalname || 'stb_capture.jpg'}"`,
      'Content-Transfer-Encoding: base64',
      '',
      enhancedImageBuffer.toString('base64'),
      '',
      `--${boundary}--`
    ].join(nl);

    const encodedMessage = Buffer.from(str)
      .toString('base64')
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');

    await gmail.users.messages.send({
      userId: 'me',
      requestBody: {
        raw: encodedMessage,
        threadId: threadId
      }
    });

    console.log('Reply sent successfully!');
    res.json({ success: true, message: 'Reply sent successfully' });

  } catch (error) {
    console.error('Error sending reply:', error.message);
    res.status(500).json({ error: 'Failed to send reply', details: error.message });
  }
});

app.listen(port, () => {
  console.log(`\n🚀 STB Backend API running on port ${port}`);
  console.log(`🤖 OCR Engine: Google Gemini 2.5 Flash`);
  console.log(`📧 Gmail: ${process.env.GMAIL_REFRESH_TOKEN ? 'Configured ✓' : 'Not configured ✗'}`);
  console.log(`🔑 Gemini: ${geminiKeys.length > 0 ? `Configured ✓ (${geminiKeys.length} keys active)` : 'Not configured ✗'}\n`);
});

// EXPORT UNTUK VERCEL SERVERLESS FUNCTION
module.exports = app;
