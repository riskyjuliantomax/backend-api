require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { google } = require('googleapis');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const sharp = require('sharp');
const fs = require('fs');
const { createWorker } = require('tesseract.js');
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
// Baca semua API key dari .env sebagai fallback
const envGeminiKeys = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
  .split(',')
  .map(k => k.trim())
  .filter(Boolean);

let currentKeyIndex = 0;

// Ambil daftar API Key aktif dari Supabase. Jika Supabase tidak tersedia, pakai key dari .env.
async function getActiveGeminiKeys() {
  try {
    if (!supabase) return envGeminiKeys;
    const { data, error } = await supabase
      .from('api_keys')
      .select('key_value')
      .eq('service_name', 'gemini')
      .eq('is_active', true);
    if (error || !data || data.length === 0) return envGeminiKeys;
    return data.map(r => r.key_value);
  } catch (e) {
    console.warn('[Keys] Supabase key fetch failed, using .env keys:', e.message);
    return envGeminiKeys;
  }
}

// --- TESSERACT FALLBACK OCR ---
// Parsing regex untuk ekstrak noPermintaan dan namaKapal dari teks mentah Tesseract
function parseTesseractText(text) {
  const armadas = [
    { name: 'Maju Daya', codes: ['MD'] },
    { name: 'Mega Daya', codes: ['MED'] },
    { name: 'Intan Daya', codes: ['ID'] },
    { name: 'Terus Daya', codes: ['TD'] },
    { name: 'Central Daya', codes: ['CD'] },
  ];

  // Regex fleksibel: KODE/BULAN/TAHUN/URUT (toleran terhadap spasi dan separator tak biasa)
  const requestRegex = /([A-Z]{2,5}\d{1,3})\s*[/\\|.]\s*(\d{1,2})\s*[/\\|.]\s*(\d{4})\s*[/\\|.]\s*(\d{2,4})/gi;
  const matches = [...text.matchAll(requestRegex)];
  const noPermintaan = matches.length > 0
    ? matches.map(m => `${m[1]}/${m[2]}/${m[3]}/${m[4]}`.toUpperCase()).join(', ')
    : null;

  // Cari nama kapal armada
  let namaKapal = null;
  const upperText = text.toUpperCase();
  for (const armada of armadas) {
    if (upperText.includes(armada.name.toUpperCase())) {
      // Coba tangkap nomor kapal di belakang nama armada
      const shipNumRegex = new RegExp(armada.name.replace(' ', '\\s+') + '\\s*(\\d+)', 'i');
      const shipMatch = text.match(shipNumRegex);
      namaKapal = shipMatch ? `${armada.name} ${shipMatch[1]}` : armada.name;
      break;
    }
    // Fallback: cek via kode singkatan
    if (noPermintaan) {
      for (const code of armada.codes) {
        if (noPermintaan.toUpperCase().startsWith(code)) {
          namaKapal = armada.name;
          break;
        }
      }
    }
    if (namaKapal) break;
  }

  // Nama barang: ambil baris-baris yang mengandung angka urut dan kata kerja material
  const itemLines = text.split('\n').filter(line =>
    /^\s*\d+[.)\s]/.test(line) && line.trim().length > 5
  );
  const namaBarang = itemLines.length > 0
    ? itemLines.map((l, i) => `${i + 1}. ${l.trim().replace(/^\d+[.)\s]+/, '').trim()}`).join(', ')
    : null;

  return { noPermintaan, namaKapal, namaBarang };
}

async function processWithTesseract(buffer) {
  console.log('[OCR] Running Tesseract.js fallback...');
  // OEM 1 = LSTM only (faster), PSM 6 = Assume a single uniform block of text
  const worker = await createWorker('eng', 1, {
    langPath: __dirname,
    logger: m => { if (m.status === 'recognizing text') console.log(`[Tesseract] ${Math.round(m.progress * 100)}%`); }
  });
  try {
    await worker.setParameters({ tessedit_pageseg_mode: '6' });
    const { data: { text } } = await worker.recognize(buffer);
    console.log('[Tesseract] Raw text (first 300 chars):', text.substring(0, 300));
    return parseTesseractText(text);
  } finally {
    await worker.terminate();
  }
}

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

// Root endpoint - penting untuk Vercel health check
app.get('/', (req, res) => {
  res.json({ status: 'ok', message: 'STB Backend API is running with Gemini Vision' });
});

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

    // Buffer untuk GEMINI: resolusi tinggi 2560px agar AI bisa baca teks kecil
    const compressedBuffer = await sharp(req.file.buffer)
      .resize({ width: 2560, withoutEnlargement: true })
      .grayscale()
      .jpeg({ quality: 95 })
      .toBuffer();

    // Buffer untuk TESSERACT: ukuran lebih kecil 1200px + kontras tinggi agar cepat diproses
    // (Vercel free limit ~10s, Tesseract butuh gambar kecil agar tidak timeout)
    const tesseractBuffer = await sharp(req.file.buffer)
      .resize({ width: 1200, withoutEnlargement: true })
      .grayscale()
      .normalize()  // kontras tinggi khusus untuk keterbacaan Tesseract
      .sharpen()
      .jpeg({ quality: 85 })
      .toBuffer();

    console.log(`OCR Buffer (Gemini): ${(compressedBuffer.length / 1024).toFixed(1)} KB | Tesseract: ${(tesseractBuffer.length / 1024).toFixed(1)} KB`);
    console.log('Sending optimized image to OCR engine...');

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

    // Cek apakah user meminta pakai AI atau Tesseract lokal
    const useAI = req.body.useAI !== 'false'; // default: true

    let parsedData = { noPermintaan: null, namaKapal: null, namaBarang: null };
    let ocrEngine = 'tesseract';

    if (useAI) {
      // --- COBA OCR DENGAN GEMINI (KEY DARI SUPABASE / .ENV) ---
      const geminiKeys = await getActiveGeminiKeys();
      let result;
      let isSuccess = false;
      let attempts = 0;
      const maxAttempts = geminiKeys.length;

      if (maxAttempts > 0) {
        while (!isSuccess && attempts < maxAttempts) {
          const apiKey = geminiKeys[currentKeyIndex % geminiKeys.length];
          const genAI = new GoogleGenerativeAI(apiKey);
          const geminiModel = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
          console.log(`[OCR] Using Gemini API Key slot: #${(currentKeyIndex % geminiKeys.length) + 1}`);

          let keySuccess = false;
          let keyAttempts = 0;
          const maxKeyRetries = 2; // total 3 attempts per key

          while (!keySuccess && keyAttempts <= maxKeyRetries) {
            try {
              result = await geminiModel.generateContent({
                contents: [{ role: 'user', parts: [
                  { text: prompt },
                  { inlineData: { data: imageBase64, mimeType: mimeType } }
                ]}],
                generationConfig: { temperature: 0.1, responseMimeType: 'application/json' }
              });

              keySuccess = true;
              isSuccess = true;
              ocrEngine = 'gemini';
            } catch (geminiError) {
              keyAttempts++;
              const errMsg = geminiError.message || '';
              const isTransient = errMsg.includes('503') || 
                                  errMsg.includes('429') || 
                                  errMsg.toLowerCase().includes('demand') ||
                                  errMsg.toLowerCase().includes('overloaded') ||
                                  errMsg.toLowerCase().includes('service unavailable') ||
                                  errMsg.toLowerCase().includes('resource exhausted') ||
                                  errMsg.toLowerCase().includes('fetch failed') ||
                                  errMsg.toLowerCase().includes('timeout');

              if (isTransient && keyAttempts <= maxKeyRetries) {
                console.warn(`[OCR] Transient error on Key #${(currentKeyIndex % geminiKeys.length) + 1}: ${errMsg}. Retrying in 1.5s... (Attempt ${keyAttempts}/${maxKeyRetries + 1})`);
                await new Promise(resolve => setTimeout(resolve, 1500));
              } else {
                console.error(`[OCR] Key #${(currentKeyIndex % geminiKeys.length) + 1} Failed:`, errMsg);
                currentKeyIndex = (currentKeyIndex + 1) % Math.max(geminiKeys.length, 1);
                attempts++;
                if (attempts < maxAttempts) console.log('[OCR] Switching to next API Key...');
                break; // Break inner loop to try next key
              }
            }
          }
        }
      }

      if (isSuccess && result) {
        // Parse JSON dari Gemini
        const responseText = result.response.text().trim();
        console.log('Gemini Response:', responseText);
        try {
          const cleanJson = responseText.replace(/```json\n?|\n?```/g, '').trim();
          parsedData = JSON.parse(cleanJson);
        } catch {
          console.warn('[OCR] Gemini JSON parse failed, running Tesseract fallback...');
          parsedData = await processWithTesseract(tesseractBuffer);
          ocrEngine = 'tesseract-fallback';
        }
      } else {
        // Semua key Gemini gagal — fallback ke Tesseract
        console.warn('[OCR] All Gemini keys exhausted. Falling back to Tesseract...');
        parsedData = await processWithTesseract(tesseractBuffer);
        ocrEngine = 'tesseract-fallback';
      }
    } else {
      // --- OCR LANGSUNG DENGAN TESSERACT (TANPA GEMINI) ---
      parsedData = await processWithTesseract(tesseractBuffer);
      ocrEngine = 'tesseract';
    }

    console.log(`[OCR] Engine used: ${ocrEngine}`);
    console.log('Final Result:', parsedData);

    res.json({
      success: true,
      ocrEngine,
      data: {
        noPermintaan: parsedData.noPermintaan,
        namaKapal: parsedData.namaKapal,
        namaBarang: parsedData.namaBarang,
      },
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

// Fungsi untuk menghitung kecocokan thread Gmail dengan query No Permintaan secara akurat
function calculateMatchScore(subject, snippet, query) {
  if (!query) return 0;
  
  const clean = (str) => (str || '').toLowerCase().replace(/[\s/\\|.-]/g, '');
  const cleanSubject = clean(subject);
  const cleanSnippet = clean(snippet);
  const cleanQuery = clean(query);
  
  // 1. Kecocokan persis (mengabaikan spasi/separator)
  if (cleanSubject.includes(cleanQuery) || cleanSnippet.includes(cleanQuery)) {
    return 100;
  }
  
  // 2. Kecocokan sebagian (persentase kecocokan potongan nomor permintaan)
  // Misal: "SL9/11/2025/014" -> parts: ["SL9", "11", "2025", "014"]
  const parts = query.split(/[\s/\\|.-]+/).filter(Boolean);
  if (parts.length === 0) return 0;
  
  let matchedParts = 0;
  const lowerSubject = (subject || '').toLowerCase();
  const lowerSnippet = (snippet || '').toLowerCase();
  
  for (const part of parts) {
    const lowerPart = part.toLowerCase();
    if (lowerSubject.includes(lowerPart) || lowerSnippet.includes(lowerPart)) {
      matchedParts++;
    }
  }
  
  const percentage = Math.round((matchedParts / parts.length) * 100);
  return Math.min(percentage, 85); // Batasi kecocokan parsial di 85% agar beda dengan 100% exact match
}

// 2. Endpoint to search Gmail threads based on No Permintaan
app.get('/api/gmail/search', async (req, res) => {
  try {
    const { query } = req.query;
    if (!query) return res.status(400).json({ success: false, error: 'Query parameter is required' });

    if (!process.env.GMAIL_REFRESH_TOKEN) {
      return res.status(500).json({ success: false, error: 'Gmail OAuth not configured' });
    }

    console.log(`Searching Gmail for: "${query}"`);

    // Gunakan query fleksibel: biarkan Gmail API mencari di subject/body/attachment sesuai query
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
      const toHeader = headers.find(h => h.name.toLowerCase() === 'to')?.value || '';
      const ccHeader = headers.find(h => h.name.toLowerCase() === 'cc')?.value || '';
      // Extract email addresses from headers
      const emailRegex = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/ig;
      const recipientsList = [];
      const collect = (s) => {
        if (!s) return;
        const matches = s.match(emailRegex);
        if (matches) matches.forEach(m => { if (!recipientsList.includes(m)) recipientsList.push(m); });
      };
      collect(from);
      collect(toHeader);
      collect(ccHeader);
      const recipients = recipientsList.join(', ');
      const date = headers.find(h => h.name.toLowerCase() === 'date')?.value || '';

      const matchScore = calculateMatchScore(subject, detail.data.snippet, query);

      threadDetails.push({
        id: t.id,
        messageId: detail.data.messages[detail.data.messages.length - 1].id,
        subject,
        sender: from,
        to: toHeader,
        cc: ccHeader,
        recipients,
        recipientsList,
        date,
        snippet: detail.data.snippet,
        matchScore
      });
    }

    // Urutkan berdasarkan matchScore tertinggi agar kecocokan persis selalu berada di posisi teratas
    threadDetails.sort((a, b) => b.matchScore - a.matchScore);

    console.log(`Found ${threadDetails.length} threads. Sorted by match score.`);
    res.json({ success: true, threads: threadDetails });

  } catch (error) {
    console.error('Error searching Gmail:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to search Gmail',
      details: error.message,
      stack: error.stack
    });
  }
});

// 3. Endpoint to send reply with attachment
app.post('/api/gmail/reply', upload.single('image'), async (req, res) => {
  try {
    const { threadId, messageId, subject, toEmail, ccEmail, toEmailsAll } = req.body;
    const replyAll = req.body.replyAll === 'true' || req.body.replyAll === true;
    const file = req.file;

    if (!file) return res.status(400).json({ error: 'Image attachment is required' });

    console.log(`Sending reply to thread: ${threadId}`);

    // Gunakan buffer asli agar sesuai dengan crop di frontend
    console.log('Using original cropped image buffer for Gmail attachment...');
    const enhancedImageBuffer = file.buffer;

    // Construct MIME Message
    const boundary = 'STB_SCANNER_BOUNDARY';
    const nl = '\r\n';

    let toHeader = toEmail || '';
    let ccHeader = '';

    if (replyAll) {
      const toList = [];
      if (toEmail) toList.push(toEmail);
      if (toEmailsAll) {
        const emailRegex = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/ig;
        const matches = toEmailsAll.match(emailRegex);
        if (matches) {
          matches.forEach(m => {
            if (!toList.some(x => x.toLowerCase() === m.toLowerCase())) {
              toList.push(m);
            }
          });
        }
      }
      toHeader = toList.join(', ');
      ccHeader = ccEmail || '';
    }

    const mimeHeaders = [
      `To: ${toHeader}`,
      `Subject: Re: ${subject}`,
      `In-Reply-To: ${messageId}`,
      `References: ${messageId}`
    ];

    if (ccHeader) {
      mimeHeaders.push(`Cc: ${ccHeader}`);
    }

    mimeHeaders.push(`Content-Type: multipart/mixed; boundary="${boundary}"`);
    mimeHeaders.push('');

    const str = [
      ...mimeHeaders,
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

// 404 Handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found', path: req.path, method: req.method });
});

app.listen(port, () => {
  console.log(`\n🚀 STB Backend API running on port ${port}`);
  console.log(`🤖 OCR Engine: Google Gemini 2.5 Flash`);
  console.log(`📧 Gmail: ${process.env.GMAIL_REFRESH_TOKEN ? 'Configured ✓' : 'Not configured ✗'}`);
  console.log(`🔑 Gemini: ${geminiKeys.length > 0 ? `Configured ✓ (${geminiKeys.length} keys active)` : 'Not configured ✗'}\n`);
});

// EXPORT UNTUK VERCEL SERVERLESS FUNCTION
module.exports = app;
