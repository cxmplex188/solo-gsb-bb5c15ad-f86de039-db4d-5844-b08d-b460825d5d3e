'use strict';

/**
 * ABX 盲听测试服务端
 *
 * 设计要点：
 * - 音频文件只保存在本机磁盘（DATA_DIR，默认 ./data/<sessionId>/）
 * - 会话状态（含 X 映射）只保存在本机内存中，提交前任何接口都不会泄露映射
 * - X 音频通过 chunked 流式响应下发（不带 Content-Length / 文件名），
 *   避免客户端通过文件大小或扩展名推断 X 的真实身份
 */

const express = require('express');
const multer = require('multer');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = Number(process.env.PORT || 3000);
const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, 'data');
const SESSION_TTL_MS = Number(process.env.SESSION_TTL_MS || 24 * 3600 * 1000);
const MAX_TRIALS = 50;

const ALLOWED_EXT = new Set([
  '.wav', '.mp3', '.flac', '.ogg', '.oga',
  '.m4a', '.aac', '.opus', '.webm', '.aif', '.aiff',
]);

fs.mkdirSync(DATA_DIR, { recursive: true });

/** 会话仅保存在本机内存：id -> session */
const sessions = new Map();

/** 恢复标识（仅存 SHA-256 哈希）：tokenHash -> sessionId，提交/删除/过期即移除 */
const recoveryTokens = new Map();

function hashToken(token) {
  return crypto.createHash('sha256').update(String(token)).digest('hex');
}

/** 使会话的恢复标识立即失效（提交、删除、过期时调用） */
function dropRecoveryToken(s) {
  if (s && s.tokenHash) {
    recoveryTokens.delete(s.tokenHash);
    s.tokenHash = null;
  }
}

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ---------------------------------------------------------------------------
// 上传（A / B 两个音频文件，要求同格式）
// ---------------------------------------------------------------------------

const storage = multer.diskStorage({
  destination(req, file, cb) {
    if (!req.abxDir) {
      req.abxSessionId = crypto.randomUUID();
      req.abxDir = path.join(DATA_DIR, req.abxSessionId);
      fs.mkdirSync(req.abxDir, { recursive: true });
    }
    cb(null, req.abxDir);
  },
  filename(req, file, cb) {
    const ext = path.extname(file.originalname).toLowerCase();
    // 统一命名为 A.<ext> / B.<ext>，磁盘上也不保留原始文件名
    cb(null, (file.fieldname === 'a' ? 'A' : 'B') + ext);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024, files: 2 },
  fileFilter(req, file, cb) {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXT.has(ext)) {
      return cb(new Error(`不支持的音频格式：${ext || '(无扩展名)'}`));
    }
    // 要求 A、B 为同格式音频
    if (!req.abxExt) req.abxExt = ext;
    else if (req.abxExt !== ext) {
      return cb(new Error('A 与 B 必须是同格式音频（扩展名不一致）'));
    }
    cb(null, true);
  },
});

function clampInt(value, min, max, fallback) {
  const n = Number.parseInt(value, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function removeDir(dir) {
  if (dir) fs.rm(dir, { recursive: true, force: true }, () => {});
}

app.post('/api/session', (req, res) => {
  upload.fields([{ name: 'a', maxCount: 1 }, { name: 'b', maxCount: 1 }])(req, res, (err) => {
    if (err) {
      removeDir(req.abxDir);
      return res.status(400).json({ error: err.message });
    }
    if (!req.files || !req.files.a || !req.files.b) {
      removeDir(req.abxDir);
      return res.status(400).json({ error: '请同时上传音频 A 与音频 B' });
    }

    const numTrials = clampInt(req.body.numTrials, 1, MAX_TRIALS, 8);
    const id = req.abxSessionId;

    // 服务端生成随机 X 题序：每题独立随机为 A 或 B，映射只存在于此
    const trials = Array.from({ length: numTrials }, () => ({
      x: crypto.randomInt(2) === 0 ? 'A' : 'B',
      answer: null,
    }));

    // 恢复标识：高熵随机 token，仅哈希存内存，明文只下发给本机浏览器一次
    const recoveryToken = crypto.randomBytes(32).toString('base64url');
    const tokenHash = hashToken(recoveryToken);

    sessions.set(id, {
      id,
      dir: req.abxDir,
      fileA: req.files.a[0].path,
      fileB: req.files.b[0].path,
      nameA: req.files.a[0].originalname,
      nameB: req.files.b[0].originalname,
      trials,
      submitted: false,
      results: null,
      createdAt: Date.now(),
      tokenHash,
    });
    recoveryTokens.set(tokenHash, id);

    res.json({ id, numTrials, recoveryToken });
  });
});

// ---------------------------------------------------------------------------
// 会话状态（不包含 X 映射）
// ---------------------------------------------------------------------------

function getSession(req, res) {
  const s = sessions.get(req.params.id);
  if (!s) {
    res.status(404).json({ error: '会话不存在或已过期' });
    return null;
  }
  return s;
}

app.get('/api/session/:id', (req, res) => {
  const s = getSession(req, res);
  if (!s) return;
  res.json({
    id: s.id,
    numTrials: s.trials.length,
    answers: s.trials.map((t) => t.answer),
    submitted: s.submitted,
    results: s.submitted ? s.results : undefined,
  });
});

// 会话恢复：凭恢复标识找回未提交会话的非敏感状态。
// 只返回题数与已选答案，绝不包含 X 映射；已提交/删除/过期的会话一律 404。
app.post('/api/session/recover', (req, res) => {
  const token = req.body && typeof req.body.token === 'string' ? req.body.token : '';
  const s = token ? sessions.get(recoveryTokens.get(hashToken(token))) : null;
  if (!s || s.submitted) {
    return res.status(404).json({ error: '没有可恢复的会话（可能已提交、删除或过期）' });
  }
  res.json({
    id: s.id,
    numTrials: s.trials.length,
    answers: s.trials.map((t) => t.answer),
    submitted: false,
  });
});

// ---------------------------------------------------------------------------
// 音频下发
// ---------------------------------------------------------------------------

app.get('/api/session/:id/audio/:which', (req, res) => {
  const s = getSession(req, res);
  if (!s) return;
  const which = req.params.which;
  if (which !== 'A' && which !== 'B') {
    return res.status(400).json({ error: '参数错误' });
  }
  res.setHeader('Cache-Control', 'no-store');
  res.sendFile(which === 'A' ? s.fileA : s.fileB);
});

// X 音频：根据隐藏映射返回 A 或 B 的内容。
// 使用 chunked 流式响应，不带 Content-Length / Content-Disposition，
// 避免通过文件大小或文件名泄露 X 的身份。
app.get('/api/session/:id/trial/:n/x', (req, res) => {
  const s = getSession(req, res);
  if (!s) return;
  const n = Number.parseInt(req.params.n, 10);
  const t = s.trials[n];
  if (!t) return res.status(404).json({ error: '题目不存在' });
  const file = t.x === 'A' ? s.fileA : s.fileB;
  res.setHeader('Content-Type', 'application/octet-stream');
  res.setHeader('Cache-Control', 'no-store');
  fs.createReadStream(file).pipe(res);
});

// ---------------------------------------------------------------------------
// 作答 / 提交
// ---------------------------------------------------------------------------

app.post('/api/session/:id/answer', (req, res) => {
  const s = getSession(req, res);
  if (!s) return;
  if (s.submitted) return res.status(409).json({ error: '已提交，不可再修改答案' });
  const { trial, answer } = req.body || {};
  const t = s.trials[trial];
  if (!t) return res.status(400).json({ error: '题号无效' });
  if (answer !== 'A' && answer !== 'B') return res.status(400).json({ error: '答案无效' });
  t.answer = answer;
  res.json({ ok: true });
});

app.post('/api/session/:id/submit', (req, res) => {
  const s = getSession(req, res);
  if (!s) return;
  if (s.submitted) return res.json({ results: s.results }); // 幂等：重复提交返回同一结果
  if (s.trials.some((t) => !t.answer)) {
    return res.status(400).json({ error: '还有未作答的题目，无法提交' });
  }
  // 提交后一次性揭示：正确率 + 逐题映射
  const detail = s.trials.map((t, i) => ({
    trial: i + 1,
    x: t.x,
    answer: t.answer,
    correct: t.x === t.answer,
  }));
  const correct = detail.filter((d) => d.correct).length;
  s.results = {
    total: detail.length,
    correct,
    accuracy: correct / detail.length,
    detail,
  };
  s.submitted = true;
  dropRecoveryToken(s); // 提交后恢复标识立即失效
  res.json({ results: s.results });
});

app.delete('/api/session/:id', (req, res) => {
  const s = sessions.get(req.params.id);
  if (s) {
    dropRecoveryToken(s);
    sessions.delete(s.id);
    removeDir(s.dir);
  }
  res.json({ ok: true });
});

// ---------------------------------------------------------------------------
// 过期会话清理（默认 24 小时）
// ---------------------------------------------------------------------------

setInterval(() => {
  const now = Date.now();
  for (const s of sessions.values()) {
    if (now - s.createdAt > SESSION_TTL_MS) {
      dropRecoveryToken(s);
      sessions.delete(s.id);
      removeDir(s.dir);
    }
  }
}, 3600 * 1000).unref();

app.listen(PORT, () => {
  console.log(`ABX 盲听测试已启动: http://localhost:${PORT}`);
  console.log(`音频与会话数据仅保存在本机: ${DATA_DIR}`);
});
