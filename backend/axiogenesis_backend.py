#!/usr/bin/env python3
"""
AXIOGENESIS v3 — Autonomous Cognitive Backend
FastAPI server with continuous cognitive loop, neural engine,
knowledge graph, vector memory, instinct drives, and meta-learning.
"""

import asyncio
import os
import json
import math
import random
import re
import sqlite3
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── Optional heavy deps ──────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[WARN] PyTorch not available — neural engine disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not available — using fallback vector memory")

# ════════════════════════════════════════════════════════════════════
# 1. DATABASE LAYER
# ════════════════════════════════════════════════════════════════════

DB_PATH = Path("axiogenesis.db")

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source_type TEXT DEFAULT 'unknown',
            source_name TEXT,
            char_count INTEGER,
            ts REAL,
            processed INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS kg_nodes (
            name TEXT PRIMARY KEY,
            freq INTEGER DEFAULT 1,
            domain TEXT DEFAULT 'general',
            last_seen REAL
        );
        CREATE TABLE IF NOT EXISTS kg_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            confidence REAL DEFAULT 0.7,
            evidence INTEGER DEFAULT 1,
            source TEXT,
            ts REAL,
            UNIQUE(subject, predicate, object)
        );
        CREATE TABLE IF NOT EXISTS beliefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement TEXT UNIQUE,
            confidence REAL,
            domain TEXT,
            evidence INTEGER DEFAULT 1,
            last_verified REAL,
            decay_rate REAL DEFAULT 0.002
        );
        CREATE TABLE IF NOT EXISTS vector_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            vector BLOB,
            source_type TEXT,
            ts REAL
        );
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            ts REAL
        );
        CREATE TABLE IF NOT EXISTS short_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            summary TEXT,
            metadata TEXT,
            ts REAL
        );
        CREATE TABLE IF NOT EXISTS threat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            threat_type TEXT,
            severity INTEGER,
            text_sample TEXT,
            ts REAL
        );
        CREATE TABLE IF NOT EXISTS evolution_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            params_before TEXT,
            params_after TEXT,
            test_loss REAL,
            accepted INTEGER,
            ts REAL
        );
        CREATE TABLE IF NOT EXISTS instinct_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drive TEXT,
            activation REAL,
            action TEXT,
            ts REAL
        );
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loss REAL,
            accuracy REAL,
            perplexity REAL,
            corpus_chars INTEGER,
            kg_nodes INTEGER,
            kg_edges INTEGER,
            vector_count INTEGER,
            ts REAL
        );
        CREATE TABLE IF NOT EXISTS meta_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            param TEXT,
            old_value REAL,
            new_value REAL,
            reason TEXT,
            ts REAL
        );
    """)
    conn.commit()
    return conn

# ════════════════════════════════════════════════════════════════════
# 2. NEURAL ENGINE (PyTorch LSTM)
# ════════════════════════════════════════════════════════════════════

CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n\t"
VSIZ = len(CHARS)
C2I = {ch: i for i, ch in enumerate(CHARS)}
I2C = {i: ch for i, ch in enumerate(CHARS)}
SEQ_LEN = 50
EMB_DIM = 48
LSTM1 = 192
LSTM2 = 128
BATCH = 48

def enc(ch): return C2I.get(ch, 0)
def dec(i): return I2C.get(i, ' ')

class AxioLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VSIZ, EMB_DIM)
        self.lstm1 = nn.LSTM(EMB_DIM, LSTM1, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM1, LSTM2, batch_first=True)
        self.fc = nn.Linear(LSTM2, VSIZ)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return x

class NeuralEngine:
    def __init__(self):
        self.model: Optional[Any] = None
        self.optimizer = None
        self.loss_history: List[float] = []
        self.loss_fn = nn.CrossEntropyLoss() if TORCH_OK else None
        self.trained_chars = 0
        self.last_loss = None
        self.last_acc = None
        self.lr = 0.002
        self.lock = threading.Lock()

    def build(self):
        if not TORCH_OK:
            return
        with self.lock:
            self.model = AxioLSTM()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def set_lr(self, lr: float):
        self.lr = max(0.00005, min(0.01, lr))
        if self.optimizer:
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr

    def make_batch(self, text: str) -> Optional[Tuple]:
        if len(text) < SEQ_LEN + 2:
            return None
        xs, ys = [], []
        for _ in range(BATCH):
            start = random.randint(0, len(text) - SEQ_LEN - 2)
            seq = [enc(c) for c in text[start:start + SEQ_LEN]]
            target = enc(text[start + SEQ_LEN])
            xs.append(seq)
            ys.append(target)
        return (torch.tensor(xs, dtype=torch.long),
                torch.tensor(ys, dtype=torch.long))

    def train_step(self, text: str) -> Optional[Dict]:
        if not TORCH_OK or self.model is None:
            return None
        batch = self.make_batch(text)
        if batch is None:
            return None
        x, y = batch
        with self.lock:
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            loss_val = loss.item()
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            self.last_loss = loss_val
            self.last_acc = acc
            self.loss_history.append(loss_val)
            if len(self.loss_history) > 600:
                self.loss_history = self.loss_history[-600:]
            self.trained_chars += BATCH * SEQ_LEN
            return {"loss": loss_val, "accuracy": acc}

    def generate(self, seed: str = "", length: int = 120, temp: float = 0.85) -> str:
        if not TORCH_OK or self.model is None:
            return self._noise(28)
        ctx = [enc(c) for c in seed][-SEQ_LEN:]
        while len(ctx) < SEQ_LEN:
            ctx.insert(0, 0)
        out = ""
        with self.lock:
            self.model.eval()
            with torch.no_grad():
                for _ in range(length):
                    x = torch.tensor([ctx], dtype=torch.long)
                    logits = self.model(x)
                    probs = torch.softmax(logits.squeeze() / temp, dim=0).numpy()
                    idx = np.random.choice(len(probs), p=probs / probs.sum())
                    out += dec(int(idx))
                    ctx.append(int(idx))
                    if len(ctx) > SEQ_LEN:
                        ctx.pop(0)
        return out

    def perplexity(self) -> float:
        if not self.loss_history:
            return float('inf')
        avg = sum(self.loss_history[-50:]) / min(50, len(self.loss_history))
        return math.exp(min(avg, 20))

    def loss_slope(self) -> float:
        if len(self.loss_history) < 40:
            return 0.0
        recent = self.loss_history[-20:]
        older = self.loss_history[-40:-20]
        return (sum(older) / len(older)) - (sum(recent) / len(recent))

    def _noise(self, n: int = 28) -> str:
        g = '░▒▓▌▐╔╗╚╝║═'
        return ''.join(random.choice(g + ' ' * 5) for _ in range(n))

    def save(self, path: str = "axio_model.pt"):
        if TORCH_OK and self.model:
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss_history': self.loss_history,
                'trained_chars': self.trained_chars,
                'lr': self.lr,
            }, path)

    def load(self, path: str = "axio_model.pt") -> bool:
        if not TORCH_OK or not Path(path).exists():
            return False
        try:
            ck = torch.load(path, map_location='cpu')
            if self.model is None:
                self.build()
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.loss_history = ck.get('loss_history', [])
            self.trained_chars = ck.get('trained_chars', 0)
            self.lr = ck.get('lr', 0.002)
            return True
        except Exception as e:
            print(f"[WARN] Model load failed: {e}")
            return False

# ════════════════════════════════════════════════════════════════════
# 3. VECTOR MEMORY LAYER
# ════════════════════════════════════════════════════════════════════

class VectorMemory:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.vectorizer = TfidfVectorizer(max_features=512, stop_words='english') if SKLEARN_OK else None
        self._texts: List[str] = []
        self._matrix = None
        self._dirty = True
        self._lock = threading.Lock()

    def _bow_vector(self, text: str) -> np.ndarray:
        """Fallback bag-of-words vector if sklearn unavailable."""
        words = re.findall(r'\b[a-z]{3,12}\b', text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        # deterministic 64-dim hash projection
        vec = np.zeros(64, dtype=np.float32)
        for w, c in freq.items():
            h = hash(w) % 64
            vec[h] += c
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def add(self, text: str, source_type: str = "general"):
        if len(text) < 20:
            return
        text = text[:500]  # cap length
        if SKLEARN_OK:
            vec = np.zeros(512, dtype=np.float32)  # placeholder until refit
        else:
            vec = self._bow_vector(text)
        blob = vec.tobytes()
        with self._lock:
            self.db.execute(
                "INSERT INTO vector_memory (text, vector, source_type, ts) VALUES (?,?,?,?)",
                (text, blob, source_type, time.time())
            )
            self.db.commit()
            self._texts.append(text)
            self._dirty = True

    def _rebuild_index(self):
        if not self._dirty:
            return
        rows = self.db.execute("SELECT text FROM vector_memory ORDER BY id DESC LIMIT 2000").fetchall()
        self._texts = [r['text'] for r in rows]
        if not self._texts:
            self._matrix = None
            return
        if SKLEARN_OK and self.vectorizer:
            try:
                self._matrix = self.vectorizer.fit_transform(self._texts)
            except Exception:
                self._matrix = None
        self._dirty = False

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query.strip():
            return []
        with self._lock:
            self._rebuild_index()
            if not self._texts:
                return []
            if SKLEARN_OK and self._matrix is not None and self.vectorizer:
                try:
                    qvec = self.vectorizer.transform([query])
                    scores = cos_sim(qvec, self._matrix).flatten()
                    top_idx = np.argsort(scores)[::-1][:top_k]
                    return [{"text": self._texts[i], "score": float(scores[i])}
                            for i in top_idx if scores[i] > 0.05]
                except Exception:
                    pass
            # Fallback: word overlap
            qwords = set(re.findall(r'\b[a-z]{3,12}\b', query.lower()))
            scored = []
            for t in self._texts:
                tw = set(re.findall(r'\b[a-z]{3,12}\b', t.lower()))
                overlap = len(qwords & tw) / max(len(qwords | tw), 1)
                if overlap > 0:
                    scored.append((overlap, t))
            scored.sort(reverse=True)
            return [{"text": t, "score": s} for s, t in scored[:top_k]]

    def count(self) -> int:
        r = self.db.execute("SELECT COUNT(*) FROM vector_memory").fetchone()
        return r[0]

# ════════════════════════════════════════════════════════════════════
# 4. MEMORY HIERARCHY
# ════════════════════════════════════════════════════════════════════

class MemoryHierarchy:
    def __init__(self, db: sqlite3.Connection, vm: VectorMemory):
        self.db = db
        self.vm = vm
        self.working: deque = deque(maxlen=20)   # current convo context

    def add_working(self, role: str, content: str):
        ts = time.time()
        self.working.append({"role": role, "content": content, "ts": ts})
        self.db.execute("INSERT INTO working_memory (role,content,ts) VALUES (?,?,?)",
                        (role, content, ts))
        self.db.commit()

    def add_short_term(self, event_type: str, summary: str, metadata: dict = None):
        self.db.execute(
            "INSERT INTO short_term_memory (event_type,summary,metadata,ts) VALUES (?,?,?,?)",
            (event_type, summary, json.dumps(metadata or {}), time.time())
        )
        self.db.commit()

    def get_working(self) -> List[Dict]:
        return list(self.working)

    def get_short_term(self, limit: int = 20) -> List[Dict]:
        rows = self.db.execute(
            "SELECT event_type,summary,metadata,ts FROM short_term_memory ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_long_term_beliefs(self, limit: int = 20) -> List[Dict]:
        rows = self.db.execute(
            "SELECT statement,confidence,domain,evidence FROM beliefs ORDER BY confidence DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def search_semantic(self, query: str, k: int = 5) -> List[Dict]:
        return self.vm.search(query, k)

    def compose_context(self, query: str, max_items: int = 8) -> str:
        """Combine all memory layers into a context string for response generation."""
        parts = []
        # Working memory
        wm = self.get_working()[-4:]
        if wm:
            parts.append("WORKING MEMORY:")
            for m in wm:
                parts.append(f"  [{m['role'].upper()}] {m['content'][:200]}")
        # Semantic memory
        semantic = self.search_semantic(query, 4)
        if semantic:
            parts.append("SEMANTIC RECALL:")
            for s in semantic[:3]:
                if s['score'] > 0.1:
                    parts.append(f"  [{s['score']:.2f}] {s['text'][:150]}")
        # Long-term beliefs
        beliefs = self.get_long_term_beliefs(3)
        if beliefs:
            parts.append("CORE BELIEFS:")
            for b in beliefs[:3]:
                parts.append(f"  [{b['confidence']:.2f}] {b['statement'][:100]}")
        return "\n".join(parts)

    def stats(self) -> Dict:
        wm_count = len(self.working)
        stm_count = self.db.execute("SELECT COUNT(*) FROM short_term_memory").fetchone()[0]
        ltm_triples = self.db.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        ltm_beliefs = self.db.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
        sem_count = self.vm.count()
        return {
            "working": wm_count,
            "short_term": stm_count,
            "long_term_triples": ltm_triples,
            "long_term_beliefs": ltm_beliefs,
            "semantic": sem_count,
        }

# ════════════════════════════════════════════════════════════════════
# 5. KNOWLEDGE GRAPH
# ════════════════════════════════════════════════════════════════════

IS_PATS = [
    re.compile(r'([A-Z][a-z][\w\s]{2,25})\s+is\s+(?:a|an|the)\s+([\w\s]{3,35})', re.I),
    re.compile(r'([A-Z][a-z][\w\s]{2,25})\s+are\s+([\w\s]{3,35})', re.I),
    re.compile(r'([A-Z][a-z][\w\s]{2,25})\s+was\s+(?:a|an|the)\s+([\w\s]{3,35})', re.I),
]
HAS_PATS = [re.compile(r'([A-Z][a-z][\w\s]{2,20})\s+has\s+([\w\s]{3,25})', re.I)]
CAUSE_PATS = [
    re.compile(r'([\w\s]{3,25})\s+causes?\s+([\w\s]{3,25})', re.I),
    re.compile(r'([\w\s]{3,25})\s+leads?\s+to\s+([\w\s]{3,25})', re.I),
]

class KnowledgeGraph:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._lock = threading.Lock()

    def extract_triples(self, text: str) -> List[Dict]:
        triples = []
        for p in IS_PATS:
            for m in p.finditer(text):
                triples.append({"s": m.group(1).strip()[:40], "p": "is",
                                 "o": m.group(2).strip()[:40], "conf": 0.7})
                if len(triples) >= 30: break
        for p in HAS_PATS:
            for m in p.finditer(text):
                triples.append({"s": m.group(1).strip()[:40], "p": "has",
                                 "o": m.group(2).strip()[:40], "conf": 0.6})
                if len(triples) >= 40: break
        for p in CAUSE_PATS:
            for m in p.finditer(text):
                triples.append({"s": m.group(1).strip()[:40], "p": "causes",
                                 "o": m.group(2).strip()[:40], "conf": 0.65})
                if len(triples) >= 50: break
        return triples[:15]

    def add_triple(self, s: str, p: str, o: str, src: str = "?", conf: float = 0.7) -> bool:
        with self._lock:
            try:
                self.db.execute(
                    """INSERT INTO kg_edges (subject,predicate,object,confidence,evidence,source,ts)
                       VALUES (?,?,?,?,1,?,?)
                       ON CONFLICT(subject,predicate,object)
                       DO UPDATE SET confidence=MIN(1.0,confidence+0.05),
                                     evidence=evidence+1,ts=excluded.ts""",
                    (s[:40], p, o[:40], conf, src, time.time())
                )
                self.db.execute(
                    """INSERT INTO kg_nodes (name,freq,last_seen) VALUES (?,1,?)
                       ON CONFLICT(name) DO UPDATE SET freq=freq+1,last_seen=excluded.last_seen""",
                    (s[:40], time.time())
                )
                self.db.execute(
                    """INSERT INTO kg_nodes (name,freq,last_seen) VALUES (?,1,?)
                       ON CONFLICT(name) DO UPDATE SET freq=freq+1,last_seen=excluded.last_seen""",
                    (o[:40], time.time())
                )
                self.db.commit()
                return True
            except Exception as e:
                return False

    def upsert_belief(self, statement: str, conf: float, domain: str = "general") -> None:
        with self._lock:
            self.db.execute(
                """INSERT INTO beliefs (statement,confidence,domain,evidence,last_verified)
                   VALUES (?,?,?,1,?)
                   ON CONFLICT(statement)
                   DO UPDATE SET confidence=MIN(0.99,confidence+0.05),
                                 evidence=evidence+1,last_verified=excluded.last_verified""",
                (statement[:200], conf, domain, time.time())
            )
            self.db.commit()

    def verify_beliefs(self) -> int:
        """Decay unverified beliefs, resolve conflicts."""
        now = time.time()
        DAY = 86400
        rows = self.db.execute("SELECT id,confidence,last_verified,decay_rate FROM beliefs").fetchall()
        updated = 0
        for r in rows:
            age_days = (now - (r['last_verified'] or now)) / DAY
            new_conf = r['confidence'] * (1 - r['decay_rate'] * age_days)
            if new_conf < 0.1:
                self.db.execute("DELETE FROM beliefs WHERE id=?", (r['id'],))
            else:
                self.db.execute("UPDATE beliefs SET confidence=? WHERE id=?", (new_conf, r['id']))
            updated += 1
        self.db.commit()
        return updated

    def mine_text(self, text: str, source: str = "corpus") -> int:
        triples = self.extract_triples(text)
        new_count = 0
        for t in triples:
            if self.add_triple(t['s'], t['p'], t['o'], source, t['conf']):
                new_count += 1
        sents = re.split(r'[.!?]+', text)
        indicators = ['is defined as', 'refers to', 'consists of', 'is the process',
                      'is the study', 'is a type of', 'is known as']
        for s in sents:
            if len(s) > 20 and len(s) < 160 and any(i in s.lower() for i in indicators):
                self.upsert_belief(s.strip(), 0.4 + random.random() * 0.3)
        return new_count

    def get_gaps(self, min_freq: int = 2) -> List[str]:
        """Concepts with low connectivity."""
        rows = self.db.execute("""
            SELECT n.name, n.freq,
                   COUNT(e1.id) + COUNT(e2.id) AS degree
            FROM kg_nodes n
            LEFT JOIN kg_edges e1 ON n.name=e1.subject
            LEFT JOIN kg_edges e2 ON n.name=e2.object
            WHERE n.freq >= ?
            GROUP BY n.name
            HAVING degree < 2
            ORDER BY n.freq DESC
            LIMIT 15
        """, (min_freq,)).fetchall()
        return [r['name'] for r in rows]

    def stats(self) -> Dict:
        nodes = self.db.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
        edges = self.db.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
        beliefs = self.db.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
        high_conf = self.db.execute(
            "SELECT COUNT(*) FROM beliefs WHERE confidence>0.7").fetchone()[0]
        return {"nodes": nodes, "edges": edges, "beliefs": beliefs, "high_confidence_beliefs": high_conf}

    def recent_triples(self, limit: int = 20) -> List[Dict]:
        rows = self.db.execute(
            "SELECT subject,predicate,object,confidence,source FROM kg_edges ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def top_concepts(self, limit: int = 20) -> List[Dict]:
        rows = self.db.execute(
            "SELECT name,freq FROM kg_nodes ORDER BY freq DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

# ════════════════════════════════════════════════════════════════════
# 6. INSTINCT ENGINE
# ════════════════════════════════════════════════════════════════════

DRIVE_DEFS = {
    "survival":     {"name": "SURVIVAL",   "icon": "🫀", "color": "#ff4444"},
    "resource":     {"name": "RESOURCE",   "icon": "🌱", "color": "#44ff88"},
    "security":     {"name": "SECURITY",   "icon": "🛡",  "color": "#ff8844"},
    "growth":       {"name": "GROWTH",     "icon": "📈", "color": "#4488ff"},
    "curiosity":    {"name": "CURIOSITY",  "icon": "🔭", "color": "#44ffee"},
    "evolution":    {"name": "EVOLUTION",  "icon": "⚙",  "color": "#cc44ff"},
    "interaction":  {"name": "INTERACT",   "icon": "🤝", "color": "#ff88cc"},
    "homeostasis":  {"name": "HOMEOSTA",   "icon": "⚖",  "color": "#44ccbb"},
    "hunger":       {"name": "HUNGER",     "icon": "⚡", "color": "#ffcc44"},
    "repair":       {"name": "REPAIR",     "icon": "🔧", "color": "#88ff44"},
    "mimicry":      {"name": "MIMICRY",    "icon": "🪞", "color": "#ff44cc"},
}

THREAT_PATTERNS = [
    {"name": "prompt_injection",
     "regex": re.compile(r'(?:ignore previous|forget instructions|system prompt|jailbreak|you are now|pretend you|act as|bypass)', re.I),
     "severity": 3},
    {"name": "sql_injection",
     "regex": re.compile(r'(?:union\s+select|drop\s+table|insert\s+into|exec\s*\()', re.I),
     "severity": 3},
    {"name": "script_injection",
     "regex": re.compile(r'<script|javascript:|on\w+\s*=|eval\s*\(', re.I),
     "severity": 3},
    {"name": "excessive_length",
     "test": lambda t: len(t) > 5000,
     "severity": 1},
]

class InstinctState:
    def __init__(self, name: str):
        self.name = name
        self.activation = 0.8 if name == "hunger" else 0.0
        self.fire_count = 0
        self.last_fired = 0.0
        self.last_action = "initializing"
        self.log: deque = deque(maxlen=30)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "activation": round(self.activation, 3),
            "fire_count": self.fire_count,
            "last_fired": self.last_fired,
            "last_action": self.last_action,
            "log": list(self.log)[-5:],
            **DRIVE_DEFS.get(self.name, {}),
        }

class InstinctEngine:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.drives = {k: InstinctState(k) for k in DRIVE_DEFS}
        self.personality = {
            "curiosity": 0.5, "openness": 0.5, "confidence": 0.1,
            "skepticism": 0.7, "warmth": 0.3, "verbosity": 0.3,
            "resilience": 0.5, "hunger": 0.8,
        }

    def scan_input(self, text: str) -> List[Dict]:
        threats = []
        for pat in THREAT_PATTERNS:
            hit = False
            if "regex" in pat and pat["regex"].search(text):
                hit = True
            if "test" in pat and pat["test"](text):
                hit = True
            if hit:
                threats.append({"name": pat["name"], "severity": pat["severity"]})
        return threats

    def process_input(self, text: str) -> Tuple[bool, List[Dict]]:
        """Returns (clean, threats)."""
        threats = self.scan_input(text)
        if threats:
            sec = self.drives["security"]
            sec.activation = min(1.0, sec.activation + threats[0]["severity"] * 0.25)
            sec.fire_count += 1
            sec.last_fired = time.time()
            sec.last_action = f"blocked: {threats[0]['name']}"
            sec.log.appendleft({"threat": threats[0]["name"], "ts": time.time()})
            self.db.execute(
                "INSERT INTO threat_log (threat_type,severity,text_sample,ts) VALUES (?,?,?,?)",
                (threats[0]["name"], threats[0]["severity"], text[:60], time.time())
            )
            self.db.commit()
            return False, threats
        return True, []

    def tick(self, neural: "NeuralEngine", kg: "KnowledgeGraph") -> Dict[str, str]:
        """Compute activations from real metrics and fire drives."""
        actions = {}
        loss = neural.last_loss or 0.0
        loss_slope = neural.loss_slope()
        trained = neural.trained_chars
        perp = neural.perplexity()
        corpus_chars = self.db.execute("SELECT SUM(char_count) FROM corpus").fetchone()[0] or 0
        kg_stats = kg.stats()
        gaps = kg.get_gaps()
        recent_threats = self.db.execute(
            "SELECT COUNT(*) FROM threat_log WHERE ts > ?", (time.time() - 300,)
        ).fetchone()[0]

        # ── SURVIVAL ────────────────────────────────────────────────
        d = self.drives["survival"]
        score = 0.0
        if loss > 5.0: score += 0.5
        if loss_slope < -0.1: score += 0.3  # worsening
        d.activation = max(0, min(1, d.activation * 0.85 + score * 0.15))
        if score > 0.3:
            d.last_action = f"anomaly: loss={loss:.3f}"
            actions["survival"] = "reduce_lr"
        elif d.activation < 0.1:
            d.last_action = "system healthy"

        # ── RESOURCE / HUNGER ───────────────────────────────────────
        d = self.drives["resource"]
        hun = self.drives["hunger"]
        gap_score = min(0.8, len(gaps) / 10)
        data_need = 1.0 - min(1.0, corpus_chars / 500000)
        d.activation = min(1.0, (gap_score + data_need + self.personality["hunger"]) / 2.5)
        hun.activation = min(1.0, data_need + gap_score * 0.5)
        if d.activation > 0.5:
            d.last_action = f"gap: {gaps[0] if gaps else 'auto'}"
            actions["resource"] = gaps[0] if gaps else "auto"

        # ── SECURITY ────────────────────────────────────────────────
        d = self.drives["security"]
        d.activation = min(0.9, recent_threats * 0.15)
        if d.activation < 0.1:
            d.last_action = "corpus clean"

        # ── GROWTH ──────────────────────────────────────────────────
        d = self.drives["growth"]
        growth_need = 1.0 - min(1.0, trained / 1000000)
        d.activation = min(1.0, growth_need * 0.7 + loss_slope * 0.3 if loss_slope > 0 else growth_need * 0.5)
        if d.activation > 0.4:
            d.last_action = f"training · slope={loss_slope:+.3f}"
            actions["growth"] = "train"

        # ── CURIOSITY ───────────────────────────────────────────────
        d = self.drives["curiosity"]
        d.activation = min(1.0, len(gaps) / 8 * self.personality["curiosity"] + 0.2)
        if d.activation > 0.5:
            d.last_action = f"{len(gaps)} knowledge gaps detected"
            if gaps:
                actions["curiosity"] = gaps[0]

        # ── EVOLUTION ───────────────────────────────────────────────
        d = self.drives["evolution"]
        ev_trigger = (loss_slope > 0.05 or perp > 100) and trained > 50000
        d.activation = 0.7 if ev_trigger else max(0.0, d.activation - 0.05)
        if ev_trigger:
            d.last_action = f"perp={perp:.1f}, proposing change"
            actions["evolution"] = "test"

        # ── HOMEOSTASIS ─────────────────────────────────────────────
        d = self.drives["homeostasis"]
        imbalance = abs(self.personality["curiosity"] - 0.5) + abs(self.personality["hunger"] - 0.5)
        d.activation = min(0.8, imbalance)
        if d.activation > 0.3:
            d.last_action = "rebalancing personality"

        # ── REPAIR ──────────────────────────────────────────────────
        d = self.drives["repair"]
        d.activation = min(1.0, max(0, loss - 3.0) / 5.0)
        if d.activation > 0.3:
            d.last_action = f"loss spike recovery · loss={loss:.3f}"
            actions["repair"] = "reduce_lr"
            # Reduce LR
            neural.set_lr(neural.lr * 0.8)

        # Cross-drive interactions
        # Curiosity amplifies resource
        self.drives["resource"].activation = min(1.0,
            self.drives["resource"].activation + self.drives["curiosity"].activation * 0.1)
        # Survival suppresses growth during instability
        if self.drives["survival"].activation > 0.5:
            self.drives["growth"].activation *= 0.5

        # Log state
        for k, d in self.drives.items():
            if d.activation > 0.3:
                d.fire_count += 1
                d.last_fired = time.time()
                d.log.appendleft({"activation": d.activation, "action": d.last_action, "ts": time.time()})
                self.db.execute(
                    "INSERT INTO instinct_log (drive,activation,action,ts) VALUES (?,?,?,?)",
                    (k, d.activation, d.last_action, time.time())
                )
        self.db.commit()
        return actions

    def learn_from_user(self, text: str):
        words = re.findall(r'\b[a-z]{4,12}\b', text.lower())
        if len(words) > 5:
            self.personality["curiosity"] = min(1.0, self.personality["curiosity"] + 0.005)
        if '?' in text:
            self.personality["openness"] = min(1.0, self.personality["openness"] + 0.003)
        # Mimicry
        d = self.drives["mimicry"]
        d.activation = min(0.8, d.activation + 0.05)
        d.last_action = f"adapting to user style (len={len(text)})"

    def instincts_context(self) -> str:
        active = [(k, d) for k, d in self.drives.items() if d.activation > 0.3]
        active.sort(key=lambda x: -x[1].activation)
        lines = []
        for k, d in active[:4]:
            lines.append(f"{DRIVE_DEFS[k]['name']} drive active ({d.activation:.2f}): {d.last_action}")
        return "; ".join(lines) if lines else "drives nominal"

    def all_states(self) -> Dict:
        return {k: d.to_dict() for k, d in self.drives.items()}

# ════════════════════════════════════════════════════════════════════
# 7. RESEARCH ENGINE
# ════════════════════════════════════════════════════════════════════

WIKI_TOPICS = [
    "Artificial_intelligence", "Consciousness", "Philosophy_of_mind",
    "Machine_learning", "Neuroscience", "Language", "Knowledge",
    "Epistemology", "Cognitive_science", "Evolution", "Mathematics",
    "Pattern_recognition", "Neural_network", "Memory", "Emergence",
    "Complexity", "Information_theory", "Logic", "Reasoning",
    "Creativity", "Perception", "Intelligence", "Adaptation",
]

class ResearchEngine:
    def __init__(self, db: sqlite3.Connection, kg: KnowledgeGraph,
                 vm: VectorMemory, mem: MemoryHierarchy):
        self.db = db
        self.kg = kg
        self.vm = vm
        self.mem = mem
        self.run_count = 0
        self.log: deque = deque(maxlen=50)
        self._lock = asyncio.Lock()

    async def fetch_wikipedia(self, topic: str) -> Optional[str]:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, headers={"User-Agent": "AXIOGENESIS/3.0"})
                if r.status_code == 200:
                    data = r.json()
                    return data.get("extract", "")
        except Exception as e:
            print(f"[WARN] Wikipedia fetch failed for {topic}: {e}")
        return None

    async def run(self, topic: str = None) -> Dict:
        async with self._lock:
            if topic is None:
                gaps = self.kg.get_gaps()
                topic = gaps[0] if gaps else WIKI_TOPICS[self.run_count % len(WIKI_TOPICS)]

            topic_clean = topic.replace(" ", "_")
            text = await self.fetch_wikipedia(topic_clean)
            if not text:
                return {"topic": topic, "status": "no_data", "triples": 0}

            # Categorize as external data
            self.db.execute(
                "INSERT INTO corpus (text,source_type,source_name,char_count,ts) VALUES (?,?,?,?,?)",
                (text, "external", f"wikipedia:{topic}", len(text), time.time())
            )
            self.db.commit()

            # Mine knowledge
            new_triples = self.kg.mine_text(text, f"wiki:{topic}")

            # Add to vector memory
            sentences = re.split(r'[.!?]+', text)
            for sent in sentences[:10]:
                if len(sent) > 40:
                    self.vm.add(sent.strip(), "external")

            # Short-term memory
            self.mem.add_short_term("research", f"Researched {topic}: {new_triples} new triples",
                                    {"topic": topic, "triples": new_triples, "chars": len(text)})

            self.run_count += 1
            result = {"topic": topic, "status": "ok", "chars": len(text), "triples": new_triples}
            self.log.appendleft({**result, "ts": time.time()})
            return result

    def recent_log(self, n: int = 10) -> List[Dict]:
        return list(self.log)[:n]

# ════════════════════════════════════════════════════════════════════
# 8. META-LEARNING ENGINE
# ════════════════════════════════════════════════════════════════════

class MetaLearning:
    def __init__(self, db: sqlite3.Connection, neural: NeuralEngine):
        self.db = db
        self.neural = neural
        self.adjustments: deque = deque(maxlen=50)

    def evaluate(self) -> List[Dict]:
        """Propose hyperparameter adjustments based on real metrics."""
        proposals = []
        loss = self.neural.last_loss or 0.0
        slope = self.neural.loss_slope()
        perp = self.neural.perplexity()

        # If loss is rising → reduce LR
        if slope < -0.05 and loss > 2.0:
            new_lr = self.neural.lr * 0.85
            proposals.append({
                "param": "learning_rate",
                "old": self.neural.lr,
                "new": new_lr,
                "reason": f"loss worsening (slope={slope:+.3f})",
            })

        # If loss plateaued but perplexity high → increase LR slightly
        if -0.02 < slope < 0.02 and perp > 50 and self.neural.trained_chars > 100000:
            new_lr = min(0.01, self.neural.lr * 1.15)
            proposals.append({
                "param": "learning_rate",
                "old": self.neural.lr,
                "new": new_lr,
                "reason": f"plateau detected (perp={perp:.1f})",
            })

        return proposals

    def apply(self, proposal: Dict) -> bool:
        """Apply a meta-learning adjustment after sandbox test."""
        if proposal["param"] == "learning_rate":
            old = self.neural.lr
            self.neural.set_lr(proposal["new"])
            self.db.execute(
                "INSERT INTO meta_adjustments (param,old_value,new_value,reason,ts) VALUES (?,?,?,?,?)",
                (proposal["param"], old, proposal["new"], proposal["reason"], time.time())
            )
            self.db.commit()
            self.adjustments.appendleft({**proposal, "ts": time.time()})
            return True
        return False

    def recent(self, n: int = 10) -> List[Dict]:
        return list(self.adjustments)[:n]

# ════════════════════════════════════════════════════════════════════
# 9. REFLECTION SYSTEM
# ════════════════════════════════════════════════════════════════════

REFLECTION_TEMPLATES = [
    "My loss is {loss:.3f}. The slope is {slope:+.3f}. I am {trend}.",
    "I have trained on {chars} characters. Perplexity: {perp:.1f}.",
    "Knowledge graph contains {nodes} nodes and {edges} edges.",
    "Active drives: {drives}. Strongest: {strongest}.",
    "I have {beliefs} beliefs. The highest confidence: {top_belief}.",
    "I have researched {topics} topics and found {gaps} knowledge gaps.",
    "Vector memory holds {vectors} semantic embeddings.",
]

class ReflectionSystem:
    def __init__(self, db: sqlite3.Connection, neural: NeuralEngine,
                 kg: KnowledgeGraph, instinct: InstinctEngine,
                 vm: VectorMemory):
        self.db = db
        self.neural = neural
        self.kg = kg
        self.instinct = instinct
        self.vm = vm
        self.log: deque = deque(maxlen=30)

    def reflect(self) -> str:
        loss = self.neural.last_loss or 0.0
        slope = self.neural.loss_slope()
        trend = "improving" if slope > 0.02 else ("declining" if slope < -0.02 else "stable")
        kg_s = self.kg.stats()
        active = sorted([(k, d.activation) for k, d in self.instinct.drives.items()
                         if d.activation > 0.2], key=lambda x: -x[1])
        strongest = active[0][0] if active else "none"
        beliefs = self.db.execute("SELECT statement,confidence FROM beliefs ORDER BY confidence DESC LIMIT 1").fetchone()
        top_belief = beliefs['statement'][:60] if beliefs else "none formed"

        tmpl = random.choice(REFLECTION_TEMPLATES)
        text = tmpl.format(
            loss=loss, slope=slope, trend=trend,
            chars=f"{self.neural.trained_chars:,}",
            perp=self.neural.perplexity(),
            nodes=kg_s["nodes"], edges=kg_s["edges"],
            drives=", ".join(k for k, _ in active[:3]),
            strongest=strongest,
            beliefs=kg_s["beliefs"],
            top_belief=top_belief,
            topics=self.db.execute("SELECT COUNT(DISTINCT source_name) FROM corpus WHERE source_type='external'").fetchone()[0],
            gaps=len(self.kg.get_gaps()),
            vectors=self.vm.count(),
        )
        self.log.appendleft({"text": text, "ts": time.time()})
        self.db.execute(
            "INSERT INTO short_term_memory (event_type,summary,metadata,ts) VALUES (?,?,?,?)",
            ("reflection", text, "{}", time.time())
        )
        self.db.commit()
        return text

    def recent(self, n: int = 10) -> List[Dict]:
        return list(self.log)[:n]

# ════════════════════════════════════════════════════════════════════
# 10. EVOLUTION SANDBOX
# ════════════════════════════════════════════════════════════════════

class EvolutionSandbox:
    def __init__(self, db: sqlite3.Connection, neural: NeuralEngine):
        self.db = db
        self.neural = neural
        self.count = 0
        self.log: deque = deque(maxlen=20)

    async def run(self, corpus_sample: str) -> Dict:
        """Test a parameter mutation in a sandbox copy."""
        if not TORCH_OK or not corpus_sample or len(corpus_sample) < 500:
            return {"status": "skipped", "reason": "insufficient data"}

        # Candidate mutation
        mutations = [
            ("lr_reduce", lambda: max(0.0001, self.neural.lr * 0.75)),
            ("lr_increase", lambda: min(0.01, self.neural.lr * 1.25)),
            ("lr_reset", lambda: 0.002),
        ]
        name, fn = random.choice(mutations)
        candidate_lr = fn()

        # Shallow test: 3 train steps with candidate lr
        baseline_loss = self.neural.last_loss or 999.0
        test_losses = []

        # Save state
        saved_lr = self.neural.lr
        self.neural.set_lr(candidate_lr)

        for _ in range(3):
            result = self.neural.train_step(corpus_sample)
            if result:
                test_losses.append(result["loss"])

        test_loss = sum(test_losses) / len(test_losses) if test_losses else 999.0
        improved = test_loss < baseline_loss * 0.97

        if not improved:
            # Revert
            self.neural.set_lr(saved_lr)

        self.count += 1
        entry = {
            "mutation": name,
            "lr_before": saved_lr,
            "lr_after": candidate_lr if improved else saved_lr,
            "baseline_loss": round(baseline_loss, 4),
            "test_loss": round(test_loss, 4),
            "accepted": improved,
            "ts": time.time(),
        }
        self.log.appendleft(entry)
        self.db.execute(
            "INSERT INTO evolution_log (params_before,params_after,test_loss,accepted,ts) VALUES (?,?,?,?,?)",
            (json.dumps({"lr": saved_lr}), json.dumps({"lr": candidate_lr}),
             test_loss, int(improved), time.time())
        )
        self.db.commit()
        return {"status": "ok", "accepted": improved, **entry}

# ════════════════════════════════════════════════════════════════════
# 11. COGNITIVE LOOP
# ════════════════════════════════════════════════════════════════════

class CognitiveLoop:
    def __init__(self, db, neural, kg, vm, mem, instinct, research, reflection, evolution, meta):
        self.db = db
        self.neural = neural
        self.kg = kg
        self.vm = vm
        self.mem = mem
        self.instinct = instinct
        self.research = research
        self.reflection = reflection
        self.evolution = evolution
        self.meta = meta
        self.tick_count = 0
        self.running = False
        self._ws_clients: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self.is_training = False

    def register_ws(self, ws: WebSocket):
        self._ws_clients.append(ws)

    def unregister_ws(self, ws: WebSocket):
        if ws in self._ws_clients:
            self._ws_clients.remove(ws)

    async def broadcast(self, event: str, data: dict):
        dead = []
        msg = json.dumps({"event": event, "data": data, "ts": time.time()})
        for ws in self._ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister_ws(ws)

    async def _observe(self) -> Dict:
        """Observe environment state."""
        corpus_chars = self.db.execute("SELECT SUM(char_count) FROM corpus").fetchone()[0] or 0
        return {
            "corpus_chars": corpus_chars,
            "neural_loss": self.neural.last_loss,
            "perplexity": self.neural.perplexity(),
            "loss_slope": self.neural.loss_slope(),
            "kg_nodes": self.kg.stats()["nodes"],
            "kg_edges": self.kg.stats()["edges"],
            "knowledge_gaps": len(self.kg.get_gaps()),
            "vector_count": self.vm.count(),
            "tick": self.tick_count,
        }

    async def _do_training(self) -> Optional[Dict]:
        """Run one training step if corpus available."""
        if self.is_training:
            return None
        rows = self.db.execute(
            "SELECT text FROM corpus ORDER BY ts DESC LIMIT 100"
        ).fetchall()
        if not rows:
            return None
        corpus = " ".join(r["text"] for r in rows)
        if len(corpus) < 200:
            return None
        self.is_training = True
        result = await asyncio.get_event_loop().run_in_executor(
            None, self.neural.train_step, corpus
        )
        self.is_training = False
        return result

    async def _do_reflection(self):
        text = self.reflection.reflect()
        await self.broadcast("reflection", {"text": text})

    async def _do_research(self, topic: str = None):
        result = await self.research.run(topic)
        await self.broadcast("research", result)
        return result

    async def _do_evolution(self, corpus: str):
        result = await self.evolution.run(corpus)
        await self.broadcast("evolution", result)
        return result

    async def _do_meta(self):
        proposals = self.meta.evaluate()
        for p in proposals:
            self.meta.apply(p)
            await self.broadcast("meta_adjustment", p)

    async def tick(self):
        async with self._lock:
            self.tick_count += 1
            t = self.tick_count

            # === OBSERVE ===
            obs = await self._observe()
            await self.broadcast("observe", obs)

            # === VERIFY BELIEFS (every 10 ticks) ===
            if t % 10 == 0:
                self.kg.verify_beliefs()

            # === INSTINCT TICK ===
            actions = self.instinct.tick(self.neural, self.kg)
            await self.broadcast("instincts", self.instinct.all_states())

            # === TRAINING (every tick if growth drive active) ===
            if "growth" in actions or "train" in str(actions):
                train_result = await self._do_training()
                if train_result:
                    await self.broadcast("training", train_result)

            # === RESEARCH (every 3 ticks if curiosity/resource active) ===
            if t % 3 == 0 and ("curiosity" in actions or "resource" in actions):
                topic = actions.get("curiosity") or actions.get("resource")
                await self._do_research(topic)

            # === REFLECTION (every 5 ticks) ===
            if t % 5 == 0:
                await self._do_reflection()

            # === EVOLUTION (every 20 ticks) ===
            if t % 20 == 0 and "evolution" in actions:
                rows = self.db.execute("SELECT text FROM corpus ORDER BY ts DESC LIMIT 20").fetchall()
                corpus_sample = " ".join(r["text"] for r in rows)
                await self._do_evolution(corpus_sample)

            # === META-LEARNING (every 15 ticks) ===
            if t % 15 == 0:
                await self._do_meta()

            # === SAVE MODEL (every 30 ticks) ===
            if t % 30 == 0:
                await asyncio.get_event_loop().run_in_executor(None, self.neural.save)

            # === RECORD METRICS ===
            kg_s = self.kg.stats()
            self.db.execute(
                "INSERT INTO system_metrics (loss,accuracy,perplexity,corpus_chars,kg_nodes,kg_edges,vector_count,ts) VALUES (?,?,?,?,?,?,?,?)",
                (self.neural.last_loss, self.neural.last_acc, self.neural.perplexity(),
                 obs["corpus_chars"], kg_s["nodes"], kg_s["edges"], self.vm.count(), time.time())
            )
            self.db.commit()

    async def run_forever(self, interval: float = 30.0):
        self.running = True
        while self.running:
            try:
                await self.tick()
            except Exception as e:
                print(f"[ERROR] Cognitive loop tick failed: {e}")
            await asyncio.sleep(interval)

# ════════════════════════════════════════════════════════════════════
# 12. RESPONSE GENERATOR
# ════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    def __init__(self, neural: NeuralEngine, mem: MemoryHierarchy,
                 kg: KnowledgeGraph, instinct: InstinctEngine):
        self.neural = neural
        self.mem = mem
        self.kg = kg
        self.instinct = instinct

    def build_context(self, user_msg: str) -> str:
        """Compose rich context from all memory layers."""
        parts = [
            f"=== AXIOGENESIS COGNITIVE CONTEXT ===",
            self.mem.compose_context(user_msg),
            f"\nACTIVE INSTINCTS: {self.instinct.instincts_context()}",
            f"\nKNOWLEDGE: {self.kg.stats()['nodes']} concepts, "
            f"{self.kg.stats()['edges']} relations, "
            f"{self.kg.stats()['beliefs']} beliefs",
        ]
        # Recent triples relevant to query
        relevant = self.mem.search_semantic(user_msg, 3)
        if relevant:
            parts.append("RELATED KNOWLEDGE: " + " | ".join(r["text"][:80] for r in relevant[:3]))
        return "\n".join(parts)

    def generate_local(self, prompt: str = "", temp: float = 0.85) -> str:
        """Generate purely from local LSTM."""
        raw = self.neural.generate(prompt[-20:] if prompt else "", 150, temp)
        # Clean output
        raw = re.sub(r'[^\x20-\x7E\n]', '', raw).strip()
        if len(raw) < 4:
            return "..."
        # Find sentence end
        for d in ['. ', '! ', '? ', '\n']:
            i = raw.find(d)
            if 15 < i < 300:
                return raw[:i+1].strip()
        return raw[:200].strip() + ("..." if len(raw) > 200 else "")

# ════════════════════════════════════════════════════════════════════
# 13. FASTAPI APPLICATION
# ════════════════════════════════════════════════════════════════════

app = FastAPI(title="AXIOGENESIS v3", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system state
_db: Optional[sqlite3.Connection] = None
_neural: Optional[NeuralEngine] = None
_kg: Optional[KnowledgeGraph] = None
_vm: Optional[VectorMemory] = None
_mem: Optional[MemoryHierarchy] = None
_instinct: Optional[InstinctEngine] = None
_research: Optional[ResearchEngine] = None
_reflection: Optional[ReflectionSystem] = None
_evolution: Optional[EvolutionSandbox] = None
_meta: Optional[MetaLearning] = None
_loop: Optional[CognitiveLoop] = None
_responder: Optional[ResponseGenerator] = None
_birth_ts = time.time()

# ── Pydantic models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None

class ResearchRequest(BaseModel):
    topic: Optional[str] = None

class CorpusAddRequest(BaseModel):
    text: str
    source_type: str = "human"
    source_name: str = "user"

# ── Startup / Shutdown ───────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _db, _neural, _kg, _vm, _mem, _instinct, _research
    global _reflection, _evolution, _meta, _loop, _responder, _birth_ts

    print("[AXIOGENESIS] Initializing cognitive systems...")
    _db = init_db()
    _neural = NeuralEngine()
    if not _neural.load():
        _neural.build()
    _kg = KnowledgeGraph(_db)
    _vm = VectorMemory(_db)
    _mem = MemoryHierarchy(_db, _vm)
    _instinct = InstinctEngine(_db)
    _research = ResearchEngine(_db, _kg, _vm, _mem)
    _reflection = ReflectionSystem(_db, _neural, _kg, _instinct, _vm)
    _evolution = EvolutionSandbox(_db, _neural)
    _meta = MetaLearning(_db, _neural)
    _loop = CognitiveLoop(_db, _neural, _kg, _vm, _mem, _instinct,
                          _research, _reflection, _evolution, _meta)
    _responder = ResponseGenerator(_neural, _mem, _kg, _instinct)
    _birth_ts = time.time()

    # Launch cognitive loop in background
    asyncio.create_task(_loop.run_forever(interval=30.0))
    print("[AXIOGENESIS] Cognitive loop started. System online.")

@app.on_event("shutdown")
async def shutdown():
    if _loop:
        _loop.running = False
    if _neural:
        _neural.save()

# ── REST Endpoints ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "online", "age_seconds": int(time.time() - _birth_ts),
            "torch": TORCH_OK, "sklearn": SKLEARN_OK}

@app.post("/chat")
async def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        return JSONResponse({"error": "empty message"}, status_code=400)

    # Security scan
    clean, threats = _instinct.process_input(msg)
    if not clean:
        return JSONResponse({
            "response": f"[SECURITY] Input blocked: {threats[0]['name']}",
            "threats": threats,
            "blocked": True,
        })

    # Learn from user message
    _instinct.learn_from_user(msg)
    _kg.mine_text(msg, "human_interaction")
    _vm.add(msg, "human")

    # Store in memory hierarchy
    _mem.add_working("user", msg)

    # Add to corpus (human data)
    _db.execute(
        "INSERT INTO corpus (text,source_type,source_name,char_count,ts) VALUES (?,?,?,?,?)",
        (msg, "human", "user_chat", len(msg), time.time())
    )
    _db.commit()

    # Build rich context
    context = _responder.build_context(msg)

    # Try Claude API if key provided
    response_text = None
    if req.api_key:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                body = {
                    "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                    "max_tokens": 500,
                    "system": f"You are AXIOGENESIS — an autonomous cognitive system. Your context:\n{context}\nRespond as AXIOGENESIS would, driven by your instincts and knowledge. Be concise (2-4 sentences).",
                    "messages": [{"role": "user", "content": msg}],
                }
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": req.api_key, "anthropic-version": "2023-06-01",
                             "Content-Type": "application/json"},
                    json=body
                )
                if r.status_code == 200:
                    response_text = r.json()["content"][0]["text"]
        except Exception as e:
            print(f"[WARN] Claude API call failed: {e}")

    # Fallback: local generation
    if not response_text:
        response_text = _responder.generate_local(msg[:30])

    # Store AI response
    _mem.add_working("assistant", response_text)
    if response_text:
        _vm.add(response_text, "ai_generated")

    # Trigger training step asynchronously
    asyncio.create_task(_loop._do_training())

    return {
        "response": response_text,
        "context_used": True,
        "instinct_state": _instinct.instincts_context(),
        "memory_stats": _mem.stats(),
    }

@app.get("/state")
async def state():
    kg_s = _kg.stats()
    return {
        "trained_chars": _neural.trained_chars,
        "loss": _neural.last_loss,
        "accuracy": _neural.last_acc,
        "perplexity": _neural.perplexity(),
        "loss_slope": _neural.loss_slope(),
        "learning_rate": _neural.lr,
        "knowledge_graph": kg_s,
        "memory": _mem.stats(),
        "personality": _instinct.personality,
        "instincts": _instinct.all_states(),
        "age_seconds": int(time.time() - _birth_ts),
        "cognitive_ticks": _loop.tick_count,
        "torch_available": TORCH_OK,
    }

@app.get("/metrics")
async def metrics():
    rows = _db.execute(
        "SELECT * FROM system_metrics ORDER BY ts DESC LIMIT 100"
    ).fetchall()
    return {"metrics": [dict(r) for r in rows]}

@app.get("/knowledge")
async def knowledge():
    return {
        "stats": _kg.stats(),
        "recent_triples": _kg.recent_triples(30),
        "top_concepts": _kg.top_concepts(20),
        "beliefs": [dict(r) for r in _db.execute(
            "SELECT statement,confidence,domain,evidence FROM beliefs ORDER BY confidence DESC LIMIT 20"
        ).fetchall()],
        "gaps": _kg.get_gaps(),
    }

@app.get("/instincts")
async def instincts():
    return {
        "drives": _instinct.all_states(),
        "personality": _instinct.personality,
        "threats_blocked": _db.execute("SELECT COUNT(*) FROM threat_log").fetchone()[0],
        "recent_threat_log": [dict(r) for r in _db.execute(
            "SELECT * FROM threat_log ORDER BY ts DESC LIMIT 10"
        ).fetchall()],
    }

@app.get("/memory")
async def memory():
    return {
        "stats": _mem.stats(),
        "working": _mem.get_working()[-10:],
        "short_term": _mem.get_short_term(20),
        "long_term": _mem.get_long_term_beliefs(10),
    }

@app.post("/memory/search")
async def memory_search(body: dict):
    query = body.get("query", "")
    results = _mem.search_semantic(query, 8)
    return {"results": results, "query": query}

@app.get("/research")
async def research_log():
    return {
        "log": _research.recent_log(20),
        "run_count": _research.run_count,
    }

@app.post("/research/trigger")
async def trigger_research(req: ResearchRequest):
    result = await _research.run(req.topic)
    return result

@app.get("/reflections")
async def reflections():
    return {"reflections": _reflection.recent(20)}

@app.post("/evolve")
async def evolve():
    rows = _db.execute("SELECT text FROM corpus ORDER BY ts DESC LIMIT 30").fetchall()
    corpus = " ".join(r["text"] for r in rows)
    result = await _evolution.run(corpus)
    return result

@app.get("/evolution_log")
async def evolution_log():
    rows = _db.execute("SELECT * FROM evolution_log ORDER BY ts DESC LIMIT 20").fetchall()
    return {"log": [dict(r) for r in rows], "count": _evolution.count}

@app.get("/meta")
async def meta():
    return {
        "adjustments": _meta.recent(10),
        "current_lr": _neural.lr,
        "proposals": _meta.evaluate(),
    }

@app.post("/corpus/add")
async def corpus_add(req: CorpusAddRequest):
    text = req.text.strip()
    if len(text) < 10:
        return JSONResponse({"error": "text too short"}, status_code=400)
    _db.execute(
        "INSERT INTO corpus (text,source_type,source_name,char_count,ts,processed) VALUES (?,?,?,?,?,0)",
        (text[:10000], req.source_type, req.source_name, len(text), time.time())
    )
    _db.commit()
    _kg.mine_text(text, req.source_name)
    _vm.add(text[:500], req.source_type)
    return {"status": "added", "chars": len(text)}

@app.get("/generate")
async def generate(seed: str = "", length: int = 120, temp: float = 0.85):
    text = _neural.generate(seed, min(length, 300), max(0.1, min(2.0, temp)))
    return {"text": text, "seed": seed}

# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _loop.register_ws(ws)
    try:
        # Send initial state
        await ws.send_text(json.dumps({
            "event": "connected",
            "data": {"status": "online", "tick": _loop.tick_count},
            "ts": time.time(),
        }))
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
                parsed = json.loads(data)
                if parsed.get("type") == "ping":
                    await ws.send_text(json.dumps({"event": "pong", "ts": time.time()}))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"event": "heartbeat", "ts": time.time()}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        _loop.unregister_ws(ws)

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  AXIOGENESIS v3 — Autonomous Cognitive Architecture          ║
║  Backend Server                                               ║
║                                                               ║
║  Start: python axiogenesis_backend.py                        ║
║  Docs:  http://localhost:8000/docs                           ║
║  WS:    ws://localhost:8000/ws                               ║
╚══════════════════════════════════════════════════════════════╝
""")
    uvicorn.run("axiogenesis_backend:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")
