# AXIOGENESIS v3: Experimental Cognitive-Architecture Prototype

An experimental client-server system that combines:
- Continuous autonomous cognitive loop
- PyTorch LSTM neural engine (backend)
- 4-layer memory hierarchy (working/short-term/long-term/semantic)
- TF-IDF vector memory with cosine similarity search
- Knowledge graph with belief verification & confidence decay
- 11 instinct drives wired to real system metrics
- Meta-learning engine with evolution sandbox
- FastAPI backend + WebSocket real-time streaming
- Optional chat replies from the Claude API (you paste your own key in the browser)

---

## QUICK START

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the backend
```bash
python axiogenesis_backend.py
```
Server starts at: http://localhost:8000
API docs at:       http://localhost:8000/docs
WebSocket at:      ws://localhost:8000/ws

### 3. Open the frontend
Open `frontend/axiogenesis_v3.html` in your browser.
Enter backend URL: http://localhost:8000
Optionally enter your Claude API key (sent only to your local backend, which forwards it to Anthropic).
Click "initialize cognitive system".

---

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (HTML)                       │
│  Chat · KG Visualizer · Instinct Matrix · Memory Panel  │
│  Meta-Learning · Knowledge Verification · Vector Memory  │
└─────────────────────┬───────────────────────────────────┘
                      │ REST + WebSocket
┌─────────────────────┴───────────────────────────────────┐
│                   FASTAPI BACKEND                        │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │          CONTINUOUS COGNITIVE LOOP               │   │
│  │  observe → train → knowledge → instinct →        │   │
│  │  research → reflect → evolve → meta-learn        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Neural LSTM │  │  KG (SQLite)│  │ Vector Memory   │  │
│  │  PyTorch    │  │ triples +   │  │ TF-IDF cosine   │  │
│  │  char-level │  │ beliefs     │  │ semantic search │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Instinct   │  │  Research   │  │  Evolution      │  │
│  │  Engine     │  │  Wikipedia  │  │  Sandbox        │  │
│  │  11 drives  │  │  + corpus   │  │  param testing  │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │               SQLite (axiogenesis.db)               │ │
│  │  corpus · kg_nodes · kg_edges · beliefs ·          │ │
│  │  vector_memory · working_memory · short_term ·     │ │
│  │  threat_log · evolution_log · system_metrics       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## API ENDPOINTS

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | System health check |
| POST | /chat | Send message, get response |
| GET | /state | Full system state |
| GET | /metrics | Historical metrics (last 100) |
| GET | /knowledge | KG stats, triples, beliefs, gaps |
| GET | /instincts | All drive states |
| GET | /memory | Memory hierarchy stats |
| POST | /memory/search | Semantic vector search |
| GET | /research | Research log |
| POST | /research/trigger | Trigger research on topic |
| GET | /reflections | System reflections |
| POST | /evolve | Run evolution sandbox |
| GET | /evolution_log | Evolution history |
| GET | /meta | Meta-learning proposals & log |
| POST | /corpus/add | Add text to corpus |
| GET | /generate | Generate text from neural model |
| WS | /ws | Real-time cognitive stream |

---

## COGNITIVE LOOP EVENTS (WebSocket)

The backend broadcasts these events every 30 seconds:

- `observe` — environment state snapshot
- `training` — neural train step result (loss, accuracy)
- `instincts` — all 11 drive activation values
- `research` — Wikipedia research result
- `reflection` — system self-reflection text
- `meta_adjustment` — hyperparameter change
- `evolution` — sandbox test result
- `heartbeat` — keepalive

---

## MEMORY HIERARCHY

| Layer | Storage | Capacity | Retrieval |
|-------|---------|----------|-----------|
| Working | RAM deque | 20 items | FIFO |
| Short-Term | SQLite | unlimited | Recent N |
| Long-Term | SQLite | unlimited | Confidence-ranked |
| Semantic | SQLite + numpy | unlimited | Cosine similarity |

---

## INSTINCT CROSS-INTERACTIONS

- **Curiosity → Resource**: high gaps amplify resource acquisition
- **Survival → Growth**: instability suppresses training intensity
- **Hunger → Research**: low corpus triggers external data fetch
- **Security → Corpus**: suspicious inputs blocked before entering storage
- **Repair → LR**: loss spikes trigger automatic learning rate reduction

---

## DATA SOURCE SEPARATION

All corpus entries are tagged with `source_type`:
- `human` — direct user messages
- `external` — Wikipedia, research APIs
- `ai_generated` — model outputs

Tagging lets human, external and generated text be told apart. The tags are stored with every entry; the training step does not yet limit how much generated text it uses.

---

## KNOWLEDGE VERIFICATION

Beliefs receive confidence scores (0–1) that:
- Increase by 0.05 each time the belief is reinforced
- Decay by `decay_rate × age_in_days` when not seen
- Are deleted when confidence drops below 0.1
- Conflicting beliefs tracked by evidence count

---

## PRODUCTION UPGRADES

To scale beyond SQLite:
```python
# PostgreSQL
DATABASE_URL = "postgresql://user:pass@host/axiogenesis"

# Neo4j for knowledge graph
NEO4J_URI = "bolt://localhost:7687"

# Redis for fast state
REDIS_URL = "redis://localhost:6379"

# Dedicated vector DB (Qdrant, Weaviate, etc.)
VECTOR_DB_URL = "http://localhost:6333"
```

---

## WHAT THIS IS (AND ISN'T)

AXIOGENESIS is a learning project that wires several software components into one loop. The "instincts" are numeric scores computed from system metrics (training loss, corpus size, knowledge gaps) that decide what the loop does next. The "reflections" are filled-in text templates, and the "beliefs" are database rows with a confidence number. The character-level LSTM is small and learns from whatever text it is given, so its output is not intelligent. The value of the project is the architecture and engineering, not the model's output.

## ENVIRONMENT

| Variable | Description |
|---|---|
| `ANTHROPIC_MODEL` | Optional. Claude model used for chat replies when a key is supplied (default `claude-sonnet-5-5`) |

## PROJECT STRUCTURE

```
axiogenesis/
├── backend/
│   ├── axiogenesis_backend.py   # FastAPI app, cognitive loop, neural engine, memory
│   └── requirements.txt
├── frontend/
│   └── axiogenesis_v3.html      # browser interface
├── .gitignore
└── README.md
```

The SQLite database (`axiogenesis.db`) and the neural weights (`axio_model.pt`) are created when you run the backend and are not stored in the repository.

## LIMITATIONS
- CORS is open (`*`) and there is no authentication, so run it locally and do not expose it to the internet.
- If you enter a Claude API key in the page, the browser stores it in `localStorage` in plain text. Use a key you can revoke, and clear it with the page's reset option on shared computers.
- The threat filter is a short list of regular expressions, not a security system.
- Research uses only the Wikipedia summary endpoint.
- Research and educational use only.
