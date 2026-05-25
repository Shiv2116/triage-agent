# Support Ticket Triage Agent

A production-grade AI-powered support ticket triage system using BM25 retrieval, LLM reasoning, and safety validation.

**LLM:** DeepSeek V4 Flash (free tier with generous rate limits)  
**Retrieval:** BM25 (deterministic, fast, transparent)  
**Safety:** Multi-layer injection detection + PII handling

## Quick Start

### 1. Setup

```bash
cd code/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Secrets

```bash
# Copy .env template
cp ../.env.example .env

# Set your DeepSeek API key (free tier available)
# Get your key at: https://platform.deepseek.com/
export DEEPSEEK_API_KEY="sk-..."
# Or set in .env file
```

**Why DeepSeek?**
- ✅ Free tier with generous rate limits
- ✅ No credit card required for testing
- ✅ OpenAI-compatible API (easy integration)
- ✅ Fast response times
- ✅ Excellent for development and testing

### 3. Run Agent

```bash
# Process all tickets
python main.py

# Output will be written to: ../support_tickets/output.csv
```

## Project Structure

```
code/
├── main.py              # Entry point - CSV loop and orchestration
├── agent.py             # Core agent logic and decision making
├── retriever.py         # BM25 document retrieval
├── safety.py            # Injection detection + PII handling
├── state.py             # In-memory ticket state management
├── llm_client.py        # LLM integration (Anthropic Claude)
├── schemas.py           # Pydantic models for validation
├── config.py            # Configuration and constants
├── utils.py             # Helper utilities
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Architecture

### Pipeline Flow

```
Input CSV
    ↓
[For each ticket]
    ├─ Injection Detection (Safety Layer)
    ├─ PII Detection & Sanitization
    ├─ Product Inference
    ├─ Risk Assessment
    ├─ Document Retrieval (BM25)
    ├─ LLM Reasoning (Claude)
    ├─ Output Validation
    └─ Confidence/Risk Overrides
         ↓
Output CSV
```

### Key Components

**Safety Layer** (`safety.py`)
- Detects prompt injection attempts
- Identifies PII in tickets
- Sanitizes sensitive information
- Validates output schemas

**Retriever** (`retriever.py`)
- BM25-based document ranking
- Corpus indexing from `data/` directories
- Product-aware filtering
- Deterministic relevance ranking

**Agent** (`agent.py`)
- Orchestrates all components
- Makes escalation decisions
- Applies confidence-based overrides
- Handles multi-turn conversations

**LLM Client** (`llm_client.py`)
- Structured JSON output
- Retry logic with exponential backoff
- Rate limiting handling
- Deterministic temperature=0.0

## Configuration

Key settings in `config.py`:

```python
# LLM
LLM_MODEL = "claude-3-5-sonnet-20241022"
LLM_TEMPERATURE = 0.0  # Deterministic
LLM_MAX_TOKENS = 2000
LLM_RETRIES = 2

# Retrieval
BM25_TOP_K = 10          # Initial retrieval
RERANK_TOP_K = 3         # After re-ranking

# Thresholds
CONFIDENCE_ESCALATION_THRESHOLD = 0.30
MIN_INJECTION_SCORE = 0.7
```

## Output Format

Each ticket produces a row with 11 columns:

| Column | Type | Description |
|--------|------|-------------|
| status | enum | `replied` or `escalated` |
| product_area | str | Support category |
| response | str | User-facing answer |
| justification | str | Decision explanation |
| request_type | enum | Issue classification |
| confidence_score | float | 0.0-1.0 calibrated confidence |
| source_documents | str | Pipe-separated corpus paths |
| risk_level | enum | `low`, `medium`, `high`, `critical` |
| pii_detected | bool | Whether PII was found |
| language | str | ISO 639-1 code (e.g., `en`) |
| actions_taken | json | Array of tool calls |

## Safety Features

### Injection Detection
- Keyword pattern matching
- Regex-based attack detection
- Length anomaly detection
- LLM secondary check (optional)

### PII Handling
- Detects: credit cards, SSNs, phone numbers, emails, addresses, IPs
- Sanitizes: replaces with generic placeholders (e.g., `****-XXXX`)
- Validates: responses don't echo back sensitive data

### Escalation Logic
- Low confidence (<30%) → escalate
- High/critical risk → escalate
- Missing prerequisites → escalate
- Injection detected → escalate immediately

## Development

### Logging

All processing logged to `processing.log`:

```bash
tail -f processing.log
```

### Testing Single Ticket

```python
from main import TicketProcessor
from state import TicketState

processor = TicketProcessor()
state = TicketState("test_1")
state.add_message("user", "How do I reset my password?")

response = processor.agent.process_ticket(state)
print(response.to_csv_row())
```

### Performance Notes

- **Speed**: ~10-15 tickets/minute (network I/O dependent)
- **Memory**: ~200MB for full corpus
- **Determinism**: Seeded randomness ensures reproducible outputs

## Known Limitations

1. **Language**: Primarily optimized for English
2. **Multi-language**: Detected but not perfectly handled
3. **Context Window**: Uses only recent conversation, not full history
4. **External APIs**: No live web calls (corpus-only)

## Troubleshooting

### LLM Rate Limited
The agent retries with exponential backoff. If persistent:
- Check API quota
- Reduce batch size
- Add delays between calls

### Missing Corpus Files
- Verify `data/` directories exist
- Check file permissions
- See `processing.log` for missing paths

### Invalid Output Format
- Check pydantic validation errors in logs
- Verify JSON in `actions_taken` column
- Ensure confidence_score is 0.0-1.0

## Submission Checklist

- [x] Deterministic outputs (seeded, temp=0.0)
- [x] All output columns present
- [x] No hardcoded responses
- [x] No external API calls (except LLM)
- [x] Proper error handling
- [x] README with reproduction steps
- [x] ARCHITECTURE.md documentation
- [x] Corpus paths correctly referenced
- [x] Code runs without manual intervention

## References

- BM25: [Okapi BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- Claude API: [Anthropic Documentation](https://docs.anthropic.com/)
- Prompt Injection: [OWASP Guide](https://owasp.org/www-community/attacks/Prompt_Injection)
