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
# Support Ticket Triage Agent

This repository implements a terminal-based support triage agent that:

- Reads support tickets from `support_tickets/support_tickets.csv`.
- Uses BM25 retrieval over the local `data/` corpus.
-- Calls an LLM (DeepSeek by default) to generate structured, schema-validated replies.
- Applies a multi-layer safety stack: prompt-injection detection, PII sanitization, schema validation, and conservative escalation.

Key features
--------------

- Deterministic behavior (LLM temperature set to 0.0).
- In-memory state management (no external DB).
- BM25 retrieval with optional LLM re-ranking.
- Strict JSON output validation via Pydantic models in `code/schemas.py`.


1) Setup
---------

Run these commands from the repository root:

```bash
cd code/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2) Configuration
------------------

Copy the example env file and set the API key if you plan to use the remote LLM:

```bash
cp ../.env.example .env
export DEEPSEEK_API_KEY="sk-..."
# Or set in .env file
```

**Why DeepSeek?**
- Free tier with generous rate limits
- OpenAI-compatible API (easy integration)
- Fast response times
- Excellent for development and testing

### 3. Run Agent

```bash
# from code/ directory (venv activated)
python main.py
```

Notes:


- The agent logs to `processing.log` and writes `support_tickets/output.csv`.

4) Validate output
-------------------

Run the included structural validator (checks CSV columns / types):

```bash
python validate_output.py
```

This script only validates format, not semantic correctness.

Project structure
------------------

```
code/
├── main.py              # Entry point - CSV loop and orchestration
├── agent.py             # Core agent logic and decision making
├── retriever.py         # BM25 document retrieval
├── safety.py            # Injection detection + PII handling
├── state.py             # In-memory ticket state management
├── llm_client.py        # LLM integration (DeepSeek or mock)
├── schemas.py           # Pydantic models for validation
├── config.py            # Configuration and constants
├── utils.py             # Helper utilities
├── validate_output.py   # Output CSV structural validator
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── ARCHITECTURE.md      # Design doc
```

5) Testing
-----------

Use the `support_tickets/sample_support_tickets.csv` as a sample input; the agent will write `support_tickets/output.csv` which you can validate with `validate_output.py`.

6) Extending & tuning
-----------------------

- To enable LLM re-ranking, update `SupportAgent._retrieve_documents` to call `LLMClient.rerank_documents` and fall back to BM25 on failure.
- Tune thresholds in `code/config.py` (`CONFIDENCE_ESCALATION_THRESHOLD`, injection keywords, etc.).

7) Troubleshooting
-------------------

- If the run fails due to missing API key, set `DEEPSEEK_API_KEY`.
- Check `processing.log` for per-ticket errors.

8) Contributing
----------------

Open a PR with focused changes. Keep the system deterministic and avoid adding network calls outside the allowed LLM providers.

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
