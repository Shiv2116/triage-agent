"""
main.py - Entry point for support ticket processing pipeline.

Reads support_tickets.csv, processes each ticket, writes results to output.csv
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import INPUT_CSV, OUTPUT_CSV, CSV_COLUMNS, MAX_TICKETS, LOG_FORMAT, LOG_LEVEL
from state import TicketStateManager
from retriever import create_retriever
from safety import create_safety_layer
from llm_client import create_llm_client
from agent import SupportAgent

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class TicketProcessor:
    """Main ticket processing pipeline."""

    def __init__(self):
        """Initialize processor."""
        logger.info("Initializing ticket processor...")

        # Initialize components
        self.retriever = create_retriever()
        self.safety = create_safety_layer()
        self.llm = create_llm_client()
        self.agent = SupportAgent(self.retriever, self.llm, self.safety)
        self.state_manager = TicketStateManager()

        logger.info("Processor initialized")

    def process_all_tickets(self, input_path: Optional[Path] = None) -> int:
        """
        Process all tickets from CSV.

        Args:
            input_path: Path to input CSV (default: support_tickets.csv)

        Returns:
            Number of tickets processed
        """
        if input_path is None:
            input_path = INPUT_CSV

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 0

        logger.info(f"Reading tickets from {input_path}")

        output_rows = []
        processed_count = 0
        failed_count = 0

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    if idx >= MAX_TICKETS:
                        logger.warning(f"Reached max tickets limit ({MAX_TICKETS})")
                        break

                    try:
                        # Process single ticket
                        output_row = self.process_single_ticket(row, idx)

                        if output_row:
                            output_rows.append(output_row)
                            processed_count += 1
                        else:
                            failed_count += 1

                    except Exception as e:
                        logger.error(f"Error processing row {idx}: {e}")
                        failed_count += 1
                        continue

        except Exception as e:
            logger.exception(f"Error reading CSV: {e}")
            return 0

        # Write output
        output_count = self._write_output(output_rows)

        logger.info(f"Processing complete: {processed_count} processed, {failed_count} failed")
        logger.info(f"Output written to {OUTPUT_CSV}: {output_count} rows")

        return processed_count

    def process_single_ticket(self, row: dict, index: int) -> Optional[dict]:
        """
        Process a single ticket row.

        Args:
            row: CSV row as dict
            index: Row index

        Returns:
            Output row dict, or None if processing failed
        """
        ticket_id = f"ticket_{index}"

        try:
            # Parse input
            issue_json = row.get("Issue", "{}")
            subject = row.get("Subject", "")
            company = row.get("Company")

            # Validate JSON
            try:
                issue_data = json.loads(issue_json)
                if not isinstance(issue_data, list):
                    logger.warning(f"Issue is not a list for {ticket_id}")
                    issue_data = [{"role": "user", "content": str(issue_data)}]
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in issue for {ticket_id}")
                issue_data = [{"role": "user", "content": issue_json}]

            # Create state
            state = self.state_manager.create_state(ticket_id)
            state.subject = subject if subject else None
            state.company_hint = company if company else None

            # Parse conversation
            for msg in issue_data:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if content:
                        state.add_message(role, content)

            # Process via agent
            response = self.agent.process_ticket(state)

            # Convert to CSV row
            output_row = response.to_csv_row()
            output_row["index"] = index

            logger.info(
                f"Row {index}: status={response.status}, "
                f"confidence={response.confidence_score:.2f}"
            )

            # Cleanup state
            self.state_manager.delete_state(ticket_id)

            return output_row

        except Exception as e:
            logger.error(f"Error processing ticket {ticket_id}: {e}")
            return None

    def _write_output(self, rows: list) -> int:
        """
        Write results to output CSV.

        Args:
            rows: List of output rows

        Returns:
            Number of rows written
        """
        if not rows:
            logger.warning("No rows to write")
            return 0

        try:
            # Create output directory if needed
            OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

            with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS + ["index"])
                writer.writeheader()

                for row in rows:
                    writer.writerow(row)

            logger.info(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
            return len(rows)

        except Exception as e:
            logger.error(f"Error writing output: {e}")
            return 0


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("SUPPORT TICKET TRIAGE AGENT")
    logger.info("=" * 80)

    try:
        processor = TicketProcessor()
        processed = processor.process_all_tickets()

        if processed > 0:
            logger.info("Processing completed successfully")
            return 0
        else:
            logger.error("No tickets processed")
            return 1

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
