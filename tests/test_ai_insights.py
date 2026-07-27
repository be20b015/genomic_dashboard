import unittest
from unittest.mock import patch

from core.ai_insights import analyze_fasta_sequence, get_insights


class BedrockClaudeInsightsTests(unittest.TestCase):
    def test_get_insights_accepts_bedrock_call_without_api_key(self):
        with patch("core.ai_insights._invoke_claude_bedrock", return_value="Bedrock summary") as mock_invoke:
            result = get_insights({"records": 2, "mean_length": 100}, api_key=None)

        self.assertEqual(result, "Bedrock summary")
        mock_invoke.assert_called_once()

    def test_analyze_fasta_sequence_returns_markdown(self):
        with patch("core.ai_insights._invoke_claude_bedrock", return_value='{"sequence_metadata": {"organism_guess": "demo", "genome_type": "DNA", "sequence_length_bp": 10}, "nucleotide_composition": {"estimated_gc_content_pct": 50.0, "degenerate_bases_found": []}, "open_reading_frames": [], "sequence_features": ["test"], "clinical_or_biological_relevance": "test", "quality_warnings": []}'):
            result = analyze_fasta_sequence("demo", "ACGT", as_markdown=True)

        self.assertIn("Sequence Identification", result)
        self.assertIn("test", result)


if __name__ == "__main__":
    unittest.main()
