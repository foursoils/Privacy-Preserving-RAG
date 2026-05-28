import os
import sys

# Add parent directory of 'agents' to sys.path to support running from within the agents folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from agents.llm_client import LLMClient
from agents.privacy_extraction import PrivacyExtractionAgent
from agents.semantic_analysis import SemanticAnalysisAgent
from agents.reconstruction import ReconstructionAgent

class PPRSystem:
    """
    PPRSystem coordinates the offline multi-agent sanitization pipeline:
    1. Extracts explicit and implicit quasi-identifiers (Pri-Extra Agent).
    2. Deconstructs key semantic context into slots (Sem-Extra Agent).
    3. Reconstructs a safe, rewritten text (Reconstruction Agent).
    Supports single-document and list-based batch processing.
    """

    def __init__(self, config_path: str = "configs/agents_config.yaml"):
        """
        Initialize the system by loading the config and the agent prompt templates.
        """
        # Determine the project root directory (parent of agents folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Resolve config path relative to project root if it's relative
        if not os.path.isabs(config_path):
            alternative_path = os.path.abspath(os.path.join(project_root, config_path))
            if os.path.exists(alternative_path):
                config_path = alternative_path

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Load prompt templates (resolved relative to project root if relative)
        prompts_config = self.config.get("agents", {})
        
        def resolve_path(p: str) -> str:
            if not os.path.isabs(p):
                alt = os.path.abspath(os.path.join(project_root, p))
                if os.path.exists(alt):
                    return alt
            return p

        # Privacy prompt
        privacy_prompt_path = resolve_path(prompts_config.get("privacy_extraction", {}).get("prompt_path", "prompts/privacy_extraction.txt"))
        with open(privacy_prompt_path, 'r', encoding='utf-8') as f:
            privacy_prompt = f.read()

        # Semantic prompt
        semantic_prompt_path = resolve_path(prompts_config.get("semantic_analysis", {}).get("prompt_path", "prompts/semantic_analysis.txt"))
        with open(semantic_prompt_path, 'r', encoding='utf-8') as f:
            semantic_prompt = f.read()

        # Reconstruction prompt
        reconstruct_prompt_path = resolve_path(prompts_config.get("reconstruction", {}).get("prompt_path", "prompts/reconstruction.txt"))
        with open(reconstruct_prompt_path, 'r', encoding='utf-8') as f:
            reconstruct_prompt = f.read()

        # Initialize global LLM Client
        self.llm_client = LLMClient(self.config)

        # Initialize agents with shared LLM Client
        privacy_agent_config = prompts_config.get("privacy_extraction", {})
        self.privacy_agent = PrivacyExtractionAgent(privacy_agent_config, privacy_prompt, self.llm_client)

        semantic_agent_config = prompts_config.get("semantic_analysis", {})
        self.semantic_agent = SemanticAnalysisAgent(semantic_agent_config, semantic_prompt, self.llm_client)

        reconstruct_agent_config = prompts_config.get("reconstruction", {})
        self.reconstruct_agent = ReconstructionAgent(reconstruct_agent_config, reconstruct_prompt, self.llm_client)

    # --- Single Document Methods ---

    def extract_privacy(self, text: str) -> list:
        """
        Step 1: Extract privacy constraints (P_seq) for a single document.
        """
        return self.privacy_agent.extract_privacy(text)

    def analyze_semantics(self, text: str) -> list:
        """
        Step 2: Deconstruct semantic core (K_seq) for a single document.
        """
        return self.semantic_agent.analyze_semantics(text)

    def reconstruct(self, text: str, detected_pii: list, semantic_slots: list) -> str:
        """
        Step 3: Rewrite and reconstruct text (T_safe) for a single document.
        """
        return self.reconstruct_agent.reconstruct_text(
            original_text=text,
            detected_pii=detected_pii,
            semantic_slots=semantic_slots
        )

    def process_document(self, text: str) -> dict:
        """
        Runs the full offline rewrite pipeline on a single document.
        """
        detected_pii = self.extract_privacy(text)
        semantic_slots = self.analyze_semantics(text)
        rewritten_text = self.reconstruct(text, detected_pii, semantic_slots)

        return {
            "original_text": text,
            "detected_pii": detected_pii,
            "semantic_slots": semantic_slots,
            "rewritten_text": rewritten_text
        }

    # --- Batch Methods ---

    def extract_privacy_batch(self, texts: list) -> list:
        """
        Step 1 (Batch): Extract privacy constraints (P_seq) for a list of documents.
        """
        return self.privacy_agent.extract_privacy_batch(texts)

    def analyze_semantics_batch(self, texts: list) -> list:
        """
        Step 2 (Batch): Deconstruct semantic core (K_seq) for a list of documents.
        """
        return self.semantic_agent.analyze_semantics_batch(texts)

    def reconstruct_batch(self, texts: list, detected_piis: list, semantic_slotss: list) -> list:
        """
        Step 3 (Batch): Rewrite and reconstruct text (T_safe) for a list of documents.
        """
        return self.reconstruct_agent.reconstruct_batch(texts, detected_piis, semantic_slotss)

    def process_documents_batch(self, texts: list) -> list:
        """
        Runs the full offline rewrite pipeline in batch mode.
        """
        detected_piis = self.extract_privacy_batch(texts)
        semantic_slotss = self.analyze_semantics_batch(texts)
        rewritten_texts = self.reconstruct_batch(texts, detected_piis, semantic_slotss)

        results = []
        for i in range(len(texts)):
            results.append({
                "original_text": texts[i],
                "detected_pii": detected_piis[i],
                "semantic_slots": semantic_slotss[i],
                "rewritten_text": rewritten_texts[i]
            })
        return results


if __name__ == "__main__":
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description="PPR Multi-Agent Sanitization Pipeline Step-by-Step Test Runner")
    parser.add_argument("--config", type=str, default="configs/agents_config.yaml", help="Path to config file")
    parser.add_argument("--text", type=str, default="", help="Single input text to sanitize (if empty, default batch sample will be used)")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], default=None, help="Specific step to run (1: Privacy Extraction, 2: Semantic Analysis, 3: Reconstruction)")
    args = parser.parse_args()
    
    # Determine the project root directory (parent of agents folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Resolve config relative to project root if relative
    config_path = args.config
    if not os.path.isabs(config_path):
        alt = os.path.abspath(os.path.join(project_root, config_path))
        if os.path.exists(alt):
            config_path = alt

    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    print(f"Initializing PPRSystem with configuration: {config_path}...")
    try:
        system = PPRSystem(config_path=config_path)
        print(f"PPRSystem initialized successfully. Mode: {system.llm_client.mode}\n")
    except Exception as e:
        print(f"Failed to initialize PPRSystem: {e}")
        sys.exit(1)
        
    # Set up documents to process
    if args.text:
        sample_texts = [args.text]
        is_single = True
    else:
        sample_texts = [
            "Patient John Doe, a 45-year-old deep-sea welder, was admitted to Massachusetts General Hospital "
            "on October 12th. He presented with severe decompression sickness and abdominal pain. "
            "His daughter, who works at the local bakery, accompanied him. Dr. Smith recommended immediate "
            "hyperbaric oxygen therapy and complete rest.",
            
            "Patient Jane Smith, a 28-year-old software developer, visited the rural clinic in Oakhaven last Tuesday. "
            "She reported persistent migraines and fatigue after working late. Dr. Adams advised taking a week off "
            "and prescribed standard pain relievers."
        ]
        is_single = False
        
    print("=" * 60)
    print("INPUT TEXTS TO PROCESS:")
    for i, t in enumerate(sample_texts):
        print(f"[{i+1}] {t}")
    print("=" * 60 + "\n")
    
    if args.step == 1:
        print("Running Step 1: Privacy Extraction Agent (Batch)...")
        piis = system.extract_privacy_batch(sample_texts)
        print(json.dumps(piis, indent=2, ensure_ascii=False))
    elif args.step == 2:
        print("Running Step 2: Semantic Analysis Agent (Batch)...")
        slots = system.analyze_semantics_batch(sample_texts)
        print(json.dumps(slots, indent=2, ensure_ascii=False))
    elif args.step == 3:
        print("Running Step 3: Reconstruction Agent (Batch)...")
        print("Extracting PII and Semantics first to run reconstruction...")
        piis = system.extract_privacy_batch(sample_texts)
        slots = system.analyze_semantics_batch(sample_texts)
        print("Reconstructing texts...")
        rewrittens = system.reconstruct_batch(sample_texts, piis, slots)
        for i, r in enumerate(rewrittens):
            print(f"Sanitized Result [{i+1}]:\n{r}\n")
    else:
        # Run entire pipeline step-by-step
        print("Running batch pipeline step-by-step:\n")
        
        print("--- Step 1: Privacy Extraction Agent ---")
        piis = system.extract_privacy_batch(sample_texts)
        print(json.dumps(piis, indent=2, ensure_ascii=False))
        print("-" * 40 + "\n")
        
        print("--- Step 2: Semantic Analysis Agent ---")
        slots = system.analyze_semantics_batch(sample_texts)
        print(json.dumps(slots, indent=2, ensure_ascii=False))
        print("-" * 40 + "\n")
        
        print("--- Step 3: Reconstruction Agent ---")
        rewrittens = system.reconstruct_batch(sample_texts, piis, slots)
        for i, r in enumerate(rewrittens):
            print(f"Sanitized Result [{i+1}]:\n{r}\n")
        print("-" * 40 + "\n")
