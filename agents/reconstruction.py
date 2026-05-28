import os
import json
from agents.llm_client import LLMClient, extract_json

class ReconstructionAgent:
    """
    Reconstruction Agent responsible for synthesizing a natural, coherent text (T_safe)
    using the extracted privacy boundaries (P_seq) and semantic backbone (K_seq).
    Applies Fine-grained Conflict Routing to navigate overlaps between privacy and utility.
    Supports batch processing and LLMClient abstractions.
    """

    def __init__(self, config: dict, prompt_template: str, llm_client: LLMClient):
        """
        Initialize the agent with config, prompt template, and shared LLMClient.
        """
        self.config = config
        self.prompt_template = prompt_template
        self.llm_client = llm_client

    def reconstruct_batch(self, texts: list, detected_piis: list, semantic_slotss: list) -> list:
        """
        Generates the sanitized, rewritten texts (T_safe) in batch.
        """
        if not texts:
            return []

        prompts = []
        for text, detected_pii, semantic_slots in zip(texts, detected_piis, semantic_slotss):
            pii_str = json.dumps(detected_pii, ensure_ascii=False, indent=2)
            semantics_str = json.dumps(semantic_slots, ensure_ascii=False, indent=2)
            
            prompt = (self.prompt_template
                      .replace("{input_text}", text)
                      .replace("{detected_pii}", pii_str)
                      .replace("{semantic_backbone}", semantics_str))
            prompts.append(prompt)

        try:
            # Batch completion via LLMClient
            responses = self.llm_client.generate_batch(prompts)
            
            rewritten_texts = []
            for response in responses:
                if not response:
                    rewritten_texts.append("")
                    continue
                try:
                    parsed_json = extract_json(response)
                    rewritten_texts.append(parsed_json.get("rewritten_text", ""))
                except Exception as e:
                    print(f"Error parsing reconstructed text response: {e}")
                    rewritten_texts.append("")
                    
            return rewritten_texts
            
        except Exception as e:
            print(f"Error in text reconstruction batch: {e}")
            return ["" for _ in texts]

    def reconstruct_text(self, original_text: str, detected_pii: list, semantic_slots: list) -> str:
        """
        Generates the sanitized, rewritten text (T_safe) for a single document.
        Delegates to the batch method.
        """
        results = self.reconstruct_batch([original_text], [detected_pii], [semantic_slots])
        return results[0] if results else ""
