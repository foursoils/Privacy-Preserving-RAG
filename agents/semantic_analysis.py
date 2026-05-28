import os
import json
from agents.llm_client import LLMClient, extract_json

class SemanticAnalysisAgent:
    """
    Semantic Analysis Agent (Sem-Extra Agent) performing Structured Attribute Deconstruction
    to parse raw documents into generic semantic knowledge-slot tuples.
    Supports batch processing and LLMClient abstractions.
    """

    def __init__(self, config: dict, prompt_template: str, llm_client: LLMClient):
        """
        Initialize the agent with config, prompt template, and shared LLMClient.
        """
        self.config = config
        self.prompt_template = prompt_template
        self.llm_client = llm_client

    def analyze_semantics_batch(self, texts: list) -> list:
        """
        Deconstruct multiple texts into structured semantic knowledge slots in batch.
        """
        if not texts:
            return []

        prompts = [self.prompt_template.replace("{input_text}", text) for text in texts]

        try:
            # Batch completion via LLMClient
            responses = self.llm_client.generate_batch(prompts)
            
            batch_validated_slots = []
            for response in responses:
                if not response:
                    batch_validated_slots.append([])
                    continue
                try:
                    parsed_json = extract_json(response)
                    key_info = parsed_json.get("key_information", [])
                    
                    validated_slots = []
                    for slot in key_info:
                        validated_slots.append({
                            "entity": slot.get("entity", "Unknown"),
                            "relation": slot.get("relation", "Unknown"),
                            "value": slot.get("value", ""),
                            "importance_weight": slot.get("importance_weight", "medium")
                        })
                    batch_validated_slots.append(validated_slots)
                except Exception as e:
                    print(f"Error parsing semantic response: {e}")
                    batch_validated_slots.append([])
                    
            return batch_validated_slots
            
        except Exception as e:
            print(f"Error in semantic deconstruction batch: {e}")
            return [[] for _ in texts]

    def analyze_semantics(self, text: str) -> list:
        """
        Deconstruct a single text into structured semantic knowledge slots.
        Delegates to the batch method.
        """
        results = self.analyze_semantics_batch([text])
        return results[0] if results else []
