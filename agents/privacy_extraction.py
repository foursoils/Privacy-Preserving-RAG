import re
import json
import os
from agents.llm_client import LLMClient, extract_json

class PrivacyExtractionAgent:
    """
    Privacy Extraction Agent (Pri-Extra Agent) operating under a
    Rule-LLM Synergistic Extraction Paradigm.
    Supports batch processing and LLMClient abstractions.
    """

    def __init__(self, config: dict, prompt_template: str, llm_client: LLMClient):
        """
        Initialize the agent with config, prompt template, and shared LLMClient.
        """
        self.config = config
        self.prompt_template = prompt_template
        self.llm_client = llm_client

    def extract_explicit_rules(self, text: str) -> list:
        """
        Extract explicit identifiers (Explicit PII) deterministically using regex.
        Matches common patterns like ages, heights, weights, phone numbers, and dates.
        """
        explicit_entities = []

        # 1. Matches ages: "32 years old", "age 32", "32-year-old"
        age_patterns = [
            r"\b\d{1,3}[ -]years?[ -]old\b",
            r"\bage\s+\d{1,3}\b",
            r"\b\d{1,3}[ -]year[ -]old\b"
        ]
        for pattern in age_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                explicit_entities.append({
                    "type": "Age",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        # 2. Matches heights: "6 feet 3 inches", "6ft 3in", "6'3\""
        height_patterns = [
            r"\b\d{1,2}\s*feet\s*\d{1,2}\s*inches?\b",
            r"\b\d{1,2}\s*ft\s*\d{1,2}\s*in\b",
            r"\b\d{1,2}'\s*\d{1,2}\"\b"
        ]
        for pattern in height_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                explicit_entities.append({
                    "type": "Height",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        # 3. Matches weights: "220 pounds", "220 lbs", "75 kg"
        weight_patterns = [
            r"\b\d{1,3}\s*pounds?\b",
            r"\b\d{1,3}\s*lbs?\b",
            r"\b\d{1,3}\s*kg\b",
            r"\b\d{1,3}\s*kilograms?\b"
        ]
        for pattern in weight_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                explicit_entities.append({
                    "type": "Weight",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        # 4. Matches standard phone numbers: "(123) 456-7890", "123-456-7890", etc.
        phone_pattern = r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"
        for match in re.finditer(phone_pattern, text):
            explicit_entities.append({
                "type": "Phone Number",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # 5. Matches common date formats: "October 12th", "10/12/2026", "2026-05-23", etc.
        date_patterns = [
            r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?\b"
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                explicit_entities.append({
                    "type": "Date",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })

        # 6. Matches relationship/family keywords: "my son", "my daughter", "my wife", "my husband"
        relation_pattern = r"\bmy\s+(?:son|daughter|wife|husband|mother|father|brother|sister|parent|child)\b"
        for match in re.finditer(relation_pattern, text, re.IGNORECASE):
            explicit_entities.append({
                "type": "Family Relation",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        return explicit_entities

    def extract_quasi_llm_batch(self, texts: list, explicit_piis_list: list) -> list:
        """
        Call the LLM in batch to extract latent and context-dependent quasi-identifiers.
        Passes the pre-extracted explicit PII to prevent redundant extractions.
        """
        prompts = []
        for text, explicit_pii in zip(texts, explicit_piis_list):
            explicit_pii_str = json.dumps([item["value"] for item in explicit_pii], ensure_ascii=False)
            prompt = self.prompt_template.replace("{input_text}", text).replace("{explicit_pii}", explicit_pii_str)
            prompts.append(prompt)

        try:
            # Batch completion via LLMClient
            responses = self.llm_client.generate_batch(prompts)
            
            quasi_results = []
            for response in responses:
                if not response:
                    quasi_results.append([])
                    continue
                try:
                    parsed_json = extract_json(response)
                    quasi_list = parsed_json.get("detected_pii", [])
                    for item in quasi_list:
                        item["extraction_method"] = "Quasi-identifier (LLM)"
                    quasi_results.append(quasi_list)
                except Exception as e:
                    print(f"Error parsing quasi-PII response: {e}")
                    quasi_results.append([])
                    
            return quasi_results
            
        except Exception as e:
            print(f"Error in LLM-based quasi-identifier batch extraction: {e}")
            return [[] for _ in texts]

    def extract_privacy_batch(self, texts: list) -> list:
        """
        Executes the dual-stage rule-LLM synergistic extraction process in batch.
        Returns a list of identified privacy elements for each input text.
        """
        if not texts:
            return []
            
        # Stage 1: Deterministic Rule Extraction for all texts
        explicit_piis_list = []
        for text in texts:
            explicit_pii = self.extract_explicit_rules(text)
            for item in explicit_pii:
                item["extraction_method"] = "Explicit (NER/regex)"
            explicit_piis_list.append(explicit_pii)
            
        # Stage 2: Generative LLM Quasi-identifier Inference in batch
        quasi_piis_list = self.extract_quasi_llm_batch(texts, explicit_piis_list)
        
        # Combine lists for each document
        batch_combined = []
        for explicit_pii, quasi_pii in zip(explicit_piis_list, quasi_piis_list):
            combined_pii = []
            # Add explicit PII
            for item in explicit_pii:
                combined_pii.append({
                    "type": item["type"],
                    "value": item["value"],
                    "extraction_method": item["extraction_method"]
                })
            # Add quasi-identifiers
            for item in quasi_pii:
                combined_pii.append({
                    "type": item.get("type", "Quasi-identifier"),
                    "value": item.get("value", ""),
                    "extraction_method": item["extraction_method"],
                    "context": item.get("context", "")
                })
            batch_combined.append(combined_pii)
            
        return batch_combined

    def extract_privacy(self, text: str) -> list:
        """
        Executes the dual-stage extraction on a single document.
        Delegates to the batch method.
        """
        results = self.extract_privacy_batch([text])
        return results[0] if results else []
