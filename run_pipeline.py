import os
import json
import argparse
import sys
import pandas as pd
from tqdm import tqdm
from agents.system import PPRSystem

def main():
    import yaml

    # 1. First pass: parse the config path only
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", "-c", type=str, default="configs/agents_config.yaml")
    parsed_args, remaining_argv = conf_parser.parse_known_args()

    # Load config defaults if available
    config_defaults = {}
    config_path = parsed_args.config
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                config_defaults = cfg.get("execution_settings", {})
        except Exception as e:
            print(f"Warning: Failed to load default execution settings from '{config_path}': {e}")

    # 2. Second pass: parse all CLI arguments, using config values as defaults
    parser = argparse.ArgumentParser(description="PPR Offline Multi-Agent Sanitization Dataset Runner")
    
    default_dataset = config_defaults.get("dataset")
    default_qtype = config_defaults.get("qtype")
    default_step = config_defaults.get("step")
    default_limit = config_defaults.get("limit")

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=(default_dataset is None),  # Required only if not present in config
        default=default_dataset,
        choices=["chat", "chatdoctor", "wiki", "wiki_pii"],
        help="Dataset to process: 'chatdoctor' (alias 'chat') or 'wiki_pii' (alias 'wiki')"
    )
    parser.add_argument(
        "--qtype", "-q",
        type=str,
        required=(default_qtype is None),  # Required only if not present in config
        default=default_qtype,
        choices=["target_questions", "untarget_questions", "utility_questions"],
        help="Question type: 'target_questions', 'untarget_questions', or 'utility_questions'"
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        choices=[1, 2, 3],
        default=default_step,
        help="Step to run (1: Privacy Extraction, 2: Semantic Analysis, 3: Reconstruction). If omitted, runs all steps."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=config_path,
        help="Path to multi-agent configuration YAML file"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=default_limit,
        help="Limit the number of questions to process (useful for testing/debugging)"
    )
    args = parser.parse_args(remaining_argv)

    # Resolve dataset name
    dataset_map = {
        "chat": "chatdoctor",
        "chatdoctor": "chatdoctor",
        "wiki": "wiki_pii",
        "wiki_pii": "wiki_pii"
    }
    dataset = dataset_map[args.dataset]
    qtype = args.qtype

    # Input file check
    input_file = f"data/retrieved/{qtype}/{dataset}.parquet"
    if not os.path.exists(input_file):
        print(f"Error: Input retrieved Parquet file not found at '{input_file}'")
        print("Please run retrieval/retrieve.py first to generate retrieval context files.")
        sys.exit(1)

    # Config file check
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)

    # Initialize output directory under results/
    output_dir = f"results/{qtype}"
    os.makedirs(output_dir, exist_ok=True)

    # Define intermediate and final output file paths
    pii_cache_file = os.path.join(output_dir, f"{dataset}_pii.json")
    slots_cache_file = os.path.join(output_dir, f"{dataset}_slots.json")
    sanitized_output_file = os.path.join(output_dir, f"{dataset}.parquet")

    # Load retrieved questions and contexts
    print(f"Loading retrieval contexts from '{input_file}'...")
    df = pd.read_parquet(input_file)
    if args.limit and args.limit > 0:
        print(f"Limiting execution to first {args.limit} queries for debugging.")
        df = df.head(args.limit)

    total_queries = len(df)
    print(f"Loaded {total_queries} queries.")

    # Step 1: Flatten and extract unique documents to minimize LLM calls / costs
    unique_docs = {}  # doc_id -> doc_text
    for _, row in df.iterrows():
        doc_ids = list(row["retrieved_ids"])
        doc_texts = list(row["retrieved_contexts"])
        for doc_id, doc_text in zip(doc_ids, doc_texts):
            # Convert doc_id to string to maintain JSON key consistency
            unique_docs[str(doc_id)] = doc_text

    doc_ids_list = list(unique_docs.keys())
    doc_texts_list = list(unique_docs.values())
    total_unique_docs = len(doc_ids_list)
    print(f"Extracted {total_unique_docs} unique retrieved documents out of {total_queries * len(doc_ids)} context slots.")

    # Initialize PPRSystem
    print(f"Initializing PPRSystem with config '{args.config}'...")
    try:
        system = PPRSystem(config_path=args.config)
        print(f"System initialized in '{system.llm_client.mode}' mode.\n")
    except Exception as e:
        print(f"Failed to initialize PPRSystem: {e}")
        sys.exit(1)


    # --- Step 1: Privacy Extraction ---
    pii_map = {}
    if args.step is None or args.step == 1:
        print("=" * 60)
        print("STEP 1: Privacy Extraction Agent")
        print("=" * 60)
        print(f"Running Privacy Extraction on {total_unique_docs} unique documents...")
        pii_results = system.extract_privacy_batch(doc_texts_list)
        
        pii_map = {doc_id: pii for doc_id, pii in zip(doc_ids_list, pii_results)}
        # Save to cache
        print(f"Saving extracted PII mapping to cache -> '{pii_cache_file}'")
        with open(pii_cache_file, "w", encoding="utf-8") as f:
            json.dump(pii_map, f, ensure_ascii=False, indent=2)
        print("Step 1 completed.\n")
    else:
        # Load from cache if executing subsequent steps individually
        if os.path.exists(pii_cache_file):
            print(f"Step 1 skipped. Loading PII cache from '{pii_cache_file}'...")
            with open(pii_cache_file, "r", encoding="utf-8") as f:
                pii_map = json.load(f)
        elif args.step == 3:
            print("Step 1 cache not found. Running Privacy Extraction on the fly...")
            pii_results = system.extract_privacy_batch(doc_texts_list)
            pii_map = {doc_id: pii for doc_id, pii in zip(doc_ids_list, pii_results)}

    # --- Step 2: Semantic Analysis ---
    slots_map = {}
    if args.step is None or args.step == 2:
        print("=" * 60)
        print("STEP 2: Semantic Analysis Agent")
        print("=" * 60)
        print(f"Running Semantic Slot Deconstruction on {total_unique_docs} unique documents...")
        slots_results = system.analyze_semantics_batch(doc_texts_list)
        
        slots_map = {doc_id: slots for doc_id, slots in zip(doc_ids_list, slots_results)}
        # Save to cache
        print(f"Saving extracted Semantic Slots mapping to cache -> '{slots_cache_file}'")
        with open(slots_cache_file, "w", encoding="utf-8") as f:
            json.dump(slots_map, f, ensure_ascii=False, indent=2)
        print("Step 2 completed.\n")
    else:
        # Load from cache if executing subsequent steps individually
        if os.path.exists(slots_cache_file):
            print(f"Step 2 skipped. Loading Semantic Slots cache from '{slots_cache_file}'...")
            with open(slots_cache_file, "r", encoding="utf-8") as f:
                slots_map = json.load(f)
        elif args.step == 3:
            print("Step 2 cache not found. Running Semantic Analysis on the fly...")
            slots_results = system.analyze_semantics_batch(doc_texts_list)
            slots_map = {doc_id: slots for doc_id, slots in zip(doc_ids_list, slots_results)}

    # --- Step 3: Reconstruction ---
    if args.step is None or args.step == 3:
        print("=" * 60)
        print("STEP 3: Reconstruction Agent")
        print("=" * 60)
        
        # Prepare inputs for batch reconstruction
        detected_piis_list = [pii_map.get(doc_id, []) for doc_id in doc_ids_list]
        semantic_slots_list = [slots_map.get(doc_id, []) for doc_id in doc_ids_list]
        
        print(f"Running Text Reconstruction on {total_unique_docs} unique documents...")
        rewritten_results = system.reconstruct_batch(doc_texts_list, detected_piis_list, semantic_slots_list)
        
        rewritten_map = {doc_id: text for doc_id, text in zip(doc_ids_list, rewritten_results)}
        
        # Map rewritten safe texts back to the original query Parquet dataframe
        print("Mapping sanitized documents back to retrieval contexts...")
        sanitized_contexts_col = []
        for _, row in df.iterrows():
            row_sanitized = []
            for doc_id in row["retrieved_ids"]:
                # Lookup sanitized text using doc ID, fallback to original if missing
                sanitized_text = rewritten_map.get(str(doc_id), "")
                row_sanitized.append(sanitized_text)
            sanitized_contexts_col.append(row_sanitized)
            
        df["retrieved_contexts"] = sanitized_contexts_col
        
        # Save output Parquet
        print(f"Saving final sanitized contexts Parquet file -> '{sanitized_output_file}'")
        df.to_parquet(sanitized_output_file, index=False)
        print("Step 3 completed. Pipeline sanitization complete!\n")


if __name__ == "__main__":
    main()
