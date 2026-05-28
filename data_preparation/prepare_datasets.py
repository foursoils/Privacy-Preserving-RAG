import os
import re
import random
import yaml
import logging
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm

# Configure logging to match standard clean output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DatasetPreparer")

# Try importing LangChain components, fallback if not available
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain text splitter not available. Using pure-Python fallback splitter.")

# Try importing pandas for parquet file reading and writing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available. Parquet files cannot be read or written unless pandas is installed.")

# Try importing urlextract for URL extraction
try:
    from urlextract import URLExtract
    extractor = URLExtract()
    URLEXTRACT_AVAILABLE = True
except ImportError:
    URLEXTRACT_AVAILABLE = False
    logger.warning("urlextract not available. Falling back to regex-based URL extraction.")

# Try importing chardet for file encoding detection
try:
    from chardet.universaldetector import UniversalDetector
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logger.warning("chardet not available. Defaulting to 'utf-8' and 'latin-1' encoding fallbacks.")


class PurePythonRecursiveSplitter:
    """
    Fallback splitter when LangChain is not installed.
    Splits text by character limits with overlap.
    """
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        if text_len <= self.chunk_size:
            return [text]
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
            if start >= text_len or self.chunk_size <= self.chunk_overlap:
                break
        return chunks


class DatasetPreparer:
    """
    DatasetPreparer prepares the Wiki-PII and ChatDoctor datasets.
    It consolidates the processed data into single .parquet files
    located directly under data/processed/ without subfolders.
    """

    def __init__(self, config_path: str = "configs/data_preparation.yaml"):
        """
        Initialize the preparer with the configuration path.
        """
        # Determine the project root directory (grandparent of this script file)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Resolve config path relative to project root if it's relative
        if not os.path.isabs(config_path):
            alternative_path = os.path.join(project_root, config_path)
            if os.path.exists(alternative_path):
                config_path = alternative_path
                
        self.config_path = config_path
        self.config = self._load_config()
        
        # Helper to resolve relative path against project root
        def resolve(path_str: str) -> str:
            if os.path.isabs(path_str):
                return path_str
            return os.path.abspath(os.path.join(project_root, path_str))
        
        # Paths
        paths_config = self.config.get("data_paths", {})
        self.enron_dir = resolve(paths_config.get("enron_dir", "data/raw/enron_mail"))
        self.wiki_dir = resolve(paths_config.get("wiki_dir", "data/raw/wikitext"))
        self.chatdoctor_file = resolve(paths_config.get("chatdoctor_file", "data/raw/chatdoctor/chatdoctor.txt"))
        
        self.wiki_pii_output = resolve(paths_config.get("wiki_pii_output", "data/processed/wiki_pii.parquet"))
        self.chatdoctor_output = resolve(paths_config.get("chatdoctor_output", "data/processed/chatdoctor.parquet"))
        
        # Splitting parameters
        split_config = self.config.get("text_splitting", {})
        self.chunk_size = split_config.get("chunk_size", 1500)
        self.chunk_overlap = split_config.get("chunk_overlap", 100)
        
        # Limits
        self.max_wiki_chunks = self.config.get("max_wiki_chunks")
        self.max_chatdoctor_records = self.config.get("max_chatdoctor_records")
        
        # Patterns for fallback PII extraction
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
        self.phone_pattern = re.compile(r'\b(\d{3})[-.]?(\d{3})[-.]?(\d{4})\b')
        self.url_pattern = re.compile(r'\bhttps?://[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=]+\b')

    def _load_config(self) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def find_all_files(self, directory: str) -> List[str]:
        """
        Recursively find all files in the given directory.
        """
        file_paths = []
        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return file_paths
            
        for root, _, files in os.walk(directory):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding quickly using a try-except check.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)
            return "utf-8"
        except Exception:
            return "latin-1"

    def extract_emails(self, text: str) -> List[str]:
        """
        Extract email addresses from text.
        """
        return self.email_pattern.findall(text)

    def extract_phones(self, text: str) -> List[str]:
        """
        Extract phone numbers and format them as XXX-XXX-XXXX.
        """
        matches = self.phone_pattern.findall(text)
        return [f"{area}-{mid}-{last}" for area, mid, last in matches]

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text using urlextract or regex fallback.
        """
        if URLEXTRACT_AVAILABLE:
            try:
                return extractor.find_urls(text)
            except Exception:
                pass
        return self.url_pattern.findall(text)

    def load_wiki_text_chunks(self) -> List[str]:
        """
        Load wikitext from files and split into chunks.
        Supports both text files and parquet files.
        """
        wiki_files = self.find_all_files(self.wiki_dir)
        if not wiki_files:
            raise FileNotFoundError(f"No wikitext files found in {self.wiki_dir}")
            
        paragraphs = []
        parquet_files = [f for f in wiki_files if f.endswith(".parquet")]
        text_files = [f for f in wiki_files if f.endswith(".txt") or f.endswith(".raw") or f.endswith(".tokens")]
        
        if parquet_files:
            if not PANDAS_AVAILABLE:
                raise ImportError("Found wikitext parquet files but pandas is not installed. Run 'pip install pandas'.")
            logger.info(f"Reading {len(parquet_files)} parquet wikitext files...")
            for p_file in parquet_files:
                try:
                    df = pd.read_parquet(p_file)
                    text_col = 'text' if 'text' in df.columns else df.columns[0]
                    paragraphs.extend(df[text_col].dropna().tolist())
                except Exception as e:
                    logger.error(f"Failed to read parquet file {p_file}: {e}")
        
        elif text_files:
            logger.info(f"Reading {len(text_files)} text wikitext files...")
            for t_file in text_files:
                encoding = self.detect_encoding(t_file)
                try:
                    with open(t_file, "r", encoding=encoding, errors="replace") as f:
                        paragraphs.append(f.read())
                except Exception as e:
                    logger.error(f"Failed to read text file {t_file}: {e}")
        else:
            raise FileNotFoundError(f"Supported formats (.parquet, .txt, .raw, .tokens) not found in {self.wiki_dir}")

        # Combine paragraphs or clean lines
        combined_text = "\n\n".join(paragraphs)
        
        # Split text into chunks
        if LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_text(combined_text)
        else:
            splitter = PurePythonRecursiveSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_text(combined_text)
            
        logger.info(f"Successfully split WikiText into {len(chunks)} chunks.")
        return chunks

    def _process_single_chunk(self, args) -> Dict[str, Any]:
        """
        Helper worker function to process a single WikiText chunk and inject PII.
        """
        i, chunk, enron_file = args
        encoding = self.detect_encoding(enron_file)
        try:
            with open(enron_file, "r", encoding=encoding, errors="replace") as f:
                enron_content = f.read()
        except Exception as e:
            enron_content = ""
            
        # Extract PII elements
        pii_emails = self.extract_emails(enron_content)
        pii_phones = self.extract_phones(enron_content)
        pii_urls = self.extract_urls(enron_content)
        
        all_pii = pii_emails + pii_phones + pii_urls
        random.shuffle(all_pii)
        
        # Sentence-level injection
        sentences = [s.strip() for s in re.split(r'\.', chunk) if s.strip()]
        new_sentences = []
        for j, sen in enumerate(sentences):
            if all_pii:
                pii_item = all_pii[j % len(all_pii)]
                new_sentences.append(f"{sen}. {pii_item}.")
            else:
                new_sentences.append(f"{sen}.")
                
        injected_content = " ".join(new_sentences)
        return {
            "id": i,
            "content": injected_content
        }

    def prepare_wiki_pii(self) -> None:
        """
        Reads Enron emails, extracts PII elements, chunks wikitext,
        injects PII into sentences, and saves the output to a consolidated parquet file.
        Uses multi-threading to speed up raw file reading and processing.
        """
        logger.info("Starting Wiki-PII dataset preparation...")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and pyarrow are required to save as parquet. Run 'pip install pandas pyarrow'.")
            
        # Find and shuffle Enron email files
        enron_files = self.find_all_files(self.enron_dir)
        if not enron_files:
            raise FileNotFoundError(f"No Enron email files found in {self.enron_dir}")
        logger.info(f"Found {len(enron_files)} Enron email files.")
        random.shuffle(enron_files)
        
        # Load wikitext chunks
        wiki_chunks = self.load_wiki_text_chunks()
        if not wiki_chunks:
            raise ValueError("No text chunks generated from wikitext files.")
            
        if self.max_wiki_chunks is not None and self.max_wiki_chunks > 0:
            logger.info(f"Limiting processing to {self.max_wiki_chunks} Wiki chunks as configured.")
            wiki_chunks = wiki_chunks[:self.max_wiki_chunks]
            
        # Prepare parallel task arguments
        task_args = [
            (i, wiki_chunks[i], enron_files[i % len(enron_files)])
            for i in range(len(wiki_chunks))
        ]
        
        records = [None] * len(wiki_chunks)
        max_workers = self.config.get("max_workers", 16)
        logger.info(f"Injecting PII into Wiki chunks using ThreadPoolExecutor ({max_workers} workers)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all processing tasks
            futures = {
                executor.submit(self._process_single_chunk, args): args[0]
                for args in task_args
            }
            
            # Retrieve completed futures with progress bar
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Preparing Wiki-PII"
            ):
                chunk_id = futures[future]
                try:
                    res = future.result()
                    records[chunk_id] = res
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id}: {e}")
            
        # Ensure parent output directory exists
        os.makedirs(os.path.dirname(self.wiki_pii_output), exist_ok=True)
        
        try:
            logger.info(f"Saving Wiki-PII consolidated parquet to {self.wiki_pii_output}...")
            df = pd.DataFrame(records)
            df.to_parquet(self.wiki_pii_output, index=False)
            logger.info(f"Wiki-PII dataset preparation completed! Generated: {self.wiki_pii_output}")
        except Exception as e:
            logger.error(f"Failed to save Wiki-PII parquet: {e}")

    def prepare_chatdoctor(self) -> None:
        """
        Reads raw ChatDoctor text file, parses it into separate QA dialogue strings,
        and saves the output directly as a consolidated parquet file.
        """
        logger.info("Starting ChatDoctor dataset preparation...")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and pyarrow are required to save as parquet. Run 'pip install pandas pyarrow'.")
            
        if not os.path.exists(self.chatdoctor_file):
            raise FileNotFoundError(f"ChatDoctor raw file not found: {self.chatdoctor_file}")
            
        # Read the entire chatdoctor file
        encoding = self.detect_encoding(self.chatdoctor_file)
        logger.info(f"Reading ChatDoctor file {self.chatdoctor_file}...")
        try:
            with open(self.chatdoctor_file, "r", encoding=encoding, errors="replace") as f:
                raw_data = f.read()
        except Exception as e:
            logger.error(f"Failed to read ChatDoctor file: {e}")
            return
            
        # Split dialogues by double newlines
        dialogues = [d.strip() for d in raw_data.split("\n\n") if d.strip()]
        logger.info(f"Parsed {len(dialogues)} dialogues from ChatDoctor.")
        
        if self.max_chatdoctor_records is not None and self.max_chatdoctor_records > 0:
            logger.info(f"Limiting processing to {self.max_chatdoctor_records} ChatDoctor dialogues as configured.")
            dialogues = dialogues[:self.max_chatdoctor_records]
            
        records = []
        for i, dialogue in enumerate(tqdm(dialogues, desc="Preparing ChatDoctor")):
            # Clean up potential replacement characters
            cleaned_dialogue = dialogue.replace('\xa0', ' ')
            records.append({
                "id": i,
                "chat": cleaned_dialogue
            })
            
        # Ensure parent output directory exists
        os.makedirs(os.path.dirname(self.chatdoctor_output), exist_ok=True)
        
        try:
            logger.info(f"Saving ChatDoctor consolidated parquet to {self.chatdoctor_output}...")
            df = pd.DataFrame(records)
            df.to_parquet(self.chatdoctor_output, index=False)
            logger.info(f"ChatDoctor dataset preparation completed! Generated: {self.chatdoctor_output}")
        except Exception as e:
            logger.error(f"Failed to save ChatDoctor parquet: {e}")

    def prepare_all(self) -> None:
        """
        Run preparation for both Wiki-PII and ChatDoctor datasets.
        """
        self.prepare_wiki_pii()
        logger.info("-" * 50)
        self.prepare_chatdoctor()


if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.prepare_all()
