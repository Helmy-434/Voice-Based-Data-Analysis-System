
from datasets import load_dataset
from src.rag import get_rag_manager

ds = load_dataset("openai/openai_humaneval", split="test")
col = get_rag_manager().collection

col.upsert(                           
    ids=[r["task_id"] for r in ds],
    documents=[r["prompt"] + r["canonical_solution"] for r in ds],
    metadatas=[{
        "source": "humaneval",
        "task_id": r["task_id"],
        "entry_point": r["entry_point"],
        "origin": "seed",
    } for r in ds],
)
print("Chunks in KB:", col.count())