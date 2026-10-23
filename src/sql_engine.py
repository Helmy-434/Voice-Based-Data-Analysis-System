import sqlite3
import pandas as pd
import io
import re
import json

class SQLiteEngine:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def _sanitize_name(self, name: str) -> str:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', name.strip().lower())
        return clean if clean else "dataset_table"

    def load_dataset(self, file_bytes: bytes, filename: str) -> dict:
        table_name = self._sanitize_name(filename.split(".")[0])
        ext = filename.split(".")[-1].lower()

        if ext == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif ext == "json":
            df = pd.read_json(io.BytesIO(file_bytes))
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError(f"Unsupported file extension: .{ext}")

        # Convert Nested Structures (Lists/Dicts) to Strings for SQLite Compatibility
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )

        # Clean Column Names and Store in SQLite
        df.columns = [self._sanitize_name(col) for col in df.columns]
        df.to_sql(table_name, self.conn, if_exists="replace", index=False)

        # Extract Schema Information for the LLM
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns_info = cursor.fetchall()
        formatted_schema = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])

        return {
            "table_name": table_name,
            "row_count": len(df),
            "schema_info": formatted_schema,
            "columns": [col[1] for col in columns_info],
            "sample_records": df.head(3).to_dict(orient="records")
        }