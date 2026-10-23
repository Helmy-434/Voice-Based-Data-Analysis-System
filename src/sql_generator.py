import os
import re
from langchain_groq import ChatGroq


class SQLGenerator:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0  # deterministic SQL generation
        )

    def _build_prompt(self, user_query: str, table_name: str, schema_info: str) -> str:
        return f"""You are an expert SQLite database analyst.
                Your task is to convert the user's natural language request into a valid, executable SQLite SQL query.
                
                Rules:
                1. The user request may be in English or Arabic. Understand both and generate the exact SQL needed.
                2. Use ONLY the provided table name and schema column names.
                3. Output ONLY the raw SQL query inside a ```sql ... ``` code block. Do NOT include explanations before or after the code block.
                4. Only generate read-only SELECT queries. Do not generate DROP, DELETE, or UPDATE statements.
                
                Target Table Name: {table_name}
                
                Database Schema:
                {schema_info}
                
                User Request (Voice Transcribed): "{user_query}"
                """

    def generate_sql(self, user_query: str, table_name: str, schema_info: str) -> str:

        prompt = self._build_prompt(user_query, table_name, schema_info)
        response = self.llm.invoke(prompt)
        response_text = response.content.strip()

        sql_match = re.search(r"```sql\s*(.*?)\s*```", response_text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        generic_match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
        if generic_match:
            return generic_match.group(1).strip()

        return response_text