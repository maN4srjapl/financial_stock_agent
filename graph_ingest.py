import json
import uuid
import time
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError

from agent import OPENAI_API_KEY


URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "12345678"

INPUT_FILE = "data/chunked_data.json"


driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=OPENAI_API_KEY)


def insert_chunk(tx, doc_id, company, date, source, text, embedding, metadata, position):
    metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else str(metadata)
    
    result = tx.run("""
        MERGE (d:Document {doc_id: $doc_id})
        SET d.company = $company,
            d.date = $date,
            d.source = $source

        CREATE (c:Chunk {
            chunk_id: $chunk_id,
            text: $text,
            embedding: $embedding,
            metadata: $metadata,
            position: $position
        })

        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c.chunk_id AS chunk_id
    """,
    doc_id=doc_id,
    company=company,
    date=date,
    source=source,
    chunk_id=str(uuid.uuid4()),
    text=text,
    embedding=embedding,
    metadata=metadata_json,
    position=position)

    return result.single()["chunk_id"]


def extract_graph_from_text(text, max_retries=3):
    prompt = f"""
    Extract entities and relationships from financial text.

    Rules:
    - Identify companies, sectors, technologies, concepts
    - Keep entity names short
    - Use relations like:
      OPERATES_IN, INVESTS_IN, RELATED_TO, MENTIONS

    Return ONLY JSON:

    {{
      "entities": ["Entity1", "Entity2"],
      "relations": [
        ["Entity1", "RELATION", "Entity2"]
      ]
    }}

    Text:
    {text}
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=30
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            wait_time = 2 ** attempt  # exponential backoff
            print(f"Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

        except (APIConnectionError, APITimeoutError) as e:
            print(f"Connection error: {e}. Retrying... (attempt {attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

        except Exception as e:
            print(f"LLM error: {e}")
            return None

    print(" Failed to extract graph after retries")
    return None


def parse_llm_output(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return {"entities": [], "relations": []}



def insert_graph(tx, entities, relations):
    for e in entities:
        tx.run("""
            MERGE (n:Entity {name: $name})
        """, name=e)

    for r in relations:
        if len(r) != 3:
            continue

        src, rel, tgt = r

        try:
            tx.run(f"""
                MATCH (a:Entity {{name: $src}})
                MATCH (b:Entity {{name: $tgt}})
                MERGE (a)-[:{rel}]->(b)
            """, src=src, tgt=tgt)
        except:
            continue



def link_chunk_entities(tx, chunk_id, entities):
    for e in entities:
        tx.run("""
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MATCH (e:Entity {name: $entity})
            MERGE (c)-[:MENTIONS]->(e)
        """, chunk_id=chunk_id, entity=e)


def chunk_exists(tx, chunk_text):
    """Check if chunk already exists in database"""
    result = tx.run("""
        MATCH (c:Chunk {text: $text})
        RETURN c.chunk_id AS chunk_id
    """, text=chunk_text)
    
    record = result.single()
    return record is not None


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} chunks\n")

    processed = 0
    skipped = 0
    uploaded = 0

    with driver.session() as session:
        for i, doc in enumerate(data, 1):

            text = doc.get("text", "")
            if len(text) < 20:
                print(f"[{i}]   SKIP - Text too short ({len(text)} chars)")
                skipped += 1
                continue

            if session.execute_read(chunk_exists, text):
                print(f"[{i}]   SKIP - Chunk already uploaded")
                skipped += 1
                continue

            company = doc.get("company", "unknown")
            date = doc.get("date", "unknown")
            source = doc.get("source", "unknown")
            metadata = doc.get("metadata", {})

            embedding = model.encode(text).tolist()

            try:
                chunk_id = session.execute_write(
                    insert_chunk,
                    doc_id=f"{company}_{date}",
                    company=company,
                    date=date,
                    source=source,
                    text=text,
                    embedding=embedding,
                    metadata=metadata,
                    position=i
                )
                print(f"[{i}]  Chunk uploaded - ID: {chunk_id[:8]}... | {company} | {date}")

            except Exception as e:
                print(f"[{i}]  Failed to insert chunk: {str(e)[:60]}")
                skipped += 1
                continue

            # Extract entities and relations
            llm_output = extract_graph_from_text(text)

            if not llm_output:
                print(f"No graph extracted")
                uploaded += 1
                processed += 1
                continue

            parsed = parse_llm_output(llm_output)
            entities = parsed.get("entities", [])
            relations = parsed.get("relations", [])

            if not entities:
                print(f"No entities found")
                uploaded += 1
                processed += 1
                continue

            try:
                session.execute_write(insert_graph, entities, relations)
                session.execute_write(link_chunk_entities, chunk_id, entities)
                print(f" Graph: {len(entities)} entities, {len(relations)} relations")
                uploaded += 1
            except Exception as e:
                print(f" Graph insertion failed: {str(e)[:60]}")

            processed += 1

    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"Total chunks:     {len(data)}")
    print(f"Processed:        {processed}")
    print(f"Successfully uploaded: {uploaded}")
    print(f"Skipped:          {skipped}")
    print(f"✅ Pipeline completed")


if __name__ == "__main__":
    main()

