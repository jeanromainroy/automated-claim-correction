# Libraries
import os
import json
import numpy as np

# Helper scripts
from libs.strings import remove_links
from libs.openai import complete, embed

# Variables
database = None
database_embedding_by_claim = None
database_embeddings = None


def compose_prompt_claims(text):

    # Clean
    text = remove_links(text)
    text = text[:4096]  # Limit to 4096 tokens

    return f"""
DOCUMENT:
"{text}"

TASK:
Identify all claims in the document. A claim is a statement that:
- Asserts or implies something as true, factual, or plausible (e.g., "X happened," "Y is true," or "Z will occur").
- Takes a position (e.g., supports or opposes something).
- Shows a connection, causation, or relationship (e.g., "Could X have caused Y?").
- Proposes an explanation, hypothesis, or prediction.

GUIDELINES:
- Claims must be concise and self-contained.
- Exclude unnecessary details unless essential for clarity.
- Focus each claim on a single idea.
- Keep claims objective and based only on the document content.

OUTPUT:
Return the main claims as a JSON array. For example:
claims: [
    "NATO provoked the war",
    "Neo-Nazis are in the Ukrainian government",
    ...
]

If the document does not contain any claims, return an empty list: [].
""".strip()


def find_top_k_similar(embedding, k=2, min_similarity=0.85):
    """
    Find the top-k most similar embeddings using cosine similarity.
    """
    # Normalize the query and database embeddings
    query_norm = embedding / np.linalg.norm(embedding)
    database_norms = database_embeddings / np.linalg.norm(database_embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarities
    similarities = np.dot(database_norms, query_norm)
    
    # Get the indices of the top-k most similar items
    top_k_indices = np.argsort(similarities)[::-1][:k]  # Sort in descending order

    # Filter out results with similarity below the threshold
    top_k_indices = [idx for idx in top_k_indices if similarities[idx] >= min_similarity]
    
    # Retrieve the corresponding texts and similarities
    top_k_results = [(database[idx], round(float(similarities[idx]), 4)) for idx in top_k_indices]
    
    return top_k_results


def pipeline_lookup_counterclaims(text, filepath_database):
    global database, database_embedding_by_claim, database_embeddings

    """ Step 0: Load database """

    # Initialize database if not loaded
    if database is None:

        # Load database
        with open(filepath_database, 'r') as file:
            database = json.load(file)

        # Map claims to embeddings
        database_embedding_by_claim = { row['claim']: row['embedding'] for row in database }
            
        # Convert database embeddings to numpy arrays for faster computation
        database_embeddings = np.array([ row['embedding'] for row in database ])

        # Log
        print(f'Database: Loaded {len(database):,} claims.')


    """ Step 1: Extract claims from a document """

    # Compose prompt
    prompt = compose_prompt_claims(text)

    # Send for completion 
    result = complete(prompt, max_tokens=2048, model_large=True, json_mode=True)
    if result is None:
        print("Error: OpenAI completion failed.")
        return None

    # Initial format { 'claims': [ ... ] } (due to llm processing)
    if 'claims' in result:
        claims = result['claims']
    else:
        print(f'Error: Unexpected response format: {result}')
        return None

    # Remove '.' at the end
    claims = list(set([ claim[:-1].strip() if claim[-1] == '.' else claim.strip() for claim in claims ]))

    
    """ Step 2: Embed claims """

    # Embed claims
    claims_embedded = {}

    # Iterate over claims
    for claim in claims:

        # If we already have the embedding for this exact claim, use it
        if claim in database_embedding_by_claim:
            print(f'Info: Using existing embedding for claim: {claim}')
            claims_embedded[claim] = database_embedding_by_claim[claim]
            continue

        # Embed claim
        print(f'Info: Embedding claim: {claim}')
        embedding = embed(claim)
        if embedding is None:
            print(f'Error: Embedding failed for claim: {claim}')
            continue

        # Store
        claims_embedded[claim] = embedding
        

    """ Step 3: Return most relevant counterclaims from database """
    
    # Initialize
    claims_counterclaims = []

    # Iterate over claims
    for claim, embedding in claims_embedded.items():
    
        # Find top-k most similar claims in the database
        top_claims = find_top_k_similar(embedding=embedding)

        # Log
        print(f'Info: Found {len(top_claims)} claims in the database for claim: {claim}')
        
        # Append to results
        for db_row, similarity in top_claims:
            claims_counterclaims.append({
                'claim': claim,
                'db_claim': db_row['claim'],
                'counterclaim': db_row['counterclaim'],
                'similarity': similarity
            })

    # Sort by similarity
    claims_counterclaims = sorted(claims_counterclaims, key=lambda x: x['similarity'], reverse=True)

    """ Step 4: Return results """

    return claims_counterclaims
