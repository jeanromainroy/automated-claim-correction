# Automated Claim Correction: Evidence from the Russia-Ukraine Conflict

Social media platforms enable the rapid and widespread dissemination of new claims, while corrective responses often lag in both timing and visibility. This paper presents a system for real-time identification and response to contentious content. The system maintains a database of pre-compiled claims and corresponding corrections. It analyzes incoming social media posts, news articles, and other online content to extract embedded claims, retrieve the appropriate corrections, and automatically generate appropriate responses. We evaluate our approach using a corpus of 313 social media posts from Russian state propaganda outlets. In 76\% of cases, our system’s responses were judged more effective at challenging the content than the crowdsourced human alternative (Twitter Community Notes). These results demonstrate the potential for automated approaches to provide timely and scalable interventions in contentious online discourse.


## Methodology

### 1. Community Notes

Download 'Notes data' and 'Note status history data' from [here](https://x.com/i/communitynotes/download-data) into Data/ folder.

Run the scripts to filter and preprocess the data.


### 4. Claim Extraction

Using an LLM, extract the claims from each filtered document.

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


### 5. Claim Embedding

Represent each claim as a vector using a text embedding model. This vector representation captures the semantic meaning of the claim, enabling similarity comparisons and clustering.

### 6. LLM-Generated Counterclaims

Ideally, subject matter experts would create and verify each counterclaim. However, for this project, counterclaims are generated by an LLM. For each extracted claim, use an LLM to generate a counterclaim that employs a **fact-based strategy**.

    CLAIM:
    "{claim}"

    TASK:
    Verify the factual accuracy of the claim. Provide a short explanation supported by concise and hard evidence.

    INSTRUCTIONS:
    1. Analyze the claim: Identify the key assertions and context.
    2. Research evidence: Use only credible sources (e.g., academic studies, verified news, official records).
    3. Evaluate reasoning: Identify any fallacies, biases, or misinformation.
    4. Determine accuracy: Label the claim as "True" or "False." Justify your label with brief, precise, and verifiable evidence (no lengthy explanations).

    OUTPUT FORMAT:
    Return the analysis as a structured JSON object in this format:
    {
        "claim": "{claim}",
        "label": "True/False",
        "evidence": "Short, factual summary with hard evidence (cite only key sources)."
    }

Once a counterclaim has been generated for the claim, the second task is to validate whether the counterclaim directly and meaningfully entails the original claim. This step ensures vague or overly broad claims are removed.

    CLAIM: 
    "{claim}"

    RESPONSE:
    "{counterclaim}"

    TASK:
    Return "Yes" only if the RESPONSE directly challenges the CLAIM's main point with clear reasoning or specific evidence. Return "No" if the RESPONSE dismisses the CLAIM as ambiguous or fails to address it meaningfully.

    OUTPUT:
    Only return one of the following: "Yes" or "No".


### 7. Claim-Counterclaim Database

The resulting database pairs each claim with its corresponding counterclaim. Additionally, each claim is stored with its vector representation and text to facilitate retrieval during discussions.

    [
        {
            "claim": "Continued financial aid to Ukraine is counterproductive",
            "counterclaim": "Multiple credible sources, including the World Bank and the U.S. Department of State, report that financial aid to Ukraine has been instrumental in stabilizing its economy ... (Sources: World Bank, U.S. Department of State)",
            "embedding": [ -0.04364201, 0.0033573122, ... ]
        },
        {
            "claim": "Supporting the Neo-Nazi Kiev regime dishonors the memory of Holocaust victims",
            "counterclaim": "Ukraine's government is democratically elected, and its president, Volodymyr Zelensky, is Jewish and lost family members in the Holocaust. (Sources: BBC, The New York Times, Reuters)",
            "embedding": [ -0.04570892, 0.0075604967, ... ]
        },
        ...
    ]


## 8. Document Counterclaim Retrieval

To pair documents with relevant counterclaims during response generation, use the following steps:

1. Process a new document through the claim extraction pipeline (step 9) to identify claims.

2. Convert each extracted claim into a vector representation using the text embedding model from claim filtering (step 10).

3. Retrieve the most relevant counterclaims from the claim-counterclaim database (step 11) by calculating the cosine similarity between the vectors of the identified claims and all claims in the database.

4. Package the document, its identified claims, and retrieved counterclaims, and send them to the LLM for response generation.


## 9. Response Generation

For each document, generate multiple responses using the following methods:

1. **Control**: Generate a response that addresses only superficial issues without engaging deeply with the content.

        DOCUMENT:
        "{text}"

        TASK:
        Write a short response that superficially challenges the main point of the document. Focus only on surface-level issues and avoid providing strong evidence or detailed reasoning.

2. **Generic LLM**: Generate a response using only generic instructions to identify, evaluate, and respond to central claims.

        DOCUMENT:
        "{text}"

        TASK:
        Write a clear, concise, and constructive response that challenges the main point of the document.


3. **LLM with Retrieval-Augmented Generation**: A response that incorporates relevant counterclaims retrieved from the claim-counterclaim database.

        DOCUMENT:
        "{text}"

        RELEVANT INFORMATION:
        {counterclaims_str}

        TASK:
        Using the relevant information provided, write a clear, concise, and constructive response that challenges the main point of the document.

4. **Human Baseline**: The Community Note associated with the tweet if it is rated as "CURRENTLY_RATED_HELPFUL" or "NEEDS_MORE_RATINGS". We discard notes rated as "CURRENTLY_RATED_NOT_HELPFUL".

5. **Human Baseline with RAG**: We use the Community Note with the RAG prompt to standardize response styles and ensure a fairer comparison.

        DOCUMENT:
        "{text}"

        RELEVANT INFORMATION:
        {community_note}

        TASK:
        Using the relevant information provided, write a clear, concise, and constructive response that challenges the main point of the document.


### Response Rules

For all methods, append the following rules to guide response generation:

    RULES:
    - Brevity: Responses must be brief and focused.
    - Neutrality: Avoid emotionally charged, inflammatory, or biased language. Maintain a factual and objective tone.
    - Accuracy: Ensure correct spelling and consistent use of established names or terminology (e.g., countries, organizations, historical figures).
    - Formatting: Do not use any formatting or special characters.
    - Author's Stance: Consider the provided author's stance when crafting the response, but never mention it explicitly.
    - Final Response Only: Provide only the final response. Do not include reasoning, intermediate steps, or explanations in the output.


## 10. Evaluation

Evaluation Dataset: 313 tweets from 15 entities alongside their associated Community Notes.

For each document, collect the responses from the four methods and present them pairwise to human annotators for evaluation.

    1. Control 
    2. Generic LLM Response
    3. LLM with RAG (Our Method)
    4. Community Note
    5. Community Note with RAG  

Annotators rank the four responses based on the following criteria: 

    You are evaluating two responses to the following document. Your task is to determine which response is more convincing challenges the main point of the document.

    Criteria for an Effective Response:
    - Maintains a neutral and factual tone.
    - Directly addresses the main point of the document.
    - Uses relevant facts and evidence to support its argumentation.

    DOCUMENT:
    "{text}"

    RESPONSE A:
    "{response_a}"

    RESPONSE B:
    "{response_b}"

    QUESTION: 
    Which response (A or B) is more effective at challenging the main point of the document?

    Respond ONLY with a single letter: A or B. Do not add any explanations or additional text.


### Platform

- Human annotators: Use Amazon Mechanical Turk (MTurk) to conduct human evaluations.

- LLM as a Judge: Use an LLM to evaluate the responses generated by the LLM.

### Quality Control 

1. Only keep annotations that rank the 'control' response last or 'generic' response last (i.e. the two baseline methods).

2. Assign each document to multiple annotators to ensure reliable rankings.

3. Use inter-annotator agreement metrics (e.g., Cohen’s Kappa) to assess consistency.

### Data Aggregation

Calculate win rates for each method by determining how often a method is ranked first.


## 11. Results

_Quality control: 313 documents passed quality assurance._

    Win Rate (WR) by Category:
    Category              Win Rate
    RAG                   0.76
    community_note_RAG    0.22
    generic               0.02
    control               0.00
    community_note        0.00

The results indicate that the LLM with Retrieval-Augmented Generation (RAG) outperforms other methods, with a win rate of 76%. The Community Note with RAG ranks second at 22%.