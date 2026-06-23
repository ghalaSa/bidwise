# ==================== CELL 1 ====================
!pip install -U openai pdfplumber faiss-cpu -q
!pip install pandas==2.2.2 numpy==2.0.2 scikit-learn==1.6.1 matplotlib -q



# ==================== CELL 3 ====================
import os
import json
import pdfplumber
import numpy as np
import pandas as pd
import faiss

from openai import OpenAI
from google.colab import userdata

client = OpenAI(api_key=userdata.get("new"))

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"



# ==================== CELL 5 ====================
from google.colab import files

print("Upload RFP files")
rfp_uploaded = files.upload()

print("Upload Proposal files")
proposal_uploaded = files.upload()

rfp_files = list(rfp_uploaded.keys())
proposal_files = list(proposal_uploaded.keys())

print("RFP Files:", rfp_files)
print("Proposal Files:", proposal_files)



# ==================== CELL 7 ====================
def extract_text_from_pdf(pdf_path):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                full_text += page_text + "\n"

    return full_text.strip()



# ==================== CELL 9 ====================
rfp_text = extract_text_from_pdf(rfp_files[0])
proposal_text = extract_text_from_pdf(proposal_files[0])

print("RFP length:", len(rfp_text))
print("Proposal length:", len(proposal_text))

print(rfp_text[:1000])



# ==================== CELL 11 ====================
def chunk_text(text, chunk_size=500, overlap=80):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks



# ==================== CELL 13 ====================
rfp_chunks = chunk_text(rfp_text)
proposal_chunks = chunk_text(proposal_text)

print("RFP chunks:", len(rfp_chunks))
print("Proposal chunks:", len(proposal_chunks))

print(proposal_chunks[0])



# ==================== CELL 15 ====================
def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def get_embeddings(texts, batch_size=50):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def normalize_vectors(vectors):
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)
    return vectors



# ==================== CELL 17 ====================
def build_faiss_index(chunks):
    embeddings = get_embeddings(chunks)
    vectors = normalize_vectors(embeddings)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, vectors



# ==================== CELL 19 ====================
proposal_index, proposal_vectors = build_faiss_index(proposal_chunks)

print("FAISS index created")
print("Total chunks in index:", proposal_index.ntotal)



# ==================== CELL 21 ====================
def retrieve_evidence(query, index, chunks, top_k=3):
    query_embedding = get_embedding(query)
    query_vector = normalize_vectors([query_embedding])

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "chunk": chunks[idx],
            "similarity_score": float(score)
        })

    return results



# ==================== CELL 22 ====================
def get_best_chunk_match(rfp_text, proposal_index, proposal_chunks):
    evidence = retrieve_evidence(
        query=rfp_text,
        index=proposal_index,
        chunks=proposal_chunks,
        top_k=1
    )

    best_evidence = evidence[0]

    return {
        "best_chunk": best_evidence["chunk"],
        "chunk_score": best_evidence["similarity_score"]
    }



# ==================== CELL 24 ====================
test_requirement = "The system must provide user authentication and access control"

evidence = retrieve_evidence(
    query=test_requirement,
    index=proposal_index,
    chunks=proposal_chunks,
    top_k=3
)

for i, item in enumerate(evidence, 1):
    print(f"\nEvidence {i}")
    print("Score:", item["similarity_score"])
    print(item["chunk"][:500])



# ==================== CELL 26 ====================
import re

import json

def parse_json_response(response_text):

    response_text = response_text.strip()

    # Remove markdown code block if GPT returns ```json ... ```

    response_text = re.sub(r"^```json", "", response_text)

    response_text = re.sub(r"^```", "", response_text)

    response_text = re.sub(r"```$", "", response_text)

    response_text = response_text.strip()

    try:

        return json.loads(response_text)

    except json.JSONDecodeError:

        # Try to extract JSON object from the text

        match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if match:

            return json.loads(match.group())

        print("Invalid JSON response:")

        print(response_text)

        raise



# ==================== CELL 27 ====================
def extract_rfp_items(rfp_text):
    prompt = f"""
You are an expert procurement analyst.

Extract all important evaluation items from this RFP.

Include:
- Functional requirements
- Technical requirements
- Security requirements
- Integration requirements
- Budget
- Timeline
- Vendor qualifications
- Evaluation criteria
- Deliverables
- Support and warranty requirements
- Team or experience requirements

Return ONLY valid JSON.

Format:
{{
  "rfp_items": [
    {{
      "item": "PDF Upload",
      "category": "Functional Requirements"
    }}
  ]
}}

Rules:
- Do not summarize.
- Do not invent information.
- Keep each item short and clear.
- Use one of these categories only:
  Functional Requirement, Technical Requirement, Security Requirement,
  Integration Requirement, Budget, Timeline, Vendor Qualification,
  Evaluation Criteria, Deliverable, Support/Warranty, Team/Experience Requirement.

RFP TEXT:
{rfp_text}
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract atomic structured requirements from RFP documents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return parse_json_response(response.choices[0].message.content)



# ==================== CELL 29 ====================
def validate_rfp_extraction(rfp_text, extracted_rfp_json):
    prompt = f"""
You are a Requirement Extraction Quality Checker.

Your task is to review the original RFP text and the extracted requirements.

Check if any important requirements were missed, merged incorrectly, duplicated, or categorized incorrectly.

Rules:
- Add missing requirements if they exist in the RFP text.
- Split merged requirements into atomic items.
- Remove duplicate requirements.
- Fix incorrect categories.
- Keep each item short and specific.
- Do not invent requirements not found in the RFP.

Categories:
- Functional Requirements
- Technical Requirements
- Security Requirements
- Integration Requirements
- Non-Functional Requirements
- Budget
- Timeline
- Vendor Qualifications
- Evaluation Criteria
- Deliverables
- Support & Warranty
- Team / Experience Requirements

Return ONLY valid JSON in this format:

{{
  "rfp_items": [
    {{
      "item": "Requirement text",
      "category": "Category name"
    }}
  ]
}}

Original RFP Text:
{rfp_text}

Extracted Requirements:
{json.dumps(extracted_rfp_json, indent=2)}
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You validate and improve extracted RFP requirements."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return parse_json_response(response.choices[0].message.content)



# ==================== CELL 31 ====================
def extract_proposal_items(proposal_text):
    prompt = f"""
You are an AI Proposal Reviewer.

Extract structured proposal capabilities from the following technical proposal.

Include these categories:
- Functional Capabilities
- Technical Capabilities
- Security Controls
- Integrations
- Cost
- Timeline
- Team / Experience
- Deliverables
- Support & Warranty
- Technology Stack

Return ONLY valid JSON in this format:

{{
  "proposal_items": [
    {{
      "item": "Capability text",
      "category": "Category name"
    }}
  ]
}}

PROPOSAL TEXT:
{proposal_text}
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You extract structured capabilities from technical proposal documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return parse_json_response(response.choices[0].message.content)



# ==================== CELL 33 ====================
def validate_proposal_extraction(proposal_text, extracted_proposal_json):
    prompt = f"""
You are a Proposal Capability Extraction Quality Checker.

Your task is to review the original proposal text and the extracted proposal capabilities.

Check if any important capabilities, services, technologies, security controls, integrations, timeline, cost, deliverables, support, or team details were missed.

Rules:
- Add missing capabilities if they exist in the proposal text.
- Split merged capabilities into atomic items.
- Remove duplicate items.
- Fix incorrect categories.
- Keep each item short and specific.
- Do not invent capabilities not found in the proposal.

Categories:
- Functional Capabilities
- Technical Capabilities
- Security Controls
- Integrations
- Cost
- Timeline
- Team / Experience
- Deliverables
- Support & Warranty
- Technology Stack

Return ONLY valid JSON in this format:

{{
  "proposal_items": [
    {{
      "item": "Capability text",
      "category": "Category name"
    }}
  ]
}}

Original Proposal Text:
{proposal_text}

Extracted Proposal Capabilities:
{json.dumps(extracted_proposal_json, indent=2)}
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You validate and improve extracted proposal capabilities."
            },
            {
                "role": "user",
                "content": prompt}
        ],
        temperature=0
    )

    return parse_json_response(response.choices[0].message.content)



# ==================== CELL 35 ====================
rfp_json_raw = extract_rfp_items(rfp_text)

rfp_json = validate_rfp_extraction(
    rfp_text=rfp_text,
    extracted_rfp_json=rfp_json_raw
)

proposal_json_raw = extract_proposal_items(proposal_text)

proposal_json = validate_proposal_extraction(
    proposal_text=proposal_text,
    extracted_proposal_json=proposal_json_raw
)



# ==================== CELL 36 ====================
print("RFP Items after validation:", len(rfp_json["rfp_items"]))
print("Proposal Items after validation:", len(proposal_json["proposal_items"]))
print("RFP:", len(rfp_json_raw["rfp_items"]))
print("Proposal :", len(proposal_json["proposal_items"]))

display(pd.DataFrame(rfp_json["rfp_items"]).head(30))
display(pd.DataFrame(proposal_json_raw["proposal_items"]).head(30))



# ==================== CELL 37 ====================
for item in rfp_json["rfp_items"]:
    print(item)



# ==================== CELL 38 ====================
for item in proposal_json["proposal_items"]:
    print(item)



# ==================== CELL 40 ====================
rfp_items_df = pd.DataFrame(rfp_json["rfp_items"])
proposal_items_df = pd.DataFrame(proposal_json["proposal_items"])

display(rfp_items_df.head())
display(proposal_items_df.head())



# ==================== CELL 42 ====================
def exact_match_bonus(rfp_item, proposal_item):
    rfp_words = set(rfp_item.lower().split())
    proposal_words = set(proposal_item.lower().split())

    overlap = rfp_words.intersection(proposal_words)

    if len(overlap) >= 2:
        return 0.05

    return 0.0


def category_bonus(rfp_category, proposal_category):
    rfp_category = rfp_category.lower().strip()
    proposal_category = proposal_category.lower().strip()

    if rfp_category in proposal_category or proposal_category in rfp_category:
        return 0.05

    return 0.0



# ==================== CELL 43 ====================
def classify_match_status(score):
    if score >= 0.70:
        return "Strong Match"
    elif score >= 0.50:
        return "Partial Match"
    else:
        return "Missing"



# ==================== CELL 44 ====================
def semantic_match_rfp_to_proposal_fast(
    rfp_items,
    proposal_items,
    proposal_index,
    proposal_chunks
):
    rfp_texts = [item["item"] for item in rfp_items]
    proposal_texts = [item["item"] for item in proposal_items]

    rfp_vectors = normalize_vectors(get_embeddings(rfp_texts))
    proposal_vectors = normalize_vectors(get_embeddings(proposal_texts))

    item_index = faiss.IndexFlatIP(proposal_vectors.shape[1])
    item_index.add(proposal_vectors)

    item_scores, item_indices = item_index.search(rfp_vectors, 1)
    chunk_scores, chunk_indices = proposal_index.search(rfp_vectors, 1)

    matching_results = []

    for i, rfp in enumerate(rfp_items):
        rfp_text = rfp["item"]
        rfp_category = rfp.get("category", "Unknown")

        best_item_idx = item_indices[i][0]
        best_item = proposal_items[best_item_idx]

        item_score = float(item_scores[i][0])
        item_score += exact_match_bonus(rfp_text, best_item["item"])
        item_score += category_bonus(rfp_category, best_item.get("category", "Unknown"))
        item_score = min(item_score, 1.0)

        chunk_score = float(chunk_scores[i][0])
        best_chunk = proposal_chunks[chunk_indices[i][0]]

        if item_score >= chunk_score:
            final_score = item_score
            best_match = best_item["item"]
            match_source = "Proposal Item"
        else:
            final_score = chunk_score
            best_match = best_chunk[:300]
            match_source = "Proposal Chunk"

        matching_results.append({
            "rfp_item": rfp_text,
            "rfp_category": rfp_category,
            "best_proposal_match": best_match,
            "similarity_score": round(final_score, 3),
            "status": classify_match_status(final_score),
            "match_source": match_source
        })

    return matching_results



# ==================== CELL 46 ====================
matching_results = semantic_match_rfp_to_proposal_fast(
    rfp_items=rfp_json["rfp_items"],
    proposal_items=proposal_json["proposal_items"],
    proposal_index=proposal_index,
    proposal_chunks=proposal_chunks
)

matching_df = pd.DataFrame(matching_results)
display(matching_df)

print(matching_df["status"].value_counts())
print(matching_df["match_source"].value_counts())



# ==================== CELL 47 ====================
display(
    matching_df[
        ["rfp_item", "best_proposal_match", "similarity_score", "status"]
    ].sort_values("similarity_score", ascending=False).head(20)
)



# ==================== CELL 49 ====================
def calculate_compliance_score(matching_results):
    total_items = len(matching_results)

    strong_matches = sum(1 for item in matching_results if item["status"] == "Strong Match")
    partial_matches = sum(1 for item in matching_results if item["status"] == "Partial Match")
    missing_items = sum(1 for item in matching_results if item["status"] == "Missing")

    compliance_score = ((strong_matches + (0.5 * partial_matches)) / total_items) * 100

    avg_similarity = np.mean([item["similarity_score"] for item in matching_results])

    return {
        "ComplianceScore": round(compliance_score, 2),
        "StrongMatches": strong_matches,
        "PartialMatches": partial_matches,
        "MissingItems": missing_items,
        "TotalRequirements": total_items,
        "AvgSimilarityScore": float(round(avg_similarity, 3))
    }



# ==================== CELL 50 ====================
compliance_result = calculate_compliance_score(matching_results)
compliance_result



# ==================== CELL 52 ====================
missing_requirements = [
    item for item in matching_results
    if item["status"] == "Missing"
]

missing_df = pd.DataFrame(missing_requirements)
display(missing_df)



# ==================== CELL 54 ====================
def assign_risk_type(category, item):

    category = str(category).lower()
    item = str(item).lower()

    if "security" in category:
        return "Security Risk"

    elif "integration" in category:
        return "Integration Risk"

    elif "timeline" in category:
        return "Schedule Risk"

    elif "budget" in category:
        return "Cost Risk"

    elif "vendor" in category:
        return "Vendor Capability Risk"

    elif "support & warranty" in category:
        return "Support Risk"

    elif "evaluation" in category:
        return "Evaluation Risk"

    elif "deliverable" in category:
        return "Deliverable Risk"

    elif "non-functional" in category:
        return "Performance Risk"

    else:
        return "Functional/Technical Risk"



# ==================== CELL 55 ====================
def analyze_risks(matching_results):

    risks = []

    for item in matching_results:

        if item["status"] == "Missing":

            risks.append({
                "missing_requirement": item["rfp_item"],
                "category": item["rfp_category"],
                "risk_type": assign_risk_type(
                    item["rfp_category"],
                    item["rfp_item"]
                ),
                "similarity_score": item["similarity_score"]
            })

    return risks



# ==================== CELL 56 ====================
risk_results = analyze_risks(matching_results)

risk_df = pd.DataFrame(risk_results)
display(risk_df)

print(risk_df["risk_type"].value_counts())



# ==================== CELL 58 ====================
def calculate_category_coverage(matching_df, category_name):
    category_rows = matching_df[
        matching_df["rfp_category"] == category_name
    ]

    if len(category_rows) == 0:
        return 0

    covered_rows = category_rows[
        category_rows["status"] != "Missing"
    ]

    return round((len(covered_rows) / len(category_rows)) * 100, 2)



# ==================== CELL 59 ====================
risk_counts = risk_df["risk_type"].value_counts().to_dict()

project_features = {
    "ComplianceScore": compliance_result["ComplianceScore"],
    "StrongMatches": compliance_result["StrongMatches"],
    "PartialMatches": compliance_result["PartialMatches"],
    "MissingItems": compliance_result["MissingItems"],
    "TotalRequirements": compliance_result["TotalRequirements"],
    "AvgSimilarityScore": compliance_result["AvgSimilarityScore"],

    "FunctionalCoverage": calculate_category_coverage(matching_df, "Functional Requirements"),
    "TechnicalCoverage": calculate_category_coverage(matching_df, "Technical Requirements"),
    "SecurityCoverage": calculate_category_coverage(matching_df, "Security Requirements"),
    "IntegrationCoverage": calculate_category_coverage(matching_df, "Integration Requirements"),
    "NonFunctionalCoverage": calculate_category_coverage(matching_df, "Non-Functional Requirements"),
    "BudgetCoverage": calculate_category_coverage(matching_df, "Budget"),
    "TimelineCoverage": calculate_category_coverage(matching_df, "Timeline"),
    "VendorQualificationCoverage": calculate_category_coverage(matching_df, "Vendor Qualifications"),
    "EvaluationCriteriaCoverage": calculate_category_coverage(matching_df, "Evaluation Criteria"),
    "DeliverableCoverage": calculate_category_coverage(matching_df, "Deliverables"),
    "SupportWarrantyCoverage": calculate_category_coverage(matching_df, "Support & Warranty"),

    "SecurityRiskCount": risk_counts.get("Security Risk", 0),
    "IntegrationRiskCount": risk_counts.get("Integration Risk", 0),
    "ScheduleRiskCount": risk_counts.get("Schedule Risk", 0),
    "CostRiskCount": risk_counts.get("Cost Risk", 0),
    "VendorRiskCount": risk_counts.get("Vendor Capability Risk", 0),
    "PerformanceRiskCount": risk_counts.get("Performance Risk", 0),
    "EvaluationRiskCount": risk_counts.get("Evaluation Risk", 0),
    "DeliverableRiskCount": risk_counts.get("Deliverable Risk", 0),
    "FunctionalTechnicalRiskCount": risk_counts.get("Functional/Technical Risk", 0)
}

feature_df = pd.DataFrame([project_features])
display(feature_df)



# ==================== CELL 60 ====================
def assign_win_probability_label(compliance_score, missing_items, security_risk_count):
    if compliance_score >= 75 and missing_items <= 10 and security_risk_count == 0:
        return "High"

    elif compliance_score >= 45 and missing_items <= 30:
        return "Medium"

    else:
        return "Low"



# ==================== CELL 61 ====================
feature_df["WinProbabilityLevel"] = feature_df.apply(
    lambda row: assign_win_probability_label(
        compliance_score=row["ComplianceScore"],
        missing_items=row["MissingItems"],
        security_risk_count=row["SecurityRiskCount"]
    ),
    axis=1
)

display(feature_df)



# ==================== CELL 62 ====================
training_data = [

# ======================
# HIGH
# ======================

{
"ComplianceScore": 91,
"StrongMatches": 65,
"PartialMatches": 8,
"MissingItems": 4,
"AvgSimilarityScore": 0.89,
"FunctionalCoverage": 95,
"TechnicalCoverage": 92,
"SecurityCoverage": 100,
"IntegrationCoverage": 90,
"TimelineCoverage": 100,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 100,
"EvaluationCriteriaCoverage": 100,
"DeliverableCoverage": 95,
"SupportWarrantyCoverage": 100,
"SecurityRiskCount": 0,
"IntegrationRiskCount": 0,
"ScheduleRiskCount": 0,
"PerformanceRiskCount": 0,
"FunctionalTechnicalRiskCount": 1,
"WinProbabilityLevel": "High"
},

{
"ComplianceScore": 88,
"StrongMatches": 60,
"PartialMatches": 10,
"MissingItems": 7,
"AvgSimilarityScore": 0.86,
"FunctionalCoverage": 90,
"TechnicalCoverage": 92,
"SecurityCoverage": 90,
"IntegrationCoverage": 95,
"TimelineCoverage": 100,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 100,
"EvaluationCriteriaCoverage": 90,
"DeliverableCoverage": 95,
"SupportWarrantyCoverage": 100,
"SecurityRiskCount": 0,
"IntegrationRiskCount": 1,
"ScheduleRiskCount": 0,
"PerformanceRiskCount": 0,
"FunctionalTechnicalRiskCount": 2,
"WinProbabilityLevel": "High"
},

{
"ComplianceScore": 84,
"StrongMatches": 57,
"PartialMatches": 12,
"MissingItems": 8,
"AvgSimilarityScore": 0.83,
"FunctionalCoverage": 88,
"TechnicalCoverage": 90,
"SecurityCoverage": 85,
"IntegrationCoverage": 90,
"TimelineCoverage": 90,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 100,
"EvaluationCriteriaCoverage": 90,
"DeliverableCoverage": 90,
"SupportWarrantyCoverage": 90,
"SecurityRiskCount": 1,
"IntegrationRiskCount": 1,
"ScheduleRiskCount": 1,
"PerformanceRiskCount": 0,
"FunctionalTechnicalRiskCount": 2,
"WinProbabilityLevel": "High"
},

{
"ComplianceScore": 81,
"StrongMatches": 55,
"PartialMatches": 11,
"MissingItems": 11,
"AvgSimilarityScore": 0.80,
"FunctionalCoverage": 85,
"TechnicalCoverage": 88,
"SecurityCoverage": 85,
"IntegrationCoverage": 88,
"TimelineCoverage": 85,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 90,
"EvaluationCriteriaCoverage": 90,
"DeliverableCoverage": 88,
"SupportWarrantyCoverage": 90,
"SecurityRiskCount": 1,
"IntegrationRiskCount": 1,
"ScheduleRiskCount": 1,
"PerformanceRiskCount": 1,
"FunctionalTechnicalRiskCount": 3,
"WinProbabilityLevel": "High"
},

{
"ComplianceScore": 78,
"StrongMatches": 52,
"PartialMatches": 12,
"MissingItems": 13,
"AvgSimilarityScore": 0.78,
"FunctionalCoverage": 82,
"TechnicalCoverage": 85,
"SecurityCoverage": 80,
"IntegrationCoverage": 85,
"TimelineCoverage": 85,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 90,
"EvaluationCriteriaCoverage": 85,
"DeliverableCoverage": 85,
"SupportWarrantyCoverage": 85,
"SecurityRiskCount": 1,
"IntegrationRiskCount": 2,
"ScheduleRiskCount": 1,
"PerformanceRiskCount": 1,
"FunctionalTechnicalRiskCount": 3,
"WinProbabilityLevel": "High"
},

# ======================
# MEDIUM
# ======================

{
"ComplianceScore": 68,
"StrongMatches": 42,
"PartialMatches": 16,
"MissingItems": 19,
"AvgSimilarityScore": 0.70,
"FunctionalCoverage": 72,
"TechnicalCoverage": 70,
"SecurityCoverage": 65,
"IntegrationCoverage": 70,
"TimelineCoverage": 80,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 75,
"EvaluationCriteriaCoverage": 70,
"DeliverableCoverage": 75,
"SupportWarrantyCoverage": 70,
"SecurityRiskCount": 2,
"IntegrationRiskCount": 2,
"ScheduleRiskCount": 2,
"PerformanceRiskCount": 1,
"FunctionalTechnicalRiskCount": 4,
"WinProbabilityLevel": "Medium"
},

{
"ComplianceScore": 63,
"StrongMatches": 39,
"PartialMatches": 15,
"MissingItems": 23,
"AvgSimilarityScore": 0.66,
"FunctionalCoverage": 68,
"TechnicalCoverage": 65,
"SecurityCoverage": 60,
"IntegrationCoverage": 65,
"TimelineCoverage": 70,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 70,
"EvaluationCriteriaCoverage": 60,
"DeliverableCoverage": 70,
"SupportWarrantyCoverage": 65,
"SecurityRiskCount": 2,
"IntegrationRiskCount": 3,
"ScheduleRiskCount": 2,
"PerformanceRiskCount": 1,
"FunctionalTechnicalRiskCount": 5,
"WinProbabilityLevel": "Medium"
},

{
"ComplianceScore": 59,
"StrongMatches": 35,
"PartialMatches": 16,
"MissingItems": 26,
"AvgSimilarityScore": 0.64,
"FunctionalCoverage": 62,
"TechnicalCoverage": 60,
"SecurityCoverage": 55,
"IntegrationCoverage": 60,
"TimelineCoverage": 65,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 65,
"EvaluationCriteriaCoverage": 60,
"DeliverableCoverage": 60,
"SupportWarrantyCoverage": 60,
"SecurityRiskCount": 3,
"IntegrationRiskCount": 3,
"ScheduleRiskCount": 3,
"PerformanceRiskCount": 1,
"FunctionalTechnicalRiskCount": 5,
"WinProbabilityLevel": "Medium"
},

{
"ComplianceScore": 55,
"StrongMatches": 32,
"PartialMatches": 14,
"MissingItems": 31,
"AvgSimilarityScore": 0.60,
"FunctionalCoverage": 58,
"TechnicalCoverage": 55,
"SecurityCoverage": 50,
"IntegrationCoverage": 55,
"TimelineCoverage": 60,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 60,
"EvaluationCriteriaCoverage": 55,
"DeliverableCoverage": 60,
"SupportWarrantyCoverage": 55,
"SecurityRiskCount": 3,
"IntegrationRiskCount": 3,
"ScheduleRiskCount": 3,
"PerformanceRiskCount": 2,
"FunctionalTechnicalRiskCount": 6,
"WinProbabilityLevel": "Medium"
},

{
"ComplianceScore": 50,
"StrongMatches": 30,
"PartialMatches": 12,
"MissingItems": 35,
"AvgSimilarityScore": 0.58,
"FunctionalCoverage": 55,
"TechnicalCoverage": 50,
"SecurityCoverage": 45,
"IntegrationCoverage": 50,
"TimelineCoverage": 55,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 55,
"EvaluationCriteriaCoverage": 50,
"DeliverableCoverage": 55,
"SupportWarrantyCoverage": 50,
"SecurityRiskCount": 4,
"IntegrationRiskCount": 4,
"ScheduleRiskCount": 3,
"PerformanceRiskCount": 2,
"FunctionalTechnicalRiskCount": 6,
"WinProbabilityLevel": "Medium"
},

# ======================
# LOW
# ======================

{
"ComplianceScore": 39,
"StrongMatches": 18,
"PartialMatches": 25,
"MissingItems": 34,
"AvgSimilarityScore": 0.60,
"FunctionalCoverage": 45,
"TechnicalCoverage": 40,
"SecurityCoverage": 30,
"IntegrationCoverage": 25,
"TimelineCoverage": 20,
"BudgetCoverage": 100,
"VendorQualificationCoverage": 40,
"EvaluationCriteriaCoverage": 25,
"DeliverableCoverage": 40,
"SupportWarrantyCoverage": 35,
"SecurityRiskCount": 4,
"IntegrationRiskCount": 5,
"ScheduleRiskCount": 4,
"PerformanceRiskCount": 3,
"FunctionalTechnicalRiskCount": 8,
"WinProbabilityLevel": "Low"
},

{
"ComplianceScore": 35,
"StrongMatches": 15,
"PartialMatches": 20,
"MissingItems": 42,
"AvgSimilarityScore": 0.55,
"FunctionalCoverage": 40,
"TechnicalCoverage": 35,
"SecurityCoverage": 25,
"IntegrationCoverage": 20,
"TimelineCoverage": 20,
"BudgetCoverage": 50,
"VendorQualificationCoverage": 35,
"EvaluationCriteriaCoverage": 20,
"DeliverableCoverage": 35,
"SupportWarrantyCoverage": 30,
"SecurityRiskCount": 5,
"IntegrationRiskCount": 5,
"ScheduleRiskCount": 4,
"PerformanceRiskCount": 3,
"FunctionalTechnicalRiskCount": 9,
"WinProbabilityLevel": "Low"
},

{
"ComplianceScore": 30,
"StrongMatches": 12,
"PartialMatches": 18,
"MissingItems": 47,
"AvgSimilarityScore": 0.52,
"FunctionalCoverage": 35,
"TechnicalCoverage": 30,
"SecurityCoverage": 20,
"IntegrationCoverage": 15,
"TimelineCoverage": 15,
"BudgetCoverage": 0,
"VendorQualificationCoverage": 30,
"EvaluationCriteriaCoverage": 15,
"DeliverableCoverage": 30,
"SupportWarrantyCoverage": 25,
"WinProbabilityLevel": "Low"
},

{
"ComplianceScore": 25,
"StrongMatches": 10,
"PartialMatches": 15,
"MissingItems": 52,
"AvgSimilarityScore": 0.48,
"FunctionalCoverage": 30,
"TechnicalCoverage": 25,
"SecurityCoverage": 15,
"IntegrationCoverage": 10,
"TimelineCoverage": 10,
"BudgetCoverage": 0,
"VendorQualificationCoverage": 20,
"EvaluationCriteriaCoverage": 10,
"DeliverableCoverage": 20,
"SupportWarrantyCoverage": 20,
"WinProbabilityLevel": "Low"
},

{
"ComplianceScore": 18,
"StrongMatches": 5,
"PartialMatches": 12,
"MissingItems": 60,
"AvgSimilarityScore": 0.42,
"FunctionalCoverage": 20,
"TechnicalCoverage": 15,
"SecurityCoverage": 10,
"IntegrationCoverage": 5,
"TimelineCoverage": 0,
"BudgetCoverage": 0,
"VendorQualificationCoverage": 10,
"EvaluationCriteriaCoverage": 0,
"DeliverableCoverage": 10,
"SupportWarrantyCoverage": 10,
"WinProbabilityLevel": "Low"
}

]



# ==================== CELL 63 ====================




# ==================== CELL 64 ====================
train_df = pd.DataFrame(training_data)
train_df



# ==================== CELL 65 ====================
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

train_data = train_df.copy()

label_encoder = LabelEncoder()

train_data["WinProbabilityLevel"] = label_encoder.fit_transform(
    train_data["WinProbabilityLevel"]
)

X_train = train_data.drop(
    columns=["WinProbabilityLevel"]
)

y_train = train_data["WinProbabilityLevel"]

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)



# ==================== CELL 66 ====================
X_test = feature_df[X_train.columns]



# ==================== CELL 67 ====================
prediction = rf_model.predict(X_test)[0]

predicted_label = label_encoder.inverse_transform(
    [prediction]
)[0]

print(predicted_label)



# ==================== CELL 68 ====================
probabilities = rf_model.predict_proba(X_test)[0]

for cls, prob in zip(
    label_encoder.classes_,
    probabilities
):
    print(cls, round(prob * 100, 2), "%")



# ==================== CELL 69 ====================
win_probability = max(probabilities) * 100

print(
    f"Win Probability: {win_probability:.2f}%"
)



# ==================== CELL 71 ====================
missing_requirements_list = list(
    risk_df["missing_requirement"]
)

risk_summary = risk_df["risk_type"].value_counts().to_dict()



# ==================== CELL 72 ====================
def generate_recommendations(
    compliance_result,
    predicted_label,
    win_probability,
    missing_requirements,
    risk_summary
):

    prompt = f"""
You are a Senior Proposal Review Consultant.

Analyze this proposal evaluation result.

Compliance Score:
{compliance_result}

Win Probability Level:
{predicted_label}

Win Probability:
{win_probability:.2f}%

Missing Requirements:
{missing_requirements}

Risk Summary:
{risk_summary}

Provide:

1. Strengths
2. Weaknesses
3. Recommendations
4. Estimated Win Improvement

Return ONLY valid JSON:

{{
  "strengths": [],
  "weaknesses": [],
  "recommendations": [],
  "estimated_win_improvement": ""
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role":"system",
                "content":"You are a proposal evaluation expert."
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.3
    )

    return parse_json_response(
        response.choices[0].message.content
    )



# ==================== CELL 73 ====================
recommendation_result = generate_recommendations(
    compliance_result=compliance_result,
    predicted_label=predicted_label,
    win_probability=win_probability,
    missing_requirements=missing_requirements_list,
    risk_summary=risk_summary
)
def display_recommendation_report(recommendation_result):
    print("=" * 60)
    print("BidWise Recommendation Report")
    print("=" * 60)

    print("\nStrengths:")
    for i, item in enumerate(recommendation_result["strengths"], 1):
        print(f"{i}. {item}")

    print("\nWeaknesses:")
    for i, item in enumerate(recommendation_result["weaknesses"], 1):
        print(f"{i}. {item}")

    print("\nRecommendations:")
    for i, item in enumerate(recommendation_result["recommendations"], 1):
        print(f"{i}. {item}")

    print("\nEstimated Win Improvement:")
    print(recommendation_result["estimated_win_improvement"])

display_recommendation_report(recommendation_result)



# ==================== CELL 74 ====================



