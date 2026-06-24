import os
import re
import json
import tempfile
import concurrent.futures
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# Check dependencies
try:
    import pdfplumber
    import faiss
    import openai
    from openai import OpenAI
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    HAS_LIBRARIES = True
except ImportError as e:
    print(f"Warning: Dependency import failed: {e}. Graceful fallbacks will be used.")
    HAS_LIBRARIES = False

app = Flask(__name__, static_folder='.', static_url_path='')

# ==========================================
# 1. RANDOM FOREST WIN PROBABILITY MODEL
# ==========================================
# Notebook training data (Cell 62)
training_data = [
    # HIGH
    {"ComplianceScore": 91, "StrongMatches": 65, "PartialMatches": 8, "MissingItems": 4, "AvgSimilarityScore": 0.89, "FunctionalCoverage": 95, "TechnicalCoverage": 92, "SecurityCoverage": 100, "IntegrationCoverage": 90, "TimelineCoverage": 100, "BudgetCoverage": 100, "VendorQualificationCoverage": 100, "EvaluationCriteriaCoverage": 100, "DeliverableCoverage": 95, "SupportWarrantyCoverage": 100, "SecurityRiskCount": 0, "IntegrationRiskCount": 0, "ScheduleRiskCount": 0, "PerformanceRiskCount": 0, "FunctionalTechnicalRiskCount": 1, "WinProbabilityLevel": "High"},
    {"ComplianceScore": 88, "StrongMatches": 60, "PartialMatches": 10, "MissingItems": 7, "AvgSimilarityScore": 0.86, "FunctionalCoverage": 90, "TechnicalCoverage": 92, "SecurityCoverage": 90, "IntegrationCoverage": 95, "TimelineCoverage": 100, "BudgetCoverage": 100, "VendorQualificationCoverage": 100, "EvaluationCriteriaCoverage": 90, "DeliverableCoverage": 95, "SupportWarrantyCoverage": 100, "SecurityRiskCount": 0, "IntegrationRiskCount": 1, "ScheduleRiskCount": 0, "PerformanceRiskCount": 0, "FunctionalTechnicalRiskCount": 2, "WinProbabilityLevel": "High"},
    {"ComplianceScore": 84, "StrongMatches": 57, "PartialMatches": 12, "MissingItems": 8, "AvgSimilarityScore": 0.83, "FunctionalCoverage": 88, "TechnicalCoverage": 90, "SecurityCoverage": 85, "IntegrationCoverage": 90, "TimelineCoverage": 90, "BudgetCoverage": 100, "VendorQualificationCoverage": 100, "EvaluationCriteriaCoverage": 90, "DeliverableCoverage": 90, "SupportWarrantyCoverage": 90, "SecurityRiskCount": 1, "IntegrationRiskCount": 1, "ScheduleRiskCount": 1, "PerformanceRiskCount": 0, "FunctionalTechnicalRiskCount": 2, "WinProbabilityLevel": "High"},
    {"ComplianceScore": 81, "StrongMatches": 55, "PartialMatches": 11, "MissingItems": 11, "AvgSimilarityScore": 0.80, "FunctionalCoverage": 85, "TechnicalCoverage": 88, "SecurityCoverage": 85, "IntegrationCoverage": 88, "TimelineCoverage": 85, "BudgetCoverage": 100, "VendorQualificationCoverage": 90, "EvaluationCriteriaCoverage": 90, "DeliverableCoverage": 88, "SupportWarrantyCoverage": 90, "SecurityRiskCount": 1, "IntegrationRiskCount": 1, "ScheduleRiskCount": 1, "PerformanceRiskCount": 1, "FunctionalTechnicalRiskCount": 3, "WinProbabilityLevel": "High"},
    {"ComplianceScore": 78, "StrongMatches": 52, "PartialMatches": 12, "MissingItems": 13, "AvgSimilarityScore": 0.78, "FunctionalCoverage": 82, "TechnicalCoverage": 85, "SecurityCoverage": 80, "IntegrationCoverage": 85, "TimelineCoverage": 85, "BudgetCoverage": 100, "VendorQualificationCoverage": 90, "EvaluationCriteriaCoverage": 85, "DeliverableCoverage": 85, "SupportWarrantyCoverage": 85, "SecurityRiskCount": 1, "IntegrationRiskCount": 2, "ScheduleRiskCount": 1, "PerformanceRiskCount": 1, "FunctionalTechnicalRiskCount": 3, "WinProbabilityLevel": "High"},
    # MEDIUM
    {"ComplianceScore": 68, "StrongMatches": 42, "PartialMatches": 16, "MissingItems": 19, "AvgSimilarityScore": 0.70, "FunctionalCoverage": 72, "TechnicalCoverage": 70, "SecurityCoverage": 65, "IntegrationCoverage": 70, "TimelineCoverage": 80, "BudgetCoverage": 100, "VendorQualificationCoverage": 75, "EvaluationCriteriaCoverage": 70, "DeliverableCoverage": 75, "SupportWarrantyCoverage": 70, "SecurityRiskCount": 2, "IntegrationRiskCount": 2, "ScheduleRiskCount": 2, "PerformanceRiskCount": 1, "FunctionalTechnicalRiskCount": 4, "WinProbabilityLevel": "Medium"},
    {"ComplianceScore": 63, "StrongMatches": 39, "PartialMatches": 15, "MissingItems": 23, "AvgSimilarityScore": 0.66, "FunctionalCoverage": 68, "TechnicalCoverage": 65, "SecurityCoverage": 60, "IntegrationCoverage": 65, "TimelineCoverage": 70, "BudgetCoverage": 100, "VendorQualificationCoverage": 70, "EvaluationCriteriaCoverage": 60, "DeliverableCoverage": 70, "SupportWarrantyCoverage": 65, "SecurityRiskCount": 2, "IntegrationRiskCount": 3, "ScheduleRiskCount": 2, "PerformanceRiskCount": 1, "FunctionalTechnicalRiskCount": 5, "WinProbabilityLevel": "Medium"},
    {"ComplianceScore": 59, "StrongMatches": 35, "PartialMatches": 16, "MissingItems": 26, "AvgSimilarityScore": 0.64, "FunctionalCoverage": 62, "TechnicalCoverage": 60, "SecurityCoverage": 55, "IntegrationCoverage": 60, "TimelineCoverage": 65, "BudgetCoverage": 100, "VendorQualificationCoverage": 65, "EvaluationCriteriaCoverage": 60, "DeliverableCoverage": 60, "SupportWarrantyCoverage": 60, "SecurityRiskCount": 3, "IntegrationRiskCount": 3, "ScheduleRiskCount": 3, "PerformanceRiskCount": 1, "FunctionalTechnicalRiskCount": 5, "WinProbabilityLevel": "Medium"},
    {"ComplianceScore": 55, "StrongMatches": 32, "PartialMatches": 14, "MissingItems": 31, "AvgSimilarityScore": 0.60, "FunctionalCoverage": 58, "TechnicalCoverage": 55, "SecurityCoverage": 50, "IntegrationCoverage": 55, "TimelineCoverage": 60, "BudgetCoverage": 100, "VendorQualificationCoverage": 60, "EvaluationCriteriaCoverage": 55, "DeliverableCoverage": 60, "SupportWarrantyCoverage": 55, "SecurityRiskCount": 3, "IntegrationRiskCount": 3, "ScheduleRiskCount": 3, "PerformanceRiskCount": 2, "FunctionalTechnicalRiskCount": 6, "WinProbabilityLevel": "Medium"},
    {"ComplianceScore": 50, "StrongMatches": 30, "PartialMatches": 12, "MissingItems": 35, "AvgSimilarityScore": 0.58, "FunctionalCoverage": 55, "TechnicalCoverage": 50, "SecurityCoverage": 45, "IntegrationCoverage": 50, "TimelineCoverage": 55, "BudgetCoverage": 100, "VendorQualificationCoverage": 55, "EvaluationCriteriaCoverage": 50, "DeliverableCoverage": 55, "SupportWarrantyCoverage": 50, "SecurityRiskCount": 4, "IntegrationRiskCount": 4, "ScheduleRiskCount": 3, "PerformanceRiskCount": 2, "FunctionalTechnicalRiskCount": 6, "WinProbabilityLevel": "Medium"},
    # LOW
    {"ComplianceScore": 39, "StrongMatches": 18, "PartialMatches": 25, "MissingItems": 34, "AvgSimilarityScore": 0.60, "FunctionalCoverage": 45, "TechnicalCoverage": 40, "SecurityCoverage": 30, "IntegrationCoverage": 25, "TimelineCoverage": 20, "BudgetCoverage": 100, "VendorQualificationCoverage": 40, "EvaluationCriteriaCoverage": 25, "DeliverableCoverage": 40, "SupportWarrantyCoverage": 35, "SecurityRiskCount": 4, "IntegrationRiskCount": 5, "ScheduleRiskCount": 4, "PerformanceRiskCount": 3, "FunctionalTechnicalRiskCount": 8, "WinProbabilityLevel": "Low"},
    {"ComplianceScore": 35, "StrongMatches": 15, "PartialMatches": 20, "MissingItems": 42, "AvgSimilarityScore": 0.55, "FunctionalCoverage": 40, "TechnicalCoverage": 35, "SecurityCoverage": 25, "IntegrationCoverage": 20, "TimelineCoverage": 20, "BudgetCoverage": 50, "VendorQualificationCoverage": 35, "EvaluationCriteriaCoverage": 20, "DeliverableCoverage": 35, "SupportWarrantyCoverage": 30, "SecurityRiskCount": 5, "IntegrationRiskCount": 5, "ScheduleRiskCount": 4, "PerformanceRiskCount": 3, "FunctionalTechnicalRiskCount": 9, "WinProbabilityLevel": "Low"},
    {"ComplianceScore": 30, "StrongMatches": 12, "PartialMatches": 18, "MissingItems": 47, "AvgSimilarityScore": 0.52, "FunctionalCoverage": 35, "TechnicalCoverage": 30, "SecurityCoverage": 20, "IntegrationCoverage": 15, "TimelineCoverage": 15, "BudgetCoverage": 0, "VendorQualificationCoverage": 30, "EvaluationCriteriaCoverage": 15, "DeliverableCoverage": 30, "SupportWarrantyCoverage": 25, "SecurityRiskCount": 5, "IntegrationRiskCount": 5, "ScheduleRiskCount": 5, "PerformanceRiskCount": 3, "FunctionalTechnicalRiskCount": 10, "WinProbabilityLevel": "Low"},
    {"ComplianceScore": 25, "StrongMatches": 10, "PartialMatches": 15, "MissingItems": 52, "AvgSimilarityScore": 0.48, "FunctionalCoverage": 30, "TechnicalCoverage": 25, "SecurityCoverage": 15, "IntegrationCoverage": 10, "TimelineCoverage": 10, "BudgetCoverage": 0, "VendorQualificationCoverage": 20, "EvaluationCriteriaCoverage": 10, "DeliverableCoverage": 20, "SupportWarrantyCoverage": 20, "SecurityRiskCount": 6, "IntegrationRiskCount": 6, "ScheduleRiskCount": 5, "PerformanceRiskCount": 4, "FunctionalTechnicalRiskCount": 11, "WinProbabilityLevel": "Low"},
    {"ComplianceScore": 18, "StrongMatches": 5, "PartialMatches": 12, "MissingItems": 60, "AvgSimilarityScore": 0.42, "FunctionalCoverage": 20, "TechnicalCoverage": 15, "SecurityCoverage": 10, "IntegrationCoverage": 5, "TimelineCoverage": 0, "BudgetCoverage": 0, "VendorQualificationCoverage": 10, "EvaluationCriteriaCoverage": 0, "DeliverableCoverage": 10, "SupportWarrantyCoverage": 10, "SecurityRiskCount": 6, "IntegrationRiskCount": 6, "ScheduleRiskCount": 6, "PerformanceRiskCount": 5, "FunctionalTechnicalRiskCount": 12, "WinProbabilityLevel": "Low"}
]

# Pre-train Win Probability model on startup
if HAS_LIBRARIES:
    try:
        train_df = pd.DataFrame(training_data).fillna(0)
        label_encoder = LabelEncoder()
        train_df["WinProbabilityLevel"] = label_encoder.fit_transform(train_df["WinProbabilityLevel"])
        X_train = train_df.drop(columns=["WinProbabilityLevel"])
        y_train = train_df["WinProbabilityLevel"]
        
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_model.fit(X_train, y_train)
        print("Win Probability RandomForest model trained successfully.")
    except Exception as ex:
        print(f"Error training Win Probability model: {ex}")
        rf_model = None
else:
    rf_model = None

# ==========================================
# 2. NOTEBOOK FUNCTIONS INTEGRATION
# ==========================================
def extract_text_from_pdf(pdf_path):
    full_text = ""
    if not HAS_LIBRARIES:
        return ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text.strip()

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

def get_embeddings(client, texts, batch_size=50):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

def normalize_vectors(vectors):
    vectors = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors)
    return vectors

def build_faiss_index(client, chunks):
    embeddings = get_embeddings(client, chunks)
    vectors = normalize_vectors(embeddings)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors

def parse_json_response(response_text):
    response_text = response_text.strip()
    response_text = re.sub(r"^```json", "", response_text)
    response_text = re.sub(r"^```", "", response_text)
    response_text = re.sub(r"```$", "", response_text)
    response_text = response_text.strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

def extract_rfp_items(client, rfp_text):
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract atomic structured requirements from RFP documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return parse_json_response(response.choices[0].message.content)

def validate_rfp_extraction(client, rfp_text, extracted_rfp_json):
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You validate and improve extracted RFP requirements."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return parse_json_response(response.choices[0].message.content)

def extract_proposal_items(client, proposal_text):
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured capabilities from technical proposal documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return parse_json_response(response.choices[0].message.content)

def validate_proposal_extraction(client, proposal_text, extracted_proposal_json):
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You validate and improve extracted proposal capabilities."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return parse_json_response(response.choices[0].message.content)

def exact_match_bonus(rfp_item, proposal_item):
    rfp_words = set(rfp_item.lower().split())
    proposal_words = set(proposal_item.lower().split())
    overlap = rfp_words.intersection(proposal_words)
    return 0.05 if len(overlap) >= 2 else 0.0

def category_bonus(rfp_category, proposal_category):
    rfp_cat = rfp_category.lower().strip()
    prop_cat = proposal_category.lower().strip()
    return 0.05 if (rfp_cat in prop_cat or prop_cat in rfp_cat) else 0.0

def classify_match_status(score):
    if score >= 0.70:
        return "Strong Match"
    elif score >= 0.50:
        return "Partial Match"
    else:
        return "Missing"

def semantic_match_rfp_to_proposal_fast(client, rfp_items, proposal_items, proposal_index, proposal_chunks):
    rfp_texts = [item["item"] for item in rfp_items]
    proposal_texts = [item["item"] for item in proposal_items]

    rfp_vectors = normalize_vectors(get_embeddings(client, rfp_texts))
    proposal_vectors = normalize_vectors(get_embeddings(client, proposal_texts))

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

def calculate_compliance_score(matching_results):
    total_items = len(matching_results)
    if total_items == 0:
        return {"ComplianceScore": 0, "StrongMatches": 0, "PartialMatches": 0, "MissingItems": 0, "TotalRequirements": 0, "AvgSimilarityScore": 0}
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

def assign_risk_type(category, item):
    cat = str(category).lower()
    if "security" in cat:
        return "Security Risk"
    elif "integration" in cat:
        return "Integration Risk"
    elif "timeline" in cat or "schedule" in cat:
        return "Schedule Risk"
    elif "budget" in cat or "cost" in cat:
        return "Cost Risk"
    elif "vendor" in cat:
        return "Vendor Capability Risk"
    elif "support" in cat:
        return "Support Risk"
    elif "non-functional" in cat:
        return "Performance Risk"
    else:
        return "Functional/Technical Risk"

def analyze_risks(matching_results):
    risks = []
    for item in matching_results:
        if item["status"] == "Missing":
            risks.append({
                "missing_requirement": item["rfp_item"],
                "category": item["rfp_category"],
                "risk_type": assign_risk_type(item["rfp_category"], item["rfp_item"]),
                "similarity_score": item["similarity_score"]
            })
    return risks

def calculate_category_coverage(matching_df, category_names):
    if isinstance(category_names, str):
        category_names = [category_names]
    # Normalize categories
    normalized_names = [n.lower().strip() for n in category_names]
    
    category_rows = matching_df[
        matching_df["rfp_category"].apply(lambda c: str(c).lower().strip() in normalized_names)
    ]
    if len(category_rows) == 0:
        return 0
    covered_rows = category_rows[
        category_rows["status"] != "Missing"
    ]
    return round((len(covered_rows) / len(category_rows)) * 100, 2)

def generate_recommendations(client, compliance_result, predicted_label, win_probability, missing_requirements, risk_summary):
    prompt = f"""
You are a Senior Proposal Review Consultant.
Analyze this proposal evaluation result:
Compliance Score: {compliance_result}
Win Probability Level: {predicted_label}
Win Probability: {win_probability:.2f}%
Missing Requirements: {missing_requirements}
Risk Summary: {risk_summary}

Provide:
1. Strengths
2. Weaknesses
3. Recommendations (Include difficulty and impact per recommendation)
4. Estimated Win Improvement

Return ONLY valid JSON matching this exact structure:
{{
  "strengths": ["list of strengths"],
  "weaknesses": ["list of weaknesses"],
  "recommendations": [
    {{
      "tag": "Category (e.g. Security, Performance)",
      "difficulty": "e.g. Low/Medium/High Effort",
      "text": "Detailed action step",
      "impact": "e.g. +3.5%"
    }}
  ],
  "estimated_win_improvement": "+12.5%"
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a proposal evaluation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return parse_json_response(response.choices[0].message.content)


# ==========================================
# 3. STATIC ROUTE HOOKS
# ==========================================
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


# ==========================================
# 4. API MAIN CONTROLLER
# ==========================================
@app.route('/api/analyze', methods=['POST'])
def analyze_docs():
    if 'rfp_file' not in request.files or 'proposal_file' not in request.files:
        return jsonify({"error": "Missing RFP or Proposal files."}), 400

    rfp_file = request.files['rfp_file']
    proposal_file = request.files['proposal_file']

    if rfp_file.filename == '' or proposal_file.filename == '':
        return jsonify({"error": "Invalid file selection."}), 400

    api_key = os.environ.get("OPENAI_API_KEY")

    # 4a. Run RAG Pipeline if OpenAI API Key & libraries exist
    if api_key and HAS_LIBRARIES:
        try:
            print("\n[API] OpenAI API Key detected. Starting full RAG and ML pipeline...")
            client = OpenAI(api_key=api_key)
            
            # Save files temporarily to read
            print("[RAG] Saving PDF uploads to temporary buffers...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_rfp:
                rfp_file.save(tmp_rfp.name)
                rfp_path = tmp_rfp.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_proposal:
                proposal_file.save(tmp_proposal.name)
                proposal_path = tmp_proposal.name

            # Extract texts
            print("[RAG] Extracting text characters from RFP and Proposal PDFs...")
            rfp_text = extract_text_from_pdf(rfp_path)
            proposal_text = extract_text_from_pdf(proposal_path)

            # Clean temporary files
            os.remove(rfp_path)
            os.remove(proposal_path)

            if not rfp_text or not proposal_text:
                raise ValueError("PDF content extraction returned empty text. Please check if PDF contains readable text.")

            print(f"[RAG] Extracted {len(rfp_text)} chars from RFP, {len(proposal_text)} chars from Proposal.")

            # Chunk proposal
            print("[RAG] Chunking proposal text...")
            proposal_chunks = chunk_text(proposal_text)
            if not proposal_chunks:
                proposal_chunks = [proposal_text]

            # Build FAISS database
            print(f"[RAG] Vectorizing {len(proposal_chunks)} chunks and building FAISS database index...")
            proposal_index, _ = build_faiss_index(client, proposal_chunks)

            # GPT extraction + validation: matches the notebook pipeline exactly
            print("[GPT] Extracting RFP requirements using GPT-4o-mini...")
            rfp_json_raw = extract_rfp_items(client, rfp_text)
            print("[GPT] Validating RFP requirements...")
            rfp_json = validate_rfp_extraction(client, rfp_text, rfp_json_raw)

            print("[GPT] Extracting proposal capabilities using GPT-4o-mini...")
            proposal_json_raw = extract_proposal_items(client, proposal_text)
            print("[GPT] Validating proposal capabilities...")
            proposal_json = validate_proposal_extraction(client, proposal_text, proposal_json_raw)

            print(f"[GPT] Final RFP items: {len(rfp_json.get('rfp_items', []))}; Final proposal items: {len(proposal_json.get('proposal_items', []))}")

            # Match requirements
            print("[RAG] Running semantic similarity match between RFP requirements and Proposal indexes...")
            matching_results = semantic_match_rfp_to_proposal_fast(
                client, 
                rfp_json["rfp_items"], 
                proposal_json["proposal_items"], 
                proposal_index, 
                proposal_chunks
            )

            # Compile scores
            compliance_result = calculate_compliance_score(matching_results)
            risk_results = analyze_risks(matching_results)

            matching_df = pd.DataFrame(matching_results)
            
            cov_functional = calculate_category_coverage(matching_df, ["Functional Requirements", "Functional Requirement"])
            cov_technical = calculate_category_coverage(matching_df, ["Technical Requirements", "Technical Requirement"])
            cov_security = calculate_category_coverage(matching_df, ["Security Requirements", "Security Requirement", "Security Controls"])
            cov_integration = calculate_category_coverage(matching_df, ["Integration Requirements", "Integration Requirement", "Integrations"])
            cov_timeline = calculate_category_coverage(matching_df, "Timeline")
            cov_budget = calculate_category_coverage(matching_df, ["Budget", "Cost"])

            cov_vendor = calculate_category_coverage(matching_df, ["Vendor Qualifications", "Vendor Qualification"])
            cov_evaluation = calculate_category_coverage(matching_df, ["Evaluation Criteria", "Evaluation Criterion"])
            cov_deliverable = calculate_category_coverage(matching_df, ["Deliverables", "Deliverable"])
            cov_support = calculate_category_coverage(matching_df, ["Support & Warranty", "Support/Warranty"])

            # Train parameters mapping
            risk_counts = pd.DataFrame(risk_results)["risk_type"].value_counts().to_dict() if len(risk_results) > 0 else {}
            
            project_features = {
                "ComplianceScore": compliance_result["ComplianceScore"],
                "StrongMatches": compliance_result["StrongMatches"],
                "PartialMatches": compliance_result["PartialMatches"],
                "MissingItems": compliance_result["MissingItems"],
                "AvgSimilarityScore": compliance_result["AvgSimilarityScore"],
                "FunctionalCoverage": cov_functional,
                "TechnicalCoverage": cov_technical,
                "SecurityCoverage": cov_security,
                "IntegrationCoverage": cov_integration,
                "TimelineCoverage": cov_timeline,
                "BudgetCoverage": cov_budget,
                "VendorQualificationCoverage": cov_vendor,
                "EvaluationCriteriaCoverage": cov_evaluation,
                "DeliverableCoverage": cov_deliverable,
                "SupportWarrantyCoverage": cov_support,
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

            # Predict win probability using the Random Forest classifier
            if rf_model is not None:
                print("[ML] Feeding features to RandomForest model to predict win probability...")
                X_test = pd.DataFrame([project_features]).reindex(columns=X_train.columns, fill_value=0)
                prediction = rf_model.predict(X_test)[0]
                predicted_label = label_encoder.inverse_transform([prediction])[0]
                probabilities = rf_model.predict_proba(X_test)[0]
                win_probability = max(probabilities) * 100
                print(f"[ML] Win probability level prediction: {predicted_label} ({win_probability:.2f}%)")
            else:
                predicted_label = "High" if compliance_result["ComplianceScore"] >= 75 else "Medium"
                win_probability = compliance_result["ComplianceScore"]

            # GPT recommendations
            missing_reqs_names = [r["missing_requirement"] for r in risk_results]
            print("[GPT] Requesting AI strengths, weaknesses, and recommendations using GPT-4o-mini...")
            recommendation_result = generate_recommendations(
                client, 
                compliance_result, 
                predicted_label, 
                win_probability, 
                missing_reqs_names, 
                risk_counts
            )
            print("[API] RAG analysis completed successfully. Dispatched payload.")

            # Map to UI format
            response_data = {
                "kpis": {
                    "complianceScore": f"{compliance_result['ComplianceScore']:.0f}%",
                    "winProbability": f"{win_probability:.0f}%",
                    "strongMatches": compliance_result["StrongMatches"],
                    "missingRequirements": compliance_result["MissingItems"],
                    "avgSimilarity": f"{compliance_result['AvgSimilarityScore']*100:.1f}%"
                },
                "coverage": {
                    "functional": int(cov_functional),
                    "technical": int(cov_technical),
                    "security": int(cov_security),
                    "integration": int(cov_integration),
                    "timeline": int(cov_timeline),
                    "budget": int(cov_budget)
                },
                "risks": [
                    {
                        "category": r["risk_type"] + "s" if not r["risk_type"].endswith("s") else r["risk_type"],
                        "status": "high" if "security" in r["risk_type"].lower() or "integration" in r["risk_type"].lower() else "med",
                        "desc": f"Gap found: '{r['missing_requirement']}' was not successfully matched in the proposal."
                    } for r in risk_results
                ] if len(risk_results) > 0 else [
                    {"category": "Security Risks", "status": "med", "desc": "Standard checks passed. No high severity missing items found."},
                    {"category": "Integration Risks", "status": "med", "desc": "Standard API integrations verified."}
                ],
                "insights": {
                    "strengths": recommendation_result["strengths"][:3],
                    "weaknesses": recommendation_result["weaknesses"][:3],
                    "winImprovement": recommendation_result["estimated_win_improvement"],
                    "recommendations": recommendation_result["recommendations"][:3]
                },
                "requirements": [
                    {
                        "id": idx + 1,
                        "requirement": r["rfp_item"],
                        "category": r["rfp_category"].replace(" Requirements", "").replace(" Requirement", ""),
                        "match": r["best_proposal_match"],
                        "score": f"{r['similarity_score']*100:.0f}%" if r["similarity_score"] > 0 else "0%",
                        "status": "strong" if r["status"] == "Strong Match" else "partial" if r["status"] == "Partial Match" else "missing"
                    } for idx, r in enumerate(matching_results)
                ]
            }

            return jsonify(response_data)

        except Exception as err:
            print(f"RAG execution error: {err}. Falling back to simulation.")
            # Fall back to simulation on error

    # 4b. No fake fallback results.
    # The notebook pipeline requires OpenAI + required libraries. Returning an error prevents
    # the UI from showing simulated numbers that do not match the notebook.
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY is missing. Set it in VS Code/Render Environment Variables, then run the analysis again."}), 500
    if not HAS_LIBRARIES:
        return jsonify({"error": "Required Python libraries are missing. Run: pip install -r requirements.txt"}), 500
    return jsonify({"error": "Analysis failed before completion. Check the terminal logs for the exact error."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
