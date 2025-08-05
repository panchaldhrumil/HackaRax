# Insurance Policy Q&A System

## Overview

This FastAPI application provides a question-answering system for insurance policy documents. It uses:
- Gemini AI (Google's LLM) for natural language understanding
- Pinecone for vector search and document retrieval
- Sentence Transformers for generating embeddings
- Support for PDF, DOCX, and email documents

## Features

- Upload and index insurance policy documents
- Ask natural language questions about policy terms
- Get accurate answers with references to specific policy clauses
- Supports multiple document formats (PDF, DOCX, MSG/EML)

## Prerequisites

- Python 3.8+
- Pinecone API key
- Gemini API key
- (Optional) Docker for containerized deployment

## Installation

1. Clone the repository:
   git clone https://github.com/your-repo/insurance-qa-system.git
   cd insurance-qa-system

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create a .env file with your API keys:
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   GEMINI_API_KEY=your_gemini_api_key

## Usage

### Running the Application

Start the FastAPI server:
uvicorn main:app --reload

The API will be available at http://localhost:8000

### API Endpoint

POST /hackrx/run

Headers:
Authorization: Bearer 35928de76852eb7aacd2ad7b581bee5c8ab7539bdb514be752b6479293dccb2b

Request Body:
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

{
"answers": [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    ]
}

## Deployment

### Docker

1. Build the Docker image:
   docker build -t insurance-qa .

2. Run the container:
   docker run -p 8000:8000 --env-file .env insurance-qa

### Production Deployment

For production, consider using:
- Gunicorn as the application server
- Nginx as a reverse proxy
- Supervisor for process management

Example Gunicorn command:
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

## Configuration

Environment Variable | Description
----------------------|-------------
PINECONE_API_KEY | Your Pinecone API key
PINECONE_ENVIRONMENT | Pinecone environment (e.g., "gcp-starter")
GEMINI_API_KEY | Google Gemini API key
PINECONE_INDEX_NAME | (Optional) Pinecone index name (default: "insurance-policy-index")
LLM_MODEL | (Optional) Gemini model version (default: "gemini-1.5-flash-latest")

## Troubleshooting

1. Gemini API Errors:
   - Verify your API key is correct
   - Check your Google Cloud billing status
   - Ensure the model name is valid

2. Document Parsing Issues:
   - Verify the document URL is accessible
   - Check the document format is supported (PDF/DOCX/email)

3. Pinecone Connection Problems:
   - Confirm your API key and environment are correct
   - Check the index exists in your Pinecone project

4. General Issues:
   - Check application logs for detailed error messages
   - Ensure all dependencies are properly installed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

fastapi>=0.68.0,<0.69.0
uvicorn>=0.15.0,<0.16.0
python-dotenv>=0.19.0,<0.20.0
requests>=2.26.0,<3.0.0
pinecone-client>=2.2.0,<3.0.0
google-generativeai>=0.3.0,<0.4.0
sentence-transformers>=2.2.0,<3.0.0
pymupdf>=1.18.0,<2.0.0
python-docx>=0.8.10,<0.9.0
email-validator>=1.1.3,<2.0.0



for to check that our api key is valid or expired?
in one separate file,
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-1.5-flash-latest")
response = model.generate_content("Say hello")
print(response.text)
