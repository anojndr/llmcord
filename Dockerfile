FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

# Install google-cloud-aiplatform for Vertex AI if needed, but google-genai is the primary library for Gemini API
# google-generativeai is deprecated, ensure google-genai is used.
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "llmcord.py"]