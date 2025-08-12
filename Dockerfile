# ClauseRadar

# lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml poetry.lock* /app/

# Install Poetry and project dependencies
RUN pip install --no-cache-dir poetry \
    && POETRY_VIRTUALENVS_CREATE=false POETRY_NO_INTERACTION=1 \
       poetry install --no-root --only main

# Copy the rest of the application code
COPY . /app

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Configure Streamlit to run headless on 0.0.0.0:8080
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ENABLECORS=false

# Launch the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
