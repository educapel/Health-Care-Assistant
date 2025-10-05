FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY Data_csvs/data_v1.csv Data_csvs/data_v1.csv
COPY health-care-assistant .

# Expose app port
EXPOSE 8000

# Start the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--timeout", "180"]