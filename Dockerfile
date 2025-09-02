FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the files
COPY requirements.txt .
COPY app/ ./app/
COPY diabetes.csv .
COPY train.py .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# train the model
RUN python train.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FLASK application
CMD ["python", "app/app.py"]