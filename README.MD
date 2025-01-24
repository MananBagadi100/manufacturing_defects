## Setup Instructions (Without Docker)

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd <repository-name>

    Create and Activate a Virtual Environment
        On macOS/Linux:

python3 -m venv ve
source ve/bin/activate

On Windows:

    python -m venv ve
    ve\Scripts\activate

Install Dependencies
Install the required dependencies from the requirements.txt file:

pip install -r requirements.txt

Run the Flask App Locally
Start the Flask application:

    python main.py

API Testing Instructions

Once the Flask app is running locally, use the following endpoints to test the API.
API Endpoints

    POST /upload: Upload CSV data.
    POST /train: Train the ML model and return performance metrics.
    POST /predict: Make predictions based on input JSON data.

Local Testing Examples

Use cURL commands to test the API (replace 127.0.0.1:5000 with your server's IP if necessary).

# Upload CSV Data
curl -X POST -F 'file=@data/sample.csv' http://127.0.0.1:5000/upload

# Train the Model
curl -X POST http://127.0.0.1:5000/train

# Make Predictions
curl -X POST -H "Content-Type: application/json" \
-d '{"Temperature": 80, "Run_Time": 120}' \
http://127.0.0.1:5000/predict