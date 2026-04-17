# Minimal Shiny App

This is a minimal Shiny app that displays an interactive histogram of random numbers. Users can adjust the number of bins using a slider.

## Running Locally

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
shiny run app.py
```

The app will be available at http://localhost:8000

## Deploying the App

### Option 1: Deploy to Shinyapps.io (Recommended for beginners)

1. Install the `rsconnect` package:
```bash
pip install rsconnect-python
```

2. Sign up for a free account at [shinyapps.io](https://www.shinyapps.io)

3. Configure your account:
```bash
rsconnect add-account shinyapps.io
```

4. Deploy the app:
```bash
rsconnect deploy shiny .
```

### Option 2: Deploy to a Cloud Platform

#### Heroku
1. Create a `Procfile`:
```
web: shiny run app.py --host 0.0.0.0 --port $PORT
```

2. Create a `runtime.txt`:
```
python-3.9.16
```

3. Deploy using Heroku CLI:
```bash
heroku create your-app-name
git add .
git commit -m "Initial commit"
git push heroku main
```

#### Google Cloud Run
1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
```

2. Build and deploy:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/shiny-app
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/shiny-app --platform managed
```

## Notes
- The app uses port 8000 by default when running locally
- For production deployment, make sure to set appropriate environment variables and security measures
- Consider adding error handling and logging for production use 