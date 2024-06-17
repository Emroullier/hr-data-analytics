from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'HR data analytics project': 'This is the first app of our project !'}
