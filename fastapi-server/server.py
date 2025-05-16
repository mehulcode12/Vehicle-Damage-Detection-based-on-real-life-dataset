from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model_helper import predict
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "Hello World"}

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        # Read uploaded image bytes
        image_bytes = await file.read()

        # Open image from bytes in memory
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Predict using in-memory image
        prediction = predict(image)

        return {"prediction": prediction}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
