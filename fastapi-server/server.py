from fastapi import FastAPI, File, UploadFile


app = FastAPI()
# @app.get(["/home", "/"])
# async def root():
#     return {"message": "Hello World"}

@app.get("/hello")
async def hello():
    return {"message": "Hello World"}


from model_helper import predict
from fastapi.responses import HTMLResponse

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):

    try:

        image_bytes = await file.read()
        image_path = "temp_file.jpg" 
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        prediction = predict(image_path)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

