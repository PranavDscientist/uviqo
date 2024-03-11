from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
from keras.models import load_model
import cv2
from skimage import io
import pandas as pd
import asyncio
import os
import signal
from typing import Optional
import uvicorn

app = FastAPI()

# Load the package type detection model and categories
model_package = load_model('./models/newurl_model.h5')
categories = ['bottle', 'glass', 'plastic', 'unknown']  

# Load the weight prediction model and transformers
model_weight = joblib.load('./models/tree_model (2).joblib')
numerical_transformer = joblib.load('./transformers/numerical_transformer (2).joblib')
categorical_transformer = joblib.load('./transformers/categorical_transformer (2).joblib')

class Product(BaseModel):
    image_url: str
    net_weight: float
    unit: str
    id: str
    package_type: Optional[str] = None

class PredictionResponse(BaseModel):
    image_url: str
    id: str
    package_type: Optional[str] = None
    predicted_weight: Optional[float] = None

class ProductRequest(BaseModel):
    products: list[Product]

# Asynchronous processing
async def process_image_async(img_url, net_weight, unit, product_id):
    try:
        image = io.imread(img_url)
        image_resized = cv2.resize(image, (224, 224))
        image = image_resized.reshape(1, 224, 224, 3)
        pred = model_package.predict(image)
        ind = pred.argmax()
        package_type = categories[ind]

        if package_type == 'unknown':
            # Handle unknown category
            predicted_weight = net_weight + 5  
            return PredictionResponse(
                image_url=img_url,
                id=product_id,
                package_type='unknown',
                predicted_weight=predicted_weight
            )

        if package_type:
            if unit == 'kg':
                net_weight *= 1000  

            new_data = pd.DataFrame([[package_type, net_weight, unit]], columns=['package_type', 'net_weight', 'unit'])
            new_numerical_data = numerical_transformer.transform(new_data[['net_weight']])
            new_categorical_data = categorical_transformer.transform(new_data[['package_type', 'unit']])

            new_data_transformed = pd.concat([
                pd.DataFrame(new_numerical_data, columns=numerical_transformer.get_feature_names_out()),
                pd.DataFrame(new_categorical_data, columns=categorical_transformer.get_feature_names_out())
            ], axis=1)

            predicted_weight = model_weight.predict(new_data_transformed)

            return PredictionResponse(
                image_url=img_url,
                id=product_id,
                package_type=package_type,
                predicted_weight=predicted_weight.item()
            )

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.post('/predict_package_and_weight', response_model=dict)
async def predict_package_and_weight(request: ProductRequest):
    try:
        products = request.products

        if not products:
            raise HTTPException(status_code=400, detail='No product information found in the request')

        predictions = []
        tasks = []

        for product in products:
            img_url = product.image_url
            net_weight = product.net_weight
            unit = product.unit
            product_id = product.id

            if img_url and net_weight is not None and unit and product_id:
                # Perform image processing asynchronously
                tasks.append(process_image_async(img_url, net_weight, unit, product_id))

        # Wait for all asynchronous tasks to complete
        predictions = await asyncio.gather(*tasks)
        predictions = [p.dict() for p in predictions if p is not None]

        return {'predictions': predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/exit_server')
async def exit_server():
    os.kill(os.getpid(), signal.SIGINT)
    return JSONResponse(content={"message": "Shutting down"}, status_code=200)

if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=r"./certificates/_.yrush.in_private_key (1).key",
        ssl_certfile=r"./certificates/yrush.in_ssl_certificate (1).cer", 
        ssl_ca_certs=r"./certificates/_.yrush.in_ssl_certificate_INTERMEDIATE (1).cer"
    )
