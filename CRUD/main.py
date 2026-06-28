from fastapi import FastAPI,HTTPException
app = FastAPI()
items = []
@app.get("/")
def root():
    return {"message":"Hello World"}

@app.post("/items")
def create_item(item:str):
    items.append(item)
    return items
@app.get("/page/{limit}")
def get_limit_item(limit:int):
    return items[0:limit]

@app.get("/get/{id}")
def get_item(id:int):
    if id < len(items):
        return items[id]
    else:
        raise HTTPException(status_code = 404,detail="Item not found")