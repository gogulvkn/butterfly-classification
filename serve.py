import io, json, numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

app = FastAPI(title="AI Image Classifier")
interp = Interpreter(model_path="model.tflite")
interp.allocate_tensors()
IN, OUT = interp.get_input_details()[0], interp.get_output_details()[0]
CLASSES = json.load(open("labels.json"))
THRESHOLD = 0.4

PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>AI Image Classifier</title>
<body style="font-family:sans-serif;max-width:480px;margin:30px auto;padding:0 16px;text-align:center">
<h2>AI Image Classifier</h2><p>Upload a photo of one of the trained classes. The model replies in a second.</p>
<input type=file id=f accept="image/*"><br><br>
<img id=p style="max-width:100%;border-radius:12px;display:none"><h3 id=r></h3>
<script>
f.onchange=async()=>{const file=f.files[0];if(!file)return;
p.src=URL.createObjectURL(file);p.style.display='block';r.textContent='Thinking... (first run can take ~1 min)';
const fd=new FormData();fd.append('file',file);
try{const j=await (await fetch('/predict',{method:'POST',body:fd})).json();
r.textContent=j.label+' ('+(j.confidence*100).toFixed(1)+'%)';}catch(e){r.textContent='Error, try again';}};
</script></body>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((160, 160))
    x = np.expand_dims(np.array(img, dtype="float32"), 0)
    interp.set_tensor(IN["index"], x)
    interp.invoke()
    probs = interp.get_tensor(OUT["index"])[0]
    i = int(probs.argmax())
    label = CLASSES[i] if probs[i] >= THRESHOLD else "unknown"
    return {"label": label, "confidence": float(probs[i]),
            "scores": dict(zip(CLASSES, map(float, probs)))}
