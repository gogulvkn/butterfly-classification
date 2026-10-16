# butterfly-classification: detailed guide

## 1. Overview
An image classifier trained and deployed by an automated pipeline. It loads a dataset,
preprocesses it, trains a model, evaluates it, and publishes it as a live web app.

- Classes: adonis, african_giant_swallowtail, american_snoot, an_88, appollo, atala, banded_orange_heliconian, banded_peacock, beckers_white, black_hairstreak, blue_morpho, blue_spotted_crow, brown_siproeta, cabbage_white, cairns_birdwing, checquered_skipper, chestnut, cleopatra, clodius_parnassian, clouded_sulphur, common_banded_awl, common_wood-nymph, copper_tail, crecent, crimson_patch, danaid_eggfly, eastern_coma, eastern_dapple_white, eastern_pine_elfin, elbowed_pierrot, gold_banded, great_eggfly, great_jay, green_celled_cattleheart, grey_hairstreak, indra_swallow, iphiclus_sister, julia, large_marble, malachite, mangrove_skipper, mestra, metalmark, milberts_tortoiseshell, monarch, mourning_cloak, orange_oakleaf, orange_tip, orchard_swallow, painted_lady, paper_kite, peacock, pine_white, pipevine_swallow, popinjay, purple_hairstreak, purplish_copper, question_mark, red_admiral, red_cracker, red_postman, red_spotted_purple, scarce_swallow, silver_spot_skipper, sleepy_orange, sootywing, southern_dogface, straited_queen, tropical_leafwing, two_barred_flasher, ulyses, viceroy, wood_satyr, yellow_swallow_tail, zebra_long_wing
- Stack: TensorFlow/Keras (training), TFLite (serving), FastAPI, GitHub, Render

## 2. How the pipeline works
1. **Load**: reads `dataset/<class_name>/*.jpg`; 80% train, 20% held out, and the held-out part is split into validation and test.
2. **Preprocess**: random flip, rotation and zoom on training images; pixel scaling is built into the model.
3. **Train**: MobileNetV2 pretrained on ImageNet (frozen) plus a small classifier head, 5 epochs.
4. **Evaluate**: accuracy and per-class F1 on test images the model never saw. If accuracy is below 0.90 the pipeline retrains with fine-tuning (top layers unfrozen, low learning rate).
5. **Export**: the Keras model is converted to TFLite so it fits in the free Render plan's 512 MB RAM.
6. **Deploy**: files are pushed to a public GitHub repo, then a free Render web service is created (or redeployed) and checked at `/health`.

## 3. Results
Held-out test accuracy: **85.0%**

| Class | F1 score |
|---|---|
| adonis | 0.923 |
| african_giant_swallowtail | 0.909 |
| american_snoot | 0.889 |
| an_88 | 1.000 |
| appollo | 1.000 |
| atala | 0.875 |
| banded_orange_heliconian | 0.933 |
| banded_peacock | 1.000 |
| beckers_white | 0.857 |
| black_hairstreak | 0.889 |
| blue_morpho | 0.857 |
| blue_spotted_crow | 0.857 |
| brown_siproeta | 0.833 |
| cabbage_white | 0.857 |
| cairns_birdwing | 1.000 |
| checquered_skipper | 0.960 |
| chestnut | 1.000 |
| cleopatra | 0.500 |
| clodius_parnassian | 0.917 |
| clouded_sulphur | 0.424 |
| common_banded_awl | 0.800 |
| common_wood-nymph | 0.960 |
| copper_tail | 0.640 |
| crecent | 0.778 |
| crimson_patch | 0.947 |
| danaid_eggfly | 0.750 |
| eastern_coma | 0.533 |
| eastern_dapple_white | 0.588 |
| eastern_pine_elfin | 0.923 |
| elbowed_pierrot | 0.947 |
| gold_banded | 0.857 |
| great_eggfly | 0.640 |
| great_jay | 0.857 |
| green_celled_cattleheart | 0.929 |
| grey_hairstreak | 0.933 |
| indra_swallow | 0.941 |
| iphiclus_sister | 0.941 |
| julia | 1.000 |
| large_marble | 0.815 |
| malachite | 0.933 |
| mangrove_skipper | 1.000 |
| mestra | 0.889 |
| metalmark | 0.909 |
| milberts_tortoiseshell | 0.941 |
| monarch | 0.667 |
| mourning_cloak | 0.800 |
| orange_oakleaf | 0.889 |
| orange_tip | 1.000 |
| orchard_swallow | 0.952 |
| painted_lady | 1.000 |
| paper_kite | 0.933 |
| peacock | 1.000 |
| pine_white | 0.941 |
| pipevine_swallow | 0.933 |
| popinjay | 0.941 |
| purple_hairstreak | 0.714 |
| purplish_copper | 0.778 |
| question_mark | 0.571 |
| red_admiral | 1.000 |
| red_cracker | 0.889 |
| red_postman | 0.923 |
| red_spotted_purple | 0.889 |
| scarce_swallow | 1.000 |
| silver_spot_skipper | 0.857 |
| sleepy_orange | 0.667 |
| sootywing | 0.714 |
| southern_dogface | 0.600 |
| straited_queen | 0.880 |
| tropical_leafwing | 0.600 |
| two_barred_flasher | 0.857 |
| ulyses | 0.667 |
| viceroy | 0.800 |
| wood_satyr | 0.769 |
| yellow_swallow_tail | 0.857 |
| zebra_long_wing | 1.000 |

## 4. Using the app
Open the Render URL of the service and upload a photo. The page shows the label and confidence.

### API
`POST /predict` with a form field named `file`:

```bash
curl -X POST -F "file=@photo.jpg" https://<your-service>.onrender.com/predict
```

Response:

```json
{"label": "cats", "confidence": 0.97, "scores": {"cats": 0.97, "dogs": 0.03}}
```

- If the top confidence is below 40%, `label` is `unknown`. This stops images of classes the model never saw (for example a horse) from being forced into a known class.
- `GET /health` returns `{"status": "ok", "classes": [...]}`.

## 5. Repository files
| File | Purpose |
|---|---|
| `serve.py` | FastAPI app, HTML upload page, `/predict` and `/health` |
| `model.tflite` | the trained model |
| `labels.json` | class names in model output order |
| `requirements.txt` | serving dependencies (TFLite runtime, FastAPI, Pillow) |
| `README.md` | short summary |
| `DETAILED_README.md` | this guide |

Render starts the app with `uvicorn serve:app --host 0.0.0.0 --port $PORT`, meaning "run `app` from `serve.py`".

## 6. Retrain or update
Run the training notebook again with the same service name. It overwrites the files in this repo and redeploys the same Render service, so the URL stays the same. Files are regenerated on every run, so manual edits to `serve.py` or the READMEs are replaced.

## 7. Limitations
- Free Render plan: the service sleeps after about 15 minutes idle; the first request can take up to a minute.
- The model only knows its training classes; other images should come back as `unknown`, but this is not guaranteed.
- Accuracy depends on the training data: blurry, cropped or unusual photos can be misclassified.
- The repository is public, so the model file is public too.

## 8. Troubleshooting
- **Render build fails**: open the service's Logs tab on dashboard.render.com and check the error.
- **Service not responding**: wait about a minute for the cold start, then retry.
- **Wrong labels**: check that `labels.json` matches your dataset's folder names.
