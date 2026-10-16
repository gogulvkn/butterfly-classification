# butterfly-classification

Image classifier trained and deployed automatically by an AI pipeline
(Kaggle GPU -> GitHub -> Render).

## Model
- Classes: adonis, african_giant_swallowtail, american_snoot, an_88, appollo, atala, banded_orange_heliconian, banded_peacock, beckers_white, black_hairstreak, blue_morpho, blue_spotted_crow, brown_siproeta, cabbage_white, cairns_birdwing, checquered_skipper, chestnut, cleopatra, clodius_parnassian, clouded_sulphur, common_banded_awl, common_wood-nymph, copper_tail, crecent, crimson_patch, danaid_eggfly, eastern_coma, eastern_dapple_white, eastern_pine_elfin, elbowed_pierrot, gold_banded, great_eggfly, great_jay, green_celled_cattleheart, grey_hairstreak, indra_swallow, iphiclus_sister, julia, large_marble, malachite, mangrove_skipper, mestra, metalmark, milberts_tortoiseshell, monarch, mourning_cloak, orange_oakleaf, orange_tip, orchard_swallow, painted_lady, paper_kite, peacock, pine_white, pipevine_swallow, popinjay, purple_hairstreak, purplish_copper, question_mark, red_admiral, red_cracker, red_postman, red_spotted_purple, scarce_swallow, silver_spot_skipper, sleepy_orange, sootywing, southern_dogface, straited_queen, tropical_leafwing, two_barred_flasher, ulyses, viceroy, wood_satyr, yellow_swallow_tail, zebra_long_wing
- Base model: MobileNetV2 (transfer learning), exported to TFLite
- Test accuracy: **85.0%**
- Below 40% confidence the API answers `unknown`

## Use it
- Open the service URL on Render to get the upload page.
- API: `POST /predict` with an image file (form field `file`).
- Health check: `GET /health`.

```
curl -X POST -F "file=@photo.jpg" https://<your-service>.onrender.com/predict
```

## Files
- `serve.py`: FastAPI app and HTML page
- `model.tflite`, `labels.json`: the model and its class names
- `requirements.txt`: dependencies

Note: on the free Render plan the first request after idle can take about a minute.

Full guide: see [DETAILED_README.md](DETAILED_README.md).
