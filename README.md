# Social Media-Based Travel Trend Prediction with Deepfake Content Filtering

An AI system that forecasts how popular Indian travel destinations will be in the coming months, and a detector that checks whether a travel photo is real or AI-generated. The two parts are being combined into one app.

**Status: under development.** The trend-forecasting module and the deepfake-detection experiments are done. The final detector test and the combined app are in progress. See [Project Status](#project-status).

> **Scope note:** the trend module currently uses **Google Trends search data only**. Social media post collection and sentiment scoring are not part of the current scope.

## Problem

Travel decisions now come largely from what people see on Instagram, YouTube and X. One viral reel can turn a quiet place into a tourist hotspot within weeks, so travellers and tourism boards look for the destinations that are gaining attention.

But not everything online is real. AI image tools make fake "photos" of places easy to produce, and a fake photo can make a place look better than it is. Travellers also need to know *when* a place is worth visiting, since interest and conditions change through the year.

## Proposed Solution

Two modules, joined in one app:

- **Forecast** destination interest from five years of search data, and recommend where to go in a given month (implemented).
- **Detect** whether a travel image is real or AI-generated (implemented as experiments, with known limits).
- **Combine** both in a single Streamlit app, so a user can get destination suggestions for a month and also check a travel photo (planned).

## System Workflow

```
Google Trends (5 years, weekly) --> Cleaning --> Seasonal index --> Forecast (Prophet + seasonal average) --> Month recommendations

Travel photo --> Standardise (512x512) --> Xception features --> Classifier --> Real / AI-generated

Both parts will be combined in one Streamlit app (planned)
```

## Data Sources

| Source | Used for | Status |
| --- | --- | --- |
| Google Trends (via pytrends) | Weekly search interest, last 5 years, India, 100 destinations | In use |
| CIFAKE (Kaggle) | Real vs AI-generated images to train the first detector | In use |
| Custom image set (158 travel images after removing duplicates: 74 fake, 84 real) | Training and testing the detector on realistic photos | In use |

## Modules

### 1. Travel Trend Forecasting (`notebooks/social_media_part.ipynb`)

- Downloads 5 years of weekly Google Trends interest for **100 Indian destinations** in three tiers (popular, mid, offbeat). It batches requests, retries when blocked, and caches results to avoid refetching.
- Drops destinations with too little data (1 dropped, **99 usable**).
- Builds a **seasonal index** per destination: how busy each month is compared with that destination's own average.
- Offers these functions:
  - **Suggest** well-known and offbeat places for a given month.
  - **Check** whether a given destination is a good time to visit in a given month, with its overall trend (Rising / Stable / Falling).
  - **Forecast** the next 12 months for a destination.
- Compares forecasting methods in a backtest: trained on the first 4 years, tested on the last 12 months, for all 99 destinations. Error is normalised mean absolute error (error divided by the destination's average interest; lower is better):

| Method | Mean error | Median error | Destinations won |
| --- | --- | --- | --- |
| **Prophet + seasonal average** (final model) | **0.179** | 0.128 | 14 |
| Prophet + seasonal naive | 0.184 | 0.123 | 14 |
| Seasonal naive | 0.187 | 0.123 | 21 |
| Seasonal average | 0.206 | 0.159 | 25 |
| Prophet alone | 0.207 | 0.154 | 10 |
| SARIMA | 0.232 | 0.168 | 15 |

- The gain of the final model over the simple seasonal baseline is small, because search interest is dominated by a yearly cycle. Prophet is kept because it also provides an interpretable trend.
- The final model forecasts Oct 2026 to Sep 2027. Of the 99 destinations, 8 are Rising, 71 Stable and 20 Falling.
- A simple text menu lets anyone use the recommender without touching the code.
- Exports `month_index_forecast.csv` and `forecast_next12.csv`.

### 2. Deepfake Image Detection (`notebooks/deepfake_part.ipynb`)

**Step 1: baseline trained on CIFAKE.** Xception pre-trained on ImageNet, with a pooling layer, dropout (0.3) and a sigmoid output. The last 20 layers are fine-tuned at a learning rate of 1e-5. Training is **staged** on growing amounts of data, with a checkpoint saved after each stage:

| Stage | Images | Train acc. | Val. acc. |
| --- | --- | --- | --- |
| 1 | 20,000 | 93.6% | 94.1% |
| 2 | 40,000 | 96.9% | 96.3% |
| 3 | 60,000 | 98.8% | 96.9% |
| 4 | 80,000 | 98.9% | 97.3% |
| 5 | 100,000 | 99.3% | 97.4% |

**CIFAKE test accuracy: 97.53%.**

**Step 2: finding that the model does not transfer to realistic photos.** On our own travel images, the CIFAKE model scored **46.9%** and labelled every image fake.

| Experiment | Result |
| --- | --- |
| CIFAKE model on custom images | 46.9% accuracy, every image predicted fake |
| Same, with images downscaled to 32x32 to mimic CIFAKE | 48.8% accuracy |
| ROC analysis of that version | AUC 0.44, so no usable signal |

**Why it fails:** CIFAKE images are tiny (32x32), and every fake comes from one generator. The model learned that generator's pixel traces, not whether a scene is realistic. Sharp, high-resolution travel photos have none of those traces. This is domain shift.

**Step 3: training on our own travel images.**

- Images are standardised to **512x512 JPEG** (centre-crop, similar compression for real and fake) and exact duplicates are removed, leaving 158 images (74 fake, 84 real). 20% (32 images) is kept as a locked test set; the other 126 are used for model selection with 5-fold cross-validation.
- Following the guidance of using a CNN with pretrained layers, we varied the **number of layers** (how deep into Xception we cut, or how many layers are trained) and the **global pooling** type (average, max, or both).

| Experiment (5-fold cross-validation, 126 images) | Accuracy |
| --- | --- |
| Xception fine-tuned end to end, last 0 / 20 / 40 / 60 layers trained | 59.5% / 61.9% / 62.8% / 63.5% |
| Xception features, final layer (block 14) with average / max / both pooling | 64.3% / 58.8% / 61.9% |
| **Xception features cut at block 4, average pooling** | **77.0% (±5.9), AUC 0.804** |
| Same, with an extra hidden layer of 64 units | 74.6% (±3.1), AUC 0.817 |
| Logistic regression on Xception features (separate experiment, 5x5 repeated cross-validation, 160 images) | 74.0%, AUC 0.793 |

**What we learned:** early layers (blocks 3 and 4) work better than the final layers (AUC 0.77 to 0.81 against 0.64 to 0.65). Fake-image cues are fine textures, which early layers capture, while later layers describe what is in the picture. Training the whole network on about 126 images overfits and performs worse than using frozen features.

The best setting was chosen from 18 combinations on the same folds, so the 77% figure is slightly optimistic. With so few images the result varies by about 6 points between folds. The locked test set has not yet been used.

## Technology Stack

| Area | Tools |
| --- | --- |
| Data collection | Python, pytrends |
| Data processing | pandas, NumPy, Pillow, OpenCV |
| Forecasting | Prophet, statsmodels (SARIMA) |
| Deepfake detection | TensorFlow / Keras, Xception, scikit-learn |
| Environment | Google Colab (GPU), Google Drive for caching |
| Planned | Streamlit, Plotly |

## Project Structure

```
.
├── notebooks/
│   ├── social_media_part.ipynb     # Trend data, seasonal index, forecasting, recommender
│   └── deepfake_part.ipynb         # Xception training, evaluation, domain-shift diagnosis, custom-image experiments
├── data/                           # Exported tables (optional)
├── docs/
│   └── Project_Synopsis.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Both notebooks were written for **Google Colab** and save to a Google Drive folder named `deepfake_project`.

1. Open a notebook in Colab (use a GPU runtime for the deepfake notebook).
2. Mount Google Drive first, then run the cells in order. The first cells install the libraries.
3. For the deepfake notebook, add your own Kaggle credentials as a Colab secret or environment variable. **Never hard-code them in the notebook.** The CIFAKE dataset is downloaded from Kaggle (`birdy654/cifake-real-and-ai-generated-synthetic-images`).
4. The custom image set is not included in this repository. Place your own real and fake images in `custom_data/real` and `custom_data/fake` on Drive; the notebook standardises them to 512x512.

For local use: `pip install -r requirements.txt`.

## Project Status

### Done

- [x] Google Trends collection for 100 destinations, with caching and retry handling
- [x] Seasonal index and month-based recommender (suggest, check, forecast)
- [x] 12-month forecasting, with a backtest comparing six methods
- [x] Text menu for using the recommender
- [x] CSV export of seasonal and forecast tables
- [x] Xception transfer-learning pipeline with staged training (97.53% on CIFAKE)
- [x] Domain-shift diagnosis on custom travel images
- [x] Custom image set standardised and de-duplicated, with depth and pooling experiments (best: 77% cross-validated)

### In progress / planned

- [ ] Compare 299x299 and full-resolution 512x512 input for the feature extractor
- [ ] Final test of the chosen detector on the locked test set, and save the final model
- [ ] Combine the recommender and the image check in one Streamlit app, with Plotly charts

## Current Limitations

- Trend forecasting uses **Google search interest only**. It is a relative scale, not a count of visitors, and does not measure weather or road conditions.
- Only 5 years of data are available, so seasonal patterns are estimated from few cycles, and the forecast backtest covers a single test year.
- The deepfake detector trained on CIFAKE **does not work on realistic photos**. The improved version is trained and tested on only 158 images, so its accuracy is an early indication, not a reliable figure.
- The detector and the trend module are not yet connected in one app.

## Disclaimer

This is a student prototype and research project. Its outputs are not verified travel advice and should not be used for operational decisions.
