# Ames Housing Price Prediction

In this project I built a regression model to predict house prices in Ames, Iowa. The idea is that a real estate agent or someone selling their house could use it to get a price estimate.

My final model is XGBoost. On the test set it has an RMSE of about $18,800 and an R² of 0.93.

## Dataset

I used the Ames Housing dataset prepared by Dean De Cock. It has 2,930 house sales from 2006 to 2010 and 82 columns. The target is `SalePrice`.

Dataset documentation: https://jse.amstat.org/v19n3/decock/DataDocumentation.txt

This was the first time I worked with a dataset with this many features. In class we usually had small, clean datasets, so a big part of this project was deciding what to do with each column.

## Data cleaning

Some things I learned and decided:

- In many columns NA does not mean missing. For example in `Pool QC` it means the house has no pool. I checked the documentation before deciding what to do with each column.
- I dropped `Pool QC` and `Pool Area` because only 13 houses have a pool.
- I dropped `Misc Feature` and `Misc Val`. Most houses don't have one, and the price difference was small ($150k vs $161k median).
- I dropped `Order` and `PID` because they are just IDs.
- `Year Built` and `Garage Yr Blt` had 0.83 correlation, so I kept only `Year Built`. It has a higher correlation with price and no missing values.
- `Garage Cars` and `Garage Area` give almost the same information, so I kept `Garage Cars`.
- `Total Bsmt SF` and `1st Flr SF` also have high correlation (0.80), but I kept both because some houses don't have a basement.
- For `Alley` and `Fence` I filled NA with "None", so it becomes its own category after one-hot encoding.

For encoding, I converted 18 ordinal columns (like `Kitchen Qual`: Ex, Gd, TA, Fa, Po) to numbers so the order is kept. NA in these columns means the feature doesn't exist, so I gave it 0. The other categorical columns are one-hot encoded.

The dataset documentation suggests removing houses over 4,000 sq ft. Three of them were partial sales with very low prices for their size. I removed all 5, so the model is meant for houses up to 4,000 sq ft.

## Some findings

- `Overall Qual` has the strongest relationship with price (0.80 correlation).
- `Foundation_CBlock` and `Garage Type_Detchd` have negative correlation with price. When I checked, these are mostly older houses, and older houses are cheaper. So the age explains most of this, not the foundation or garage type itself.

## Models

| Step | RMSE |
|---|---|
| Baseline (always predict the mean) | ~$80,800 |
| Linear, Ridge, Lasso, Random Forest | ~$25,000 |
| After removing outliers (Ridge, CV) | $24,222 |
| After log transform (XGBoost, CV) | $19,727 |
| After grid search (XGBoost, CV) | $19,236 |
| **Final model on test set** | **$18,802 (R² = 0.934)** |

What I noticed:

- At first all four models gave around $25k. Changing the model didn't help much, so I focused on the data instead.
- Removing outliers helped the linear models a lot (about 20%), but Random Forest didn't change much.
- When I used 5-fold cross validation instead of one train/test split, the ranking of the models changed. After that I compared models only with CV and used the test set once at the end.
- Taking the log of the price helped Ridge the most (24,222 → 19,899). With log, the model looks at percentage error, so expensive houses don't dominate.
- After the log transform XGBoost was slightly better than Ridge and more stable across folds, so I picked XGBoost and tuned it with grid search.

## Limitations

- The data is only from Ames, 2006-2010, so it doesn't work for other cities or today's prices.
- The model isn't reliable for houses bigger than 4,000 sq ft.
- The final score is from one test split.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn
