# Description
Custom Imputer class PanelImputer for panel data for use in a sklearn pipeline, e.g. to impute yearly data to monthly data. Class imputes on a location-by-location basis.

For better (not yet complete) documentation, see the docstring.

# Installation
Install via
```
pip install panel-imputer
```

# Example use:

```
from panel_imputer import PanelImputer

#1: use fit_transform for imputation with prepared dataframe
df = read_some_panel_data_with_missing_values()

imp = PanelImputer(
    location_index='location',
    time_index=['year', 'month'],
    imputation_method='bfill'
)

df_imputed = imp.fit_transform(df)

#2: use in a pipeline
pipe = Pipeline(
    [('impute', imp),
    ('model', RandomForestClassifier())]
)
X, y = df[features], df[target]

pipe.fit(X, y)
```    

For more examples, see the jupyter notebook.
