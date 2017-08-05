# Statement of files

some files are just used to test or abondened,those files will not be explained.

## figure
those files must be run after the file split.py has been run
- link_velocity_figure.py: draw the velocity figures with different road id and time. the figures are named as XXX_X.png. for example 100_0.png means that the road id is 100 and all the velocity data on monday will be drew.
- weather_figure.py: draw the relationship between different weathers.
- weather_velocity_figure.py: draw the weather's influence on velocity

## code
- split.py: processe datas
- timeSeries.py: use RAIMA model to predict
- task1.py: the main code,use xgboost to build model and make the feature engineering to deal with the time series problem

## using
just run the task1.py
