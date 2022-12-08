# aibedoviz
VA system for the AiBEDO project

Instructions:
- run `python aibedoviz.py`
- open the url specified in your web-browser (prefered: chrome)

Data Pre-req:
- make sure to have the full Aibedo model in `fullmodel` dir
- maintain spherical unet model definitions in `spherical_unet` dir
- install all package dependencies

VA Features:
- load different timesteps of input dataset and use different lagged models to make predictions
- Perform MCB experiments
    - specify MCB duration
    - specify MCB sites
    - specify MCB variables
    - access tipping points risk from resulting MCB
    - record interesting MCB settings


Environment:
- conda create --name <env> --file requirements.txt
