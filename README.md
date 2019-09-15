# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
   -> Your web app should now be running if there were no errors.

3. Open another Terminal Window and type.
   'env|grep WORK'
   
   -> You'll get SPACEID and SPACEDOMAIN.
   
4. Open this link.
   https://SPACEID-3001.SPACEDOMAIN
