# Disaster Response Pipeline Project


### Summary of the project:
I analyzed disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

I'll get a data set containing real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.

I created a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.



### Explanation of the files in the repository:
- process_data.py
    - Loads the messages and categories datasets
    - Cleans the data
    - Stores it in a SQLite database
- train_classifier.py
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file


### How to run the Python scripts and web app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.  
    `python run.py`
    
   -> Your web app should now be running if there were no errors.

3. Open another Terminal Window and type.  
   `env|grep WORK`
   
   -> You'll get SPACEID and SPACEDOMAIN.
   
4. Open this link.  
   https://SPACEID-3001.SPACEDOMAIN
