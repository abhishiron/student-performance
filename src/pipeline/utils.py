import os
import sys
import pickle
import dill as pickle

from sklearn.metrics import r2_score
from src.pipeline.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Save object to file using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(models, X_train, y_train, X_test, y_test, params):
    """
    Evaluate multiple models with hyperparameter tuning and return their performance scores
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # Get parameters for this model
            model_params = params.get(model_name, {})
            
            if model_params:  # If parameters exist for tuning
                print(f"Performing GridSearchCV for {model_name}...")
                gs = GridSearchCV(
                    estimator=model, 
                    param_grid=model_params, 
                    cv=3, 
                    n_jobs=-1, 
                    verbose=1,
                    scoring='r2'
                )
                gs.fit(X_train, y_train)
                
                # Use the best model from GridSearch
                best_model = gs.best_estimator_
                models[model_name] = best_model  # Update the original models dict
                
                print(f"Best params for {model_name}: {gs.best_params_}")
                print(f"Best CV score for {model_name}: {gs.best_score_:.4f}")
                
            else:  # No parameters to tune, just fit normally
                print(f"No hyperparameters to tune for {model_name}, using default parameters...")
                model.fit(X_train, y_train)
                best_model = model
            
            # Make predictions with the best/fitted model
            y_test_pred = best_model.predict(X_test)
            
            # Calculate R2 score
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store in report
            report[model_name] = test_model_score
            
            print(f"{model_name} Test R2 Score: {test_model_score:.4f}")
            print("-" * 60)
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Load object from file using pickle
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)