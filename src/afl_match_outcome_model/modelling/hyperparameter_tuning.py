import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss, mean_tweedie_deviance
from sklearn.model_selection import train_test_split

from .optuna_xgb_param_grid import OptunaXGBParamGrid


class HyperparameterTuner:
    def __init__(self, training_data, response):
        """Model agnostic hyperparameter tuner that requires training data and response.

        Args:
            training_data (Dataframe): Training data with modelling features
            response (Array): Training data response
        """
        self.training_data = training_data
        self.response = response


class XGBHyperparameterTuner(HyperparameterTuner):
    def __init__(
        self, training_data, response, optuna_param_grid, monotonicity_constraints=None, num_class=None
    ) -> None:
        super().__init__(training_data, response)
        self.monotonicity_constraints = monotonicity_constraints
        self.num_class = num_class
        self.optuna_param_grid = optuna_param_grid

    def objective(self, trial):
        train_x, valid_x, train_y, valid_y = train_test_split(
            self.training_data, self.response, test_size=self.optuna_param_grid.validation_size
        )
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        param = {
            "verbosity": self.optuna_param_grid.verbosity,
            "objective": self.optuna_param_grid.error,
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int(
                "max_depth",
                self.optuna_param_grid.max_depth_min,
                self.optuna_param_grid.max_depth_max,
                step=self.optuna_param_grid.max_depth_step,
            ),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                self.optuna_param_grid.min_child_weight_min,
                self.optuna_param_grid.min_child_weight_max,
                step=self.optuna_param_grid.min_child_weight_step,
            ),
            "eta": trial.suggest_float("eta", self.optuna_param_grid.eta_min, self.optuna_param_grid.eta_max, log=True),
            # defines how selective algorithm is.
            "gamma": trial.suggest_float(
                "gamma", self.optuna_param_grid.gamma_min, self.optuna_param_grid.gamma_max, log=True
            ),
            # L2 regularization weight.
            "lambda": trial.suggest_float(
                "lambda", self.optuna_param_grid.lambda_min, self.optuna_param_grid.lambda_max, log=True
            ),
            # L1 regularization weight.
            "alpha": trial.suggest_float(
                "alpha", self.optuna_param_grid.alpha_min, self.optuna_param_grid.alpha_max, log=True
            ),
            # sampling ratio for training data.
            "subsample": trial.suggest_float(
                "subsample", self.optuna_param_grid.subsample_min, self.optuna_param_grid.subsample_max
            ),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", self.optuna_param_grid.colsample_bytree_min, self.optuna_param_grid.colsample_bytree_max
            ),
        }
        param["monotone_constraints"] = self.monotonicity_constraints
        param["num_class"] = self.num_class

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)

        if self.error == "reg:squarederror":
            return mean_squared_error(preds, valid_y, squared=False)
        if self.error == "binary:logistic":
            return log_loss(valid_y, preds)
        if self.error == "multi:softprob":
            return log_loss(valid_y, preds)
        if self.error == "reg:tweedie":
            return mean_tweedie_deviance(valid_y, preds, power=self.tweedie_power)

    def get_objective_function(self):
        return self.objective

    def tune_hyperparameters(self):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=self.trials)

        print("Number of finished trials: ", len(self.study.trials))
        print("Best trial:")
        trial = self.study.best_trial

        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return self.study

    def get_best_params(self):
        return self.study.best_params


class XGBCVHyperparameterTuner(XGBHyperparameterTuner, OptunaXGBParamGrid):
    def __init__(
        self,
        training_data,
        response,
        optuna_param_grid,
        monotonicity_constraints=None,
        nfolds=5,
        num_class=None,
    ) -> None:
        super().__init__(training_data, response, optuna_param_grid)
        self.monotonicity_constraints = monotonicity_constraints
        self.nfolds = nfolds
        self.num_class = num_class

    def objective(self, trial):
        dtrain = xgb.DMatrix(self.training_data, label=self.response)

        param = {
            "verbosity": self.optuna_param_grid.verbosity,
            "objective": self.optuna_param_grid.error,
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int(
                "max_depth",
                self.optuna_param_grid.max_depth_min,
                self.optuna_param_grid.max_depth_max,
                step=self.optuna_param_grid.max_depth_step,
            ),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                self.optuna_param_grid.min_child_weight_min,
                self.optuna_param_grid.min_child_weight_max,
                step=self.optuna_param_grid.min_child_weight_step,
            ),
            "eta": trial.suggest_float("eta", self.optuna_param_grid.eta_min, self.optuna_param_grid.eta_max, log=True),
            # defines how selective algorithm is.
            "gamma": trial.suggest_float(
                "gamma", self.optuna_param_grid.gamma_min, self.optuna_param_grid.gamma_max, log=True
            ),
            # L2 regularization weight.
            "lambda": trial.suggest_float(
                "lambda", self.optuna_param_grid.lambda_min, self.optuna_param_grid.lambda_max, log=True
            ),
            # L1 regularization weight.
            "alpha": trial.suggest_float(
                "alpha", self.optuna_param_grid.alpha_min, self.optuna_param_grid.alpha_max, log=True
            ),
            # sampling ratio for training data.
            "subsample": trial.suggest_float(
                "subsample", self.optuna_param_grid.subsample_min, self.optuna_param_grid.subsample_max
            ),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", self.optuna_param_grid.colsample_bytree_min, self.optuna_param_grid.colsample_bytree_max
            ),
        }
        param["monotone_constraints"] = self.monotonicity_constraints
        param["num_class"] = self.num_class

        history = xgb.cv(param, dtrain, nfold=self.nfolds)
        if self.error == "reg:squarederror":
            return history["test-rmse-mean"].values[-1]
        if self.error == "binary:logistic":
            return history["test-logloss-mean"].values[-1]
        if self.error == "multi:softprob":
            return history["test-logloss-mean"].values[-1]


class XGBTimeSeriesCVHyperparameterTuner(XGBHyperparameterTuner, OptunaXGBParamGrid):
    def __init__(
        self,
        training_data,
        response,
        monotonicity_constraints=None,

    ) -> None:
        super().__init__(training_data, response)
        self.monotonicity_constraints = monotonicity_constraints

    def objective(self, trial):
        param = {
            "verbosity": self.optuna_param_grid.verbosity,
            "objective": self.optuna_param_grid.error,
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int(
                "max_depth",
                self.max_depth_min,
                self.max_depth_max,
                step=self.max_depth_step,
            ),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                self.min_child_weight_min,
                self.min_child_weight_max,
                step=self.min_child_weight_step,
            ),
            "eta": trial.suggest_float("eta", self.eta_min, self.eta_max, log=True),
            # defines how selective algorithm is.
            "gamma": trial.suggest_float(
                "gamma", self.gamma_min, self.gamma_max, log=True
            ),
            # L2 regularization weight.
            "lambda": trial.suggest_float(
                "lambda", self.lambda_min, self.lambda_max, log=True
            ),
            # L1 regularization weight.
            "alpha": trial.suggest_float(
                "alpha", self.alpha_min, self.alpha_max, log=True
            ),
            # sampling ratio for training data.
            "subsample": trial.suggest_float(
                "subsample", self.subsample_min, self.subsample_max
            ),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", self.colsample_bytree_min, self.colsample_bytree_max
            ),
        }
        param["monotone_constraints"] = self.monotonicity_constraints

        years = list(set(self.training_data['Year']))
        for year in years[1:]:
            train_x = self.training_data[self.training_data['Year'] < year]
            valid_x = self.training_data[self.training_data['Year'] == year]

            train_y = self.response.loc[train_x.index]
            valid_y = self.response.loc[valid_x.index]

            dtrain = xgb.DMatrix(train_x, label=train_y)
            dvalid = xgb.DMatrix(valid_x, label=valid_y)

            bst = xgb.train(param, dtrain)
            preds = bst.predict(dvalid)

            if self.error == "reg:squarederror":
                fold_error = mean_squared_error(preds, valid_y, squared=False)
            if self.error == "binary:logistic":
                fold_error =  log_loss(valid_y, preds)
            if self.error == "multi:softprob":
                fold_error =  log_loss(valid_y, preds)
            if self.error == "reg:tweedie":
                fold_error =  mean_tweedie_deviance(valid_y, preds, power=self.tweedie_power)

        error_list = [fold_error]
        return np.mean(error_list)
    
class XGBRoundCVHyperparameterTuner(XGBHyperparameterTuner, OptunaXGBParamGrid):
    def __init__(
        self,
        training_data,
        response,
        monotonicity_constraints=None,

    ) -> None:
        super().__init__(training_data, response)
        self.monotonicity_constraints = monotonicity_constraints
    
    def objective(self, trial):
        
        param = {
            "verbosity": self.verbosity,
            "objective": self.error,
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth": trial.suggest_int(
                "max_depth",
                self.max_depth_min,
                self.max_depth_max,
                step=self.max_depth_step,
            ),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                self.min_child_weight_min,
                self.min_child_weight_max,
                step=self.min_child_weight_step,
            ),
            "eta": trial.suggest_float("eta", self.eta_min, self.eta_max, log=True),
            # defines how selective algorithm is.
            "gamma": trial.suggest_float(
                "gamma", self.gamma_min, self.gamma_max, log=True
            ),
            # L2 regularization weight.
            "lambda": trial.suggest_float(
                "lambda", self.lambda_min, self.lambda_max, log=True
            ),
            # L1 regularization weight.
            "alpha": trial.suggest_float(
                "alpha", self.alpha_min, self.alpha_max, log=True
            ),
            # sampling ratio for training data.
            "subsample": trial.suggest_float(
                "subsample", self.subsample_min, self.subsample_max
            ),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", self.colsample_bytree_min, self.colsample_bytree_max
            ),
        }
        param["verbosity"] = self.verbosity
        param["objective"] = self.error
        param["monotone_constraints"] = self.monotonicity_constraints
        
        error_list = []
        years = list(set(self.training_data['Year']))
        for year in years:
            rounds = list(set(self.training_data[self.training_data['Year'] == year]['Round'])) 
            for round_num in rounds:
                if round_num > 5:
                    train_x = self.training_data[(self.training_data['Year'] < year) | ((self.training_data['Year'] == year) & (self.training_data['Round'] < round_num))].drop(columns = ['Year', 'Round'])
                    valid_x = self.training_data[(self.training_data['Year'] == year) & (self.training_data['Round'] == round_num)].drop(columns = ['Year', 'Round'])
                    train_y = self.response.loc[train_x.index]
                    valid_y = self.response.loc[valid_x.index]

                    dtrain = xgb.DMatrix(train_x, label=train_y)
                    dvalid = xgb.DMatrix(valid_x, label=valid_y)

                    bst = xgb.train(param, dtrain)
                    preds = bst.predict(dvalid)

                    error_functions = {
                        "reg:squarederror": lambda p, y: mean_squared_error(p, y, squared=False),
                        "binary:logistic": lambda p, y: log_loss(y, p, labels=[0, 1]),
                        "multi:softprob": lambda p, y: log_loss(y, p, labels=[0, 1]),
                        "reg:tweedie": lambda p, y: mean_tweedie_deviance(y, p, power=self.tweedie_power)
                    }
                    fold_error = error_functions[self.error](preds, valid_y)
                    error_list.append(fold_error)

        return np.mean(error_list)
