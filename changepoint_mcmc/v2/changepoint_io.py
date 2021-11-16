"""
Pipeline to handle model fitting from data extraction to saving results
"""

MODEL_SAVE_DIR = ''
MODEL_DATABASE_PATH = ''

class fit_handler():
    def __init__(self,
                data_dir,
                dig_in_num,
                experiment_name = None,
                model_params_path = None,
                preprocess_params_path = None,
                ):
        """
        dig_in_num: integer value, or 'all'
                        - There should be a way to cross-reference whether
                            the model will accept a particular array type
        """

        if experiment_name is None:
            raise Exception('Please specify an experiment name')
        self.model_save_base_dir = MODEL_SAVE_DIR
        self.model_save_dir = os.path.join(self.model_save_base_dir, experiment_name)
        self.model_database_path = MODEL_DATABASE_PATH

        self.dat_handler = database_handler()

        if model_params_path is None:
            print('MODEL_PARAMS will have to be set')
        else: 
            self.set_model_params(file_path = model_params_path)

        if preprocess_params_path is None:
            print('PREPROCESS_PARAMS will have to be set')
        else: 
            self.set_preprocess_params(file_path = preprocess_params_path)


    ########################################
    ## SET PARAMS
    ########################################

    def set_model_params(self, 
                        states, 
                        fit, 
                        samples, 
                        file_path = None): 

        if not file_path is None:
            self.model_params = dict(zip(['states','fit','samples'], [states,fit,samples]))
        else:
            # Load json and save dict

    def set_preprocess_params(self, 
                            time_lims, 
                            bin_width, 
                            data_tranform,
                            file_path = None): 

        if not file_path is None:
            self.preprocess_params = \
                    dict(zip(['time_lims','bin_width','data_transform'], 
                        [time_lims, bin_width, data_transform]))
        else:
            # Load json and save dict


    ########################################
    ## SET PIPELINE FUNCS
    ########################################

    def preprocess_selector():
        """
        Preprocessing can be set manually but it is preferred to go 
            through preprocess selector
        Function to return preprocess function based off of input flag 
        """
        # self.set_preprocessor(...)
        pass

    def set_preprocessor(self, preprocessing_func):
        """
        fit_handler.set_preprocessor(changepoint_preprocess.preprocess_single_taste)
        """

    def model_selector():
        """
        Function to return model based off of input flag 
        """
        # self.set_model(...)
        pass

    def set_model(self, model):
        """
        Models can be set manually but it is preferred to go through model selector
        fit_handler.set_model(changepoint_model.single_taste_poisson)
        """

    def set_inference(self, inference_func):
        """
        fit_handler.set_inference(changepoint_model.run_inference)
        """

    ########################################
    ## PIPELINE FUNCS 
    ########################################
    def load_spike_trains(self):
        #print('Loading spike trains from {}, dig_in {}')
        #self.data = ...
        pass

    def preprocess_data(self):
        if 'data' not in dir(self):
            self.load_spike_trains()
        #print('Preprocessing spike trains, preprocessing func: <{}>')
        #self.preprocessed_data = ...
        pass

    def create_model(self):
        if 'preprocessed_data' not in dir(self):
            self.preprocess_data()
        #print(' Generating Model, model func: <{}>')
        #self.model = ...
        pass

    def run_inference(self):
        if 'model' not in dir(self):
            self.create_model()
        #print('Running inference, inference func: <{}>')
        #self.inference_outs = ...
        pass

    def save_fit_output(self):
        if 'inference_outs' not in dir(self):
            self.run_inference()
        #print('Saving inference output to {}')
        

class database_handler():

    def __init__(self):
        self.fit_exists = None

    def check_exists():
        if self.fit_exists is None:
        else:
            return self.fit_exists

    def write_to_databse()
        #print('Writing inference out, run params {}')
