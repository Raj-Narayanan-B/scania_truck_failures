from src.constants import CONFIG_PATH, PARAMS_PATH, SCHEMA_PATH
from src.entity.entity_config import (AstraDBDataConf,
                                      DataPathConf,
                                      Stage1ProcessingConf,
                                      DataValidationConf,
                                      Stage2ProcessingConf,
                                      PreprocessorConf,
                                      DataSplitConf,
                                      ModelTrainerConf,
                                      ModelMetricsConf)
from src.utils import load_yaml, crypter
# from dvc import repo
# from github import Github, Auth
import os
# import mlflow.environment_variables


class ConfigurationManager:
    def __init__(self):
        self.config = CONFIG_PATH
        self.config_path = load_yaml(CONFIG_PATH)
        self.params = PARAMS_PATH
        self.params_path = load_yaml(PARAMS_PATH)
        self.schema = SCHEMA_PATH
        self.schema_path = load_yaml(SCHEMA_PATH)
        os.environ['MLFLOW_TRACKING_URI'] = crypter(encrypt_or_decrypt='decrypt', file_name='mlflow')['MLFLOW_TRACKING_URI']
        os.environ['MLFLOW_TRACKING_PASSWORD'] = crypter(encrypt_or_decrypt='decrypt', file_name='mlflow')['MLFLOW_TRACKING_PASSWORD']
        os.environ['MLFLOW_TRACKING_USERNAME'] = crypter(encrypt_or_decrypt='decrypt', file_name='mlflow')['MLFLOW_TRACKING_USERNAME']
        # mlflow.environment_variables.MLFLOW_TRACKING_URI = crypter(encrypt_or_decrypt='decrypt', file_name='mlflow')['MLFLOW_TRACKING_URI']  # noqa
        # mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = crypter(encrypt_or_decrypt='decrypt', file_name='mlflow')['MLFLOW_TRACKING_PASSWORD']  # noqa
        # mlflow.environment_variables.MLFLOW_TRACKING_USERNAME = crypter(encrypt_or_decrypt='decrypt', file_name='mlflow')['MLFLOW_TRACKING_USERNAME']  # noqa

    # def git_dvc_track(self, file_to_track: list):
    #     # os.environ[]
    #     dvc_config = crypter(encrypt_or_decrypt='decrypt', file_name='dvc')

    #     # git_repo = scm.SCM(root_dir=os.getcwd())
    #     dvc_repo = repo.Repo(url=dvc_config['DVC_ENDPOINT_URL'],
    #                          remote=dvc_config['DVC_REMOTE_NAME'])

    #     remote = dvc_config['DVC_REMOTE_NAME']
    #     dvc_repo.config['remote'][remote]['url'] = dvc_config['DVC_URL']
    #     dvc_repo.config['remote'][remote]['endpointurl'] = dvc_config['DVC_ENDPOINT_URL']
    #     dvc_repo.config['remote'][remote]['access_key_id'] = dvc_config['DVC_ACCESS_KEY_ID']
    #     dvc_repo.config['remote'][remote]['secret_access_key'] = dvc_config['DVC_SECRET_ACCESS_KEY']

    #     for i in range(len(file_to_track)):
    #         dvc_repo.add(file_to_track[i])
    #         # git_repo.add(file_to_track[i]+'.dvc')

    #         dvc_repo.commit(force=True)
    #         # git_repo.commit(os.path.basename(file_to_track[i])+" tracked with dvc")

    #         dvc_repo.push()
    #         # git_repo.push()
    #     print("DVC Push complete")
    #     git_config = crypter(encrypt_or_decrypt='decrypt', file_name='git')
    #     user_name = git_config['GITHUB_USERNAME']
    #     repo_name = git_config['REPO_NAME']
    #     pat = git_config['PAT']

    #     # Initialize Git Repo
    #     auth = Auth.Token(pat)
    #     git = Github(auth=auth)
    #     repo_ = git.get_user(user_name).get_repo(repo_name)
    #     all_files = []
    #     contents = repo_.get_contents("")
    #     while contents:
    #         file_content = contents.pop(0)
    #         if file_content.type == "dir":
    #             contents.extend(repo_.get_contents(file_content.path))
    #         else:
    #             file = file_content
    #             all_files.append(str(file).replace('ContentFile(path="', '').replace('")', ''))

    #     # Initialize git commit-push for all files provided
    #     for i in range(len(file_to_track)):
    #         with open(file_to_track[i], 'r') as file:
    #             content = file.read()
    #         # git_prefix = 'folder1/'
    #         git_file = file_to_track[i] + '.dvc'
    #         contents = repo_.get_contents(git_file)
    #         if contents.path in all_files:
    #             # contents = repo_.get_contents(git_file)
    #             repo_.update_file(contents.path, os.path.basename(file_to_track[i])+" tracked with dvc",
    #                               content, contents.sha, branch="main")
    #             print(git_file + ' UPDATED')
        # git_list = []
        # for i in range(len(git_repo.status()[1:])):
        #     for j in range(len(git_repo.status()[1:][i])):
        #         git_list.append(git_repo.status()[1:][i][j])
        # new_git_list = [item for item in git_list if not item.startswith('.github/workflows')]

        # git_repo.add_commit(paths=new_git_list,
        #                     message='test_commit_via_add_commit')

    def get_astra_dB_data_config(self) -> AstraDBDataConf:
        config = self.config_path['astra_dB_data_config']
        data_ingestion = AstraDBDataConf(
            train_data_1_secure_connect_bundle=os.path.abspath(config['train_data1']['secure_connect_bundle']),
            train_data_1_token=os.path.abspath(config['train_data1']['token']),
            train_data_1_key_space=config['train_data1']['key_space_train_data_1'],
            train_data_1_table=config['train_data1']['table_train_data_1'],

            train_data_2_secure_connect_bundle=os.path.abspath(config['train_data2']['secure_connect_bundle']),
            train_data_2_token=os.path.abspath(config['train_data2']['token']),
            train_data_2_key_space=config['train_data2']['key_space_train_data_2'],
            train_data_2_table=config['train_data2']['table_train_data_2'],

            train_data_3_secure_connect_bundle=os.path.abspath(config['train_data3']['secure_connect_bundle']),
            train_data_3_token=os.path.abspath(config['train_data3']['token']),
            train_data_3_key_space=config['train_data3']['key_space_train_data_3'],
            train_data_3_table=config['train_data3']['table_train_data_3'],
            # train_data_3_path=config.train_data3.path,

            test_data_1_secure_connect_bundle=os.path.abspath(config['test_data1']['secure_connect_bundle']),
            test_data_1_token=os.path.abspath(config['test_data1']['token']),
            test_data_1_key_space=config['test_data1']['key_space_test_data_1'],
            test_data_1_table=config['test_data1']['table_test_data_1'],
            # test_data_1_path=config.test_data1.path,

            test_data_2_secure_connect_bundle=os.path.abspath(config['test_data2']['secure_connect_bundle']),
            test_data_2_token=os.path.abspath(config['test_data2']['token']),
            test_data_2_key_space=config['test_data2']['key_space_test_data_2'],
            test_data_2_table=config['test_data2']['table_test_data_2'],
            # test_data_2_path=config.test_data2.path,

            test_data_3_secure_connect_bundle=os.path.abspath(config['test_data3']['secure_connect_bundle']),
            test_data_3_token=os.path.abspath(config['test_data3']['token']),
            test_data_3_key_space=config['test_data3']['key_space_test_data_3'],
            test_data_3_table=config['test_data3']['table_test_data_3'],
            # test_data_3_path=config.test_data3.path,

            root_directory=os.path.abspath(config['root_dir'])
        )
        return data_ingestion

    def get_data_path_config(self) -> DataPathConf:
        config = self.config_path['data_path_config']
        data_path_config = DataPathConf(
            train_data1=os.path.abspath(config['train_data1']),
            train_data2=os.path.abspath(config['train_data2']),
            train_data3=os.path.abspath(config['train_data3']),
            test_data1=os.path.abspath(config['test_data1']),
            test_data2=os.path.abspath(config['test_data2']),
            test_data3=os.path.abspath(config['test_data3']),

            final_test_data=os.path.abspath(config['final_test_data']),
            prediction_data=os.path.abspath(config['predicted_data']),

            temp_train_data=os.path.abspath(config['temp_train_data']),
            temp_test_data=os.path.abspath(config['temp_test_data']),

            temp_train_data1=os.path.abspath(config['temp_train_data1']),
            temp_train_data2=os.path.abspath(config['temp_train_data2']),
            temp_train_data3=os.path.abspath(config['temp_train_data3']),
            temp_test_data1=os.path.abspath(config['temp_test_data1']),
            temp_test_data2=os.path.abspath(config['temp_test_data2']),
            temp_test_data3=os.path.abspath(config['temp_test_data3']),

            temp_dir_root=os.path.abspath(config['temp_root_dir']),
            data_from_s3=os.path.abspath(config['data_loaded_from_s3_so_far'])
        )
        return (data_path_config)

    def get_stage1_processing_config(self) -> Stage1ProcessingConf:
        config = self.config_path['data_stage_1_processing_config']
        stage1_processing_config = Stage1ProcessingConf(
            root_dir=os.path.abspath(config['root_dir']),
            train_data_path=os.path.abspath(config['train_data_path']),
            test_data_path=os.path.abspath(config['test_data_path'])
        )
        return stage1_processing_config

    def get_data_validation_config(self) -> DataValidationConf:
        config = self.config_path['data_validation_config']
        data_validation_config = DataValidationConf(
            root_dir=os.path.abspath(config['root_dir']),
            validated_data=os.path.abspath(config['validated_data'])
        )
        return data_validation_config

    def get_stage2_processing_config(self) -> Stage2ProcessingConf:
        config = self.config_path['data_stage_2_processing_config']
        stage2_processing_config = Stage2ProcessingConf(
            root_dir=os.path.abspath(config['root_dir']),
            train_data_path=os.path.abspath(config['train_data_path']),
            test_data_path=os.path.abspath(config['test_data_path'])
        )
        return stage2_processing_config

    def get_preprocessor_config(self) -> PreprocessorConf:
        config = self.config_path['preprocessor']
        preprocessor_config = PreprocessorConf(
            root_dir=os.path.abspath(config['root_dir']),
            preprocessor_path=os.path.abspath(config['preprocessor_path'])
        )
        return preprocessor_config

    def get_data_split_config(self) -> DataSplitConf:
        config = self.config_path['data_split_config']
        data_split_config = DataSplitConf(
            root_dir=os.path.abspath(config['root_dir']),
            train_path=os.path.abspath(config['train_batch_path']),
            test_path=os.path.abspath(config['test_batch_path'])
        )
        return data_split_config

    def get_model_config(self) -> ModelTrainerConf:
        config = self.config_path['model_trainer']
        model_config = ModelTrainerConf(
            root_dir=os.path.abspath(config['root_dir']),
            model_path=os.path.abspath(config['model_path']),
            hp_model_path=os.path.abspath(config['hp_model_path_']),
            final_estimator_path=os.path.abspath(config['final_estimator']),
            stacking_classifier_path=os.path.abspath(config['stacking_classifier']),
            voting_classifier_path=os.path.abspath(config['voting_classifier'])
        )
        return model_config

    def get_metric_config(self) -> ModelMetricsConf:
        config = self.config_path['model_metrics']
        metrics_config = ModelMetricsConf(
            root_dir=os.path.abspath(config['root_dir']),
            metrics=os.path.abspath(config['metrics']),
            best_metric=os.path.abspath(config['best_metric']),
            model_trial_study_df=os.path.abspath(config['model_trial_study_df'])
        )
        return metrics_config


# obj = ConfigurationManager()
# print(obj.get_data_ingestion_config().train_data_2_token)
