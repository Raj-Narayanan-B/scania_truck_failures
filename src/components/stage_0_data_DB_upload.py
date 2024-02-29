import os
import shutil  # type: ignore
import pandas as pd
from src import logger
from src.config.configuration_manager import ConfigurationManager
from src.constants import REPO, BUCKET, TEST_DATA, TRAIN_DATA, TOKEN
from src.utils import (DB_data_uploader, load_yaml, save_yaml, file_lineage_s3_files_refresher,
                       file_lineage_adder, file_lineage_updater, file_lineage_reverse_updater)
from dagshub import get_repo_bucket_client
from dagshub.auth.tokens import add_app_token
from ensure import ensure_annotations


class s3_handle(ConfigurationManager):
    def __init__(self) -> None:
        super().__init__()
        add_app_token(TOKEN)
        self.s3 = get_repo_bucket_client(REPO + '/' + BUCKET)
        self.temp_dir = self.get_data_path_config().temp_dir_root

    def s3_data_upload(self, key: str, file: pd.DataFrame):
        """
            A function to upload a file to the project's dagshub S3 bucket

            Parameters
            ----------

            key: str; default=None
                key refers to the name, the file should be saved as in the s3 bucket
            filepath: str; default=None
                filepath refers to the local filepath where the file is located
        """
        prediction_path = self.temp_dir + '/' + key
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
        file.to_csv(prediction_path, index=False)
        self.s3.upload_file(Bucket=BUCKET,
                            Filename=prediction_path,
                            Key="Prediction_"+key
                            )
        logger.info(
            f"{os.path.basename(prediction_path)} uploaded to S3 successfully")
        shutil.rmtree(self.temp_dir)

    def s3_data_download(self, key: str, filepath: str):
        """
            A function to download a file from the project's dagshub S3 bucket

            Parameters
            ----------

            key: str; default=None
                key refers to the filename in s3 bucket.
            filepath: str; default=None
                filepath refers to the local filepath where the downloaded file should be saved
        """
        file_path, file_name = os.path.split(filepath)
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        self.s3.download_file(
            Bucket=BUCKET,
            Key=key,
            Filename=filepath
        )


class file_lineage_component(s3_handle):
    def __init__(self):
        super().__init__()
        self.astra_dB_data_config = self.get_astra_dB_data_config()
        self.data_config = self.get_data_path_config()

    @ensure_annotations
    def file_lineage_tracker(self, update_list: list = None, reverse_update_list: list = None,
                             add_list: list = None):
        files_in_s3_ = self.s3.list_objects_v2(Bucket=BUCKET)['Contents']

        if os.path.exists(self.data_config.data_from_s3):
            old_files_ = load_yaml(self.data_config.data_from_s3)

            old_files = file_lineage_s3_files_refresher(
                files_in_s3=files_in_s3_, old_files=old_files_)

            old_files_for_model_training = old_files['files_for_model_training']
            old_files_predicted = old_files['files_predicted']
            old_files_to_predict = old_files['files_to_predict']

            if add_list:
                old_files_predicted = file_lineage_adder(
                    add_list, old_files_predicted)

            if update_list:
                old_files_predicted, old_files_to_predict = file_lineage_updater(update_list=update_list,
                                                                                 old_files_predicted=old_files_predicted,
                                                                                 old_files_to_predict=old_files_to_predict)

            if reverse_update_list:
                old_files_predicted, old_files_to_predict = file_lineage_reverse_updater(reverse_update_list=reverse_update_list,
                                                                                         old_files_predicted=old_files_predicted,
                                                                                         old_files_to_predict=old_files_to_predict)

            files_from_s3_dict = {}
            files_from_s3_dict['files_for_model_training'] = old_files_for_model_training
            files_from_s3_dict['files_to_predict'] = old_files_to_predict
            files_from_s3_dict['files_predicted'] = old_files_predicted

            save_yaml(files_from_s3_dict,
                      filepath=self.data_config.data_from_s3,
                      mode='w')

        # This else block is the initial file creation of the file_lineage file from S3.
        # The files present in S3 will all be tracked under the  "files_to_predict" section intially.
        # Once the file_lineage file is created, the files(if any) that have been used for prediction can ->
        # -> be updated using the "update_predicted_files" parameter to move those file to the "files_predicted" section.
        # It will not contain the details of files entered through "Bulk/Batch" prediction.
        # The tracking of files entered through "Bulk/Batch" prediction is done only after this initial creation of file_lineage file.
        else:
            with open(self.data_config.data_from_s3, 'w'):
                files_from_s3_dict = {}
                files_from_s3_dict['files_for_model_training'] = {}
                files_from_s3_dict['files_to_predict'] = {}
                files_from_s3_dict['files_predicted'] = {}
                file_counter = 1
                for i in range(len(files_in_s3_)):
                    if files_in_s3_[i]['Key'] == TEST_DATA or files_in_s3_[i]['Key'] == TRAIN_DATA:
                        file_name = "training_set" if files_in_s3_[
                            i]['Key'] == TRAIN_DATA else "testing_set"
                        files_from_s3_dict['files_for_model_training'][file_name] = files_in_s3_[
                            i]['Key']
                    else:
                        file_name = f"S3_file_{file_counter}"
                        if files_in_s3_[i]['Key'].startswith("Prediction"):
                            continue
                        else:
                            value = files_in_s3_[i]['Key']
                        files_from_s3_dict['files_to_predict'][file_name] = value
                        file_counter += 1
                save_yaml(file=files_from_s3_dict,
                          filepath=self.data_config.data_from_s3,
                          mode='w')


class data_db_uploader_component(file_lineage_component):
    def __init__(self) -> None:
        super().__init__()

    def data_db_upload(self):

        train_data_config = [[self.data_config.temp_train_data],

                             [self.astra_dB_data_config.train_data_1_secure_connect_bundle,
                                 self.astra_dB_data_config.train_data_1_token,
                                 self.astra_dB_data_config.train_data_1_key_space,
                                 self.astra_dB_data_config.train_data_1_table,
                                 self.data_config.temp_train_data1],

                             [self.astra_dB_data_config.train_data_2_secure_connect_bundle,
                                 self.astra_dB_data_config.train_data_2_token,
                                 self.astra_dB_data_config.train_data_2_key_space,
                                 self.astra_dB_data_config.train_data_2_table,
                                 self.data_config.temp_train_data2],

                             [self.astra_dB_data_config.train_data_3_secure_connect_bundle,
                                 self.astra_dB_data_config.train_data_3_token,
                                 self.astra_dB_data_config.train_data_3_key_space,
                                 self.astra_dB_data_config.train_data_3_table,
                                 self.data_config.temp_train_data3]]

        test_data_config = [[self.data_config.temp_test_data],

                            [self.astra_dB_data_config.test_data_1_secure_connect_bundle,
                            self.astra_dB_data_config.test_data_1_token,
                            self.astra_dB_data_config.test_data_1_key_space,
                            self.astra_dB_data_config.test_data_1_table,
                            self.data_config.temp_test_data1],

                            [self.astra_dB_data_config.test_data_2_secure_connect_bundle,
                            self.astra_dB_data_config.test_data_2_token,
                            self.astra_dB_data_config.test_data_2_key_space,
                            self.astra_dB_data_config.test_data_2_table,
                            self.data_config.temp_test_data2],

                            [self.astra_dB_data_config.test_data_3_secure_connect_bundle,
                            self.astra_dB_data_config.test_data_3_token,
                            self.astra_dB_data_config.test_data_3_key_space,
                            self.astra_dB_data_config.test_data_3_table,
                            self.data_config.temp_test_data3]]

        for config in [train_data_config, test_data_config]:
            self.s3_data_download(key=os.path.basename(config[0][0]),
                                  filepath=config[0][0])
            DB_data_uploader(config)

        shutil.rmtree(self.data_config.temp_dir_root)
        logger.info("Temporary directory removed")


# data_db_obj = data_db_uploader_component()
# data_db_obj.data_db_upload()
