import os
import tarfile
import tempfile
from loguru import logger
import importlib
from distutils import dir_util

import yaml
import ulid
import joblib
import skfuzzy
import sklearn

from h1st.model.repository.storage.s3 import S3Storage
from h1st.model.repository.storage.local import LocalStorage

SEP = "::"


class ModelSerDe:
    STATS_PATH = "stats.joblib"
    METRICS_PATH = "metrics.joblib"
    RULE_DETAILS_PATH = "rule_details.joblib"
    RULE_ENGINE_PATH = "rule_engine.joblib"
    METAINFO_FILE = "METAINFO.yaml"

    def _is_builtin_class_instance(self, obj):
        return (
            obj.__class__.__module__ == "builtins"
            or obj.__class__.__module__ == "__builtin__"
        )

    def _serialize_basic_obj(self, d, path, obj_file):
        joblib.dump(d, os.path.join(path, obj_file))

    def _deserialize_basic_obj(self, path, obj_file):
        return joblib.load(os.path.join(path, obj_file))

    def _serialize_dict(self, d, path, dict_file):
        joblib.dump(d, path + "/%s" % dict_file)

    def _deserialize_dict(self, path, dict_file):
        return joblib.load(path + "/%s" % dict_file)

    def _get_rule_engine_type(self, rules):
        if isinstance(rules, skfuzzy.control.ControlSystemSimulation) or isinstance(
            rules, skfuzzy.control.controlsystem.ControlSystemSimulation
        ):
            return "skfuzzy"
        else:
            return "custom"

    def _get_model_type(self, model):
        if isinstance(model, sklearn.base.BaseEstimator):
            return "sklearn"
            
        if model is None:
            return "custom"

    def _get_supported_rule_engines(self):
        return set(["skfuzzy"])

    def _serialize_rule_engine(self, rules, path, rules_name="rules"):
        rule_engine_type = self._get_rule_engine_type(rules)

        if rule_engine_type == "skfuzzy":
            rules_path = "%s.joblib" % rules_name
            joblib.dump(rules, path + "/%s" % rules_path)
        else:
            raise ValueError("Unsupported model!!!")

        return rule_engine_type, rules_path

    def _deserialize_rule_engine(self, model, path, rule_engine_type, rules_path):
        if rule_engine_type == "skfuzzy":
            model = joblib.load(path + "/%s" % rules_path)
        else:
            model = joblib.load(path + "/%s" % rules_path)

        return model

    def _serialize_single_model(self, model, path, model_name="model"):
        model_type = self._get_model_type(model)

        if model_type == "sklearn":
            model_path = "%s.joblib" % model_name
            joblib.dump(model, path + "/%s" % model_path)
        elif model_type == "custom":
            model_path = model_name  # XXX
        else:
            raise ValueError("Unsupported model!!!")

        return model_type, model_path

    def _deserialize_single_model(self, model, path, model_type, model_path):
        if model_type == "sklearn":
            # This is a sklearn model
            model = joblib.load(path + "/%s" % model_path)
            # print(str(type(model)))
        elif model_type == "custom":
            model = None

        return model

    def serialize(self, model, path):
        """
        Serialize a H1ST model's model property to disk.

        :param model: H1ST Model
        :param path: path to save models to
        """
        from h1st.model.ml_model import MLModel
        from h1st.model.knowledge_model import RuleBasedModel

        meta_info = {}

        if model.metrics:
            logger.info("Saving metrics property...")
            meta_info["metrics"] = self.METRICS_PATH
            self._serialize_dict(model.metrics, path, self.METRICS_PATH)

        if model.stats is not None:
            logger.info("Saving stats property...")
            meta_info["stats"] = self.STATS_PATH
            self._serialize_dict(model.stats, path, self.STATS_PATH)

        if isinstance(model, MLModel):
            if model.base_model:
                logger.info("Saving model property...")
                if type(model.base_model) == list:
                    meta_info["models"] = []
                    for i, model in enumerate(model.base_model):
                        model_type, model_path = self._serialize_single_model(
                            model, path, "model_%d" % i
                        )
                        meta_info["models"].append(
                            {"model_type": model_type, "model_path": model_path}
                        )
                elif type(model.base_model) == dict:
                    meta_info["models"] = {}
                    for k, model in model.base_model.items():
                        model_type, model_path = self._serialize_single_model(
                            model, path, "model_%s" % k
                        )
                        meta_info["models"][k] = {
                            "model_type": model_type,
                            "model_path": model_path,
                        }
                else:
                    # this is a single model
                    model_type, model_path = self._serialize_single_model(
                        model.base_model, path
                    )
                    meta_info["models"] = [
                        {"model_type": model_type, "model_path": model_path}
                    ]
            else:
                logger.error(".base_model was not assigned.")

        elif isinstance(model, RuleBasedModel):

            if model.rule_details is not None:
                logger.info("Saving rule_details property...")
                meta_info["rule_details"] = self.RULE_DETAILS_PATH
                self._serialize_dict(model.rule_details, path, self.RULE_DETAILS_PATH)

            if model.rule_engine is not None:
                logger.info("Saving rule_engine property...")
                if type(model.rule_engine) in self._get_supported_rule_engines():
                    rule_engine_type, rules_path = self._serialize_rule_engine(
                        model.rule_engine, path
                    )
                    meta_info["rule_engine"] = [
                        {"rules_type": rule_engine_type, "rules_path": rules_path}
                    ]
                elif (
                    type(model.rule_engine) == list
                    and type(model.rule_engine[0]) in self._get_supported_rule_engines()
                ):
                    meta_info["rule_engine"] = []
                    for i, rules in enumerate(model.rule_engine):
                        rule_engine_type, rules_path = self._serialize_rule_engine(
                            rules, path, f"rules_{i}"
                        )
                        meta_info["rule_engine"].append(
                            {"rules_type": rule_engine_type, "rules_path": rules_path}
                        )
                elif (
                    type(model.rule_engine) == dict
                    and type(list(model.rule_engine.values())[0])
                    in self._get_supported_rule_engines()
                ):
                    meta_info["rule_engine"] = {}
                    for key, rules in model.rule_engine.items():
                        rule_engine_type, rules_path = self._serialize_rule_engine(
                            rules, path, f"rules_{i}"
                        )
                        meta_info["rule_engine"][key] = {
                            "rules_type": rule_engine_type,
                            "rules_path": rules_path,
                        }
                elif self._is_builtin_class_instance(model.rule_engine):
                    self._serialize_basic_obj(
                        model.rule_engine, path, self.RULE_ENGINE_PATH
                    )
                    meta_info["rule_engine"] = {
                        "rules_type": type(model.rule_engine),
                        "rule_path": self.RULE_ENGINE_PATH,
                    }
                else:
                    logger.warning(
                        (
                            "This rule engine is custom, so may not work well with "
                            "joblib which is the python package that we use to persist rules."
                        )
                    )
                    self._serialize_basic_obj(
                        model.rule_engine, path, self.RULE_ENGINE_PATH
                    )
                    meta_info["rule_engine"] = {
                        "rules_type": type(model.rule_engine),
                        "rule_path": self.RULE_ENGINE_PATH,
                    }

            else:
                logger.warning(".rule_engine was not assigned.")

        elif hasattr(model, "base_model"):
            logger.warning(
                (
                    "Your .base_model will not be persisted. "
                    "If you want to persist your .base_model, "
                    "must inherit from h1st.MLModel."
                )
            )

        elif hasattr(model, "rules"):
            logger.warning(
                (
                    "Your .rules will not be persisted. "
                    "If you want to persist your .rules, "
                    "must inherit from h1st.RuleBasedModel."
                )
            )

        if len(meta_info) == 0:
            logger.info(
                "Model persistence currently supports only stats, "
                "model and metrics properties."
            )
            logger.info(
                "Make sure you store stastistic in stats property, "
                "models in model property and model metrics in metrics one."
            )

        with open(os.path.join(path, self.METAINFO_FILE), "w") as file:
            yaml.dump(meta_info, file)

    def deserialize(self, model, path):
        """
        Populate a H1ST model's model property with saved atomic models.

        :param model: H1ST Model
        :param path: path to model folder
        """
        # Read METAINFO
        with open(os.path.join(path, self.METAINFO_FILE), "r") as file:
            meta_info = yaml.load(file, Loader=yaml.Loader)

        if "metrics" in meta_info.keys():
            model.metrics = self._deserialize_dict(path, self.METRICS_PATH)

        if "stats" in meta_info.keys():
            model.stats = self._deserialize_dict(path, self.STATS_PATH)

        if "models" in meta_info.keys():
            model_infos = meta_info["models"]
            org_model = model.base_model  # original model object from Model class
            if type(model_infos) == list:
                if len(model_infos) == 1:
                    # Single model
                    model_info = model_infos[0]
                    # print(model_info)
                    model_type = model_info["model_type"]
                    model_path = model_info["model_path"]
                    model.base_model = self._deserialize_single_model(
                        org_model, path, model_type, model_path
                    )
                else:
                    # A list of models
                    model.base_model = []
                    org_model = org_model or [None for _ in range(len(model_infos))]
                    for i, model_info in enumerate(model_infos):
                        model_type = model_info["model_type"]
                        model_path = model_info["model_path"]
                        model.base_model.append(
                            self._deserialize_single_model(
                                org_model[i], path, model_type, model_path
                            )
                        )

            elif type(model_infos) == dict:
                # A dict of models
                model.base_model = {}
                org_model = org_model or {k: None for k in model_infos.keys()}
                for model_name, model_info in model_infos.items():
                    model_type = model_info["model_type"]
                    model_path = model_info["model_path"]
                    model.base_model[model_name] = self._deserialize_single_model(
                        org_model[model_name], path, model_type, model_path
                    )
            else:
                raise ValueError("Not a valid H1ST Model METAINFO file!")

        if "rule_details" in meta_info.keys():
            model.rule_details = self._deserialize_dict(path, self.RULE_DETAILS_PATH)

        if "rule_engine" in meta_info.keys():
            rules_infos = meta_info["rule_engine"]
            org_rules = model.rule_engine
            if type(rules_infos) == list:
                if rules_infos[0]["rules_type"] in self._get_supported_rule_engines():
                    if len(rules_infos) == 1:
                        model.rule_engine = self._deserialize_rule_engine(
                            org_rules,
                            path,
                            rules_infos[0]["rules_type"],
                            rules_infos[0]["rules_path"],
                        )
                    else:
                        model.rule_engine = []
                        org_rules = org_rules or [None for _ in range(len(rules_infos))]
                        for i, model_info in enumerate(rules_infos):
                            model_type = model_info["model_type"]
                            model_path = model_info["model_path"]
                            model.rule_engine.append(
                                self._deserialize_rule_engine(
                                    org_rules[i],
                                    path,
                                    rules_infos[i]["rules_type"],
                                    rules_infos[i]["rules_path"],
                                )
                            )
            elif type(rules_infos) == dict:
                if "rules_type" in rules_infos:
                    if not self._is_builtin_class_instance(rules_infos["rules_type"]):
                        logger.warning(
                            (
                                "This rule engine is custom, so may not work well with "
                                "joblib which is the python package that we use to persist rules."
                            )
                        )
                    model.rule_engine = self._deserialize_basic_obj(
                        path, self.RULE_ENGINE_PATH
                    )
                else:
                    if (
                        list(rules_infos.values())[0]["rules_type"]
                        in self._get_supported_rule_engines()
                    ):
                        model.rule_engine = {}
                        org_rules = org_rules or {k: None for k in rules_infos.keys()}
                        for rules_name, rules_info in rules_infos.items():
                            model.rule_engine[
                                rules_name
                            ] = self._deserialize_rule_engine(
                                org_rules[rules_name],
                                path,
                                rules_info["rules_type"],
                                rules_info["rules_path"],
                            )
            else:
                raise ValueError("Not a valid H1ST Model METAINFO file!")


class ModelRepository:
    """
    Model repository allows user to persist and load model to different storage system.

    Model repository uses ``ModelSerDer`` to serialize a model into a temporary folder
    and then create a tar archive to store on storage. For loading, the repo retrieve
    the tar archive from the storage and then extract to temporary folder for restoring
    the model object.
    """

    _NAMESPACE = "_models"

    _DEFAULT_STORAGE = S3Storage

    def __init__(self, storage=None):
        if isinstance(storage, str) and "s3://" in storage:
            storage = storage.replace("s3://", "").strip("/") + "/"
            bucket, prefix = storage.split("/", 1)
            storage = S3Storage(bucket, prefix.strip("/"))
            self._NAMESPACE = ""
        elif isinstance(storage, str):  # local folder
            storage = LocalStorage(storage)
            self._NAMESPACE = ""

        self._storage = storage or ModelRepository._DEFAULT_STORAGE()
        self._serder = ModelSerDe()

    def persist(self, model, version=None):
        """
        Save a model to the model repository

        :param model: target model
        :param version: version name, leave blank for autogeneration
        :returns: model version
        """
        # assert isinstance(model, Model)
        # TODO: use version format: v_20200714-1203
        version = version or str(ulid.new())

        try:
            # serialize a model to a temporary folder and then clean up later
            tmpdir = tempfile.mkdtemp()
            serialized_dir = os.path.join(tmpdir, "serialized")
            tar_file = os.path.join(tmpdir, "model.tar")
            os.makedirs(serialized_dir)

            self._serder.serialize(model, serialized_dir)
            _tar_create(tar_file, serialized_dir)

            with open(tar_file, mode="rb") as f:
                self._storage.set_bytes(
                    self._get_key(model, version),
                    f.read(),
                )

                self._storage.set_obj(
                    self._get_key(model, "latest"),
                    version,
                )

                model.version = version
        finally:
            dir_util.remove_tree(tmpdir)

        return version

    def load(self, model, version=None):
        """
        Restore the model from the model repository

        :param model: target model
        :param version: version name, leave blank to load the latest version
        """
        # assert isinstance(model, Model)
        if version is None:
            version = self._storage.get_obj(self._get_key(model, "latest"))

        logger.info("Loading version %s ...." % version)

        try:
            tmpdir = tempfile.mkdtemp()
            serialized_dir = os.path.join(tmpdir, "serialized")
            tar_file = os.path.join(tmpdir, "model.tar")
            os.makedirs(serialized_dir)

            with open(tar_file, "wb") as f:
                f.write(self._storage.get_bytes(self._get_key(model, version)))

            _tar_extract(tar_file, serialized_dir)
            self._serder.deserialize(model, serialized_dir)
            model.version = version
        finally:
            # We get error from Tensorflow telling that it could not find the folder
            # Unsuccessful TensorSliceReader constructor: Failed to get matching files on
            # /var/folders/wb/40304xlx477cfjzbk386l2gr0000gn/T/tmpwcrvm2e2/model/weights:
            # Not found: /var/folders/wb/40304xlx477cfjzbk386l2gr0000gn/T/tmpwcrvm2e2/model; No such file or directory [Op:RestoreV2]
            #
            # dir_util.remove_tree(tmpdir)

            # instead, register the function to clean it up when the interpreter quits
            import atexit

            def clean_tmpdir(tmpdir):
                # print('Clean up %s'  % tmpdir)
                dir_util.remove_tree(tmpdir)

            atexit.register(clean_tmpdir, tmpdir=tmpdir)

    def delete(self, model, version):
        """
        Delete a model from model repository

        :param model: model instance or the model class
        :param version: target version
        """
        # assert isinstance(model, Model) or isinstance(model, type)
        assert version != "latest"  # magic key
        self._storage.delete(self._get_key(model, version))

    def download(self, model, version, path):
        """
        Download a model archive to local disk

        :param model: model instance or model class
        :param version: version name
        :param path: target folder to extract the model archive
        """
        with tempfile.NamedTemporaryFile(mode="wb") as f:
            f.write(self._storage.get_bytes(self._get_key(model, version)))
            f.flush()
            f.seek(0)

            _tar_extract(f.name, path)

        return path

    # TODO: list all versions

    def _get_key(self, model, version):
        model_class = model if isinstance(model, type) else model.__class__
        model_name = model_class.__module__ + "." + model_class.__name__

        key = f"{model_name}{SEP}{version}"

        if self._NAMESPACE:
            key = f"{self._NAMESPACE}{SEP}{key}"

        return key

    @classmethod
    def get_model_repo(cls, ref=None):
        """
        Retrieve the default model repository for the project

        :param ref: target model
        :returns: Model repository instance
        """
        if not hasattr(cls, "MODEL_REPO"):  # global ModelRepository.MODEL_REPO
            repo_path = None
            if ref is not None:
                # root module
                root_module_name = ""

                # find the first folder containing config.py to get MODEL_REPO_PATH
                for sub in ref.__class__.__module__.split("."):
                    root_module_name = (
                        sub if not root_module_name else root_module_name + "." + sub
                    )

                    try:
                        module = importlib.import_module(root_module_name + ".config")
                        repo_path = getattr(module, "MODEL_REPO_PATH", None)
                        break
                    except ModuleNotFoundError:
                        repo_path = None

            # in the new structure, the config file may be at root folder
            if not repo_path:
                try:
                    import config

                    repo_path = config.MODEL_REPO_PATH
                except ModuleNotFoundError:
                    repo_path = None

            if not repo_path:
                repo_path = os.environ.get("H1ST_MODEL_REPO_PATH", "")

            if not repo_path:
                raise RuntimeError("Please set MODEL_REPO_PATH in config.py")

            setattr(cls, "MODEL_REPO", ModelRepository(storage=repo_path))

        return getattr(cls, "MODEL_REPO")


def _tar_create(target, source):
    """
    Helper function to create a tar archive
    """
    with tarfile.open(target, "w:gz") as tf:
        tf.add(source, arcname="", recursive=True)

    return target


def _tar_extract(source, target):
    """
    Helper function to extract a tar archive
    """
    with tarfile.open(source) as tf:
        tf.extractall(target)
