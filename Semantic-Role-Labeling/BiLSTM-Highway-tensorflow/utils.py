import yaml
import json


class ConfigMeta(type):
    class __ConfigMeta:

        def __init__(self, is_new=False):
            self.is_new = is_new
            self.config = None
            self.description = None
            if is_new is False:
                self.config = self.parse_yaml(self.read_fname)

        def __call__(self, fname):
            self.is_new = False
            self.config = self.parse_yaml(fname)
            self.read_fname = fname

        def parse_yaml(self, path):
            config = self.parse_description_then_remove(path)
            return yaml.load(config)

        def parse_description_then_remove(self, path):
            self.description = {}
            config = ""
            with open(path, 'r',encoding='utf8') as infile:
                for line in infile.readlines():
                    config += line
            return config

        def to_dict(self):
            return self.config

        def get(self, name, default=None):
            try:
                return self.__getattr__(name)
            except KeyError as ke:
                return default

        def __getattr__(self, name):
            self._set_config()

            config_value = self.config[name]
            if type(config_value) == dict:
                return SubConfig(config_value, get_tag=name)
            else:
                return config_value

        def __repr__(self):
            if self.config is None:
                raise FileNotFoundError("No such files start filename")
            else:
                return f"Read config file name: {self.read_fname}\n" + json.dumps(self.config,ensure_ascii=False,indent=4)

        def _set_config(self):
            if self.config is None:
                self.is_new = False
                self.config = self.read_file(self.base_fname)

    instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = cls.__ConfigMeta(is_new=True)
        return cls.instance


class Config(metaclass=ConfigMeta):
    pass


class SubConfig:
    def __init__(self, *args, get_tag=None):
        self.get_tag = get_tag
        self.__dict__ = dict(*args)

    def __getattr__(self, name):
        if name in self.__dict__["__dict__"]:
            item = self.__dict__["__dict__"][name]
            return item
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name != "get" and name != "__dict__":
            origin_config = Config.config
            gets = self.get_tag.split(".")
            for get in gets:
                origin_config = origin_config[get]

            origin_config[name] = value

    def get(self, name, default=None):
        return self.__dict__["__dict__"].get(name, default)

    def to_dict(self):
        return self.__dict__["__dict__"]

    def __repr__(self):
        return json.dumps(self.__dict__["__dict__"], indent=4)
