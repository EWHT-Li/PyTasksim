import abc
from Base.SimEnv import SimEnv
from Base.Parameters import Parameters

class BaseObj(object, metaclass = abc.ABCMeta):
    def __init__(self, env: SimEnv):
        self.id = -1
        self.env = env
        env.register_obj(self)
        # self.init()

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def process_event(self, event):
        pass

    def send(self, target, data=None, info_type=Parameters.SimEventType.SEND, dest=None, log=''):
        if dest == None:
            dest = self.env.clock + 0.0001
        event = {
            'target':target,
            'src':self.id,
            'data':data,
            'info_type':info_type,
            'dest':dest,
            'log':log
        }
        self.env.appendEvent(event)
        pass

    def broadSend(self, data=None, info_type=Parameters.SimEventType.BROADCAST, dest=None, log=''):
        if dest == None:
            dest = self.env.clock + 0.0001
        event = {
            'src': self.id,
            'data': data,
            'info_type': info_type,
            'dest': dest,
            'log': log,
            'target':0,
        }
        self.env.appendEvent(event)
        pass


class Base(BaseObj):
    def __init__(self, env: SimEnv):
        self.id = -1
        self.env = env
        env.register_obj(self)

    def init(self):
        pass

    def process_event(self, event):
        pass