

import threading
import abc
from enum import Enum
import json
from Base.Parameters import Parameters




class SingletonType(type):
    _instance_lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        # if cls._reset == 1:
        #     with SingletonType._instance_lock:
        #         if not hasattr(cls, "_instance"):
        #             cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        #         cls._reset = 0

        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance

    def __reset__(cls, *args, **kwargs):
        del cls._instance
        with SingletonType._instance_lock:
            cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


# class SimEnv(object, metaclass=SingletonType):
class SimEnv(object):
    __clock__ = 0
    obj_list = None
    runing = False

    state = Parameters.SimSTATEType.INIT # 0 stop 1 run 2 pause
    logger = None

    # def reset(self):
    #     self._instance = self.__class__.__reset__()

    def set_logging(self, logger):
        logger.debug('this is a logger debug message')
        logger.info('this is a logger info message')
        logger.warning('this is a logger warning message')
        logger.error('this is a logger error message')
        logger.critical('this is a logger critical message')

        self.logger = logger




    @staticmethod
    def get_sim():
        if SimEnv.sim_self:
            return SimEnv.sim_self
        SimEnv.sim_self = SimEnv()

    def __init__(self):
        self.asd = 10
        self.__event_list__ = []
        self.__clock__ = 0
        self.obj_list = []
        self.runing = False
        self.state = Parameters.SimSTATEType.INIT


    def register_obj(self, obj):
        if obj.id == -1:
            obj.id = len(self.obj_list)
            self.obj_list.append(obj)

    #二分插入
    def appendEvent(self, event):
        le = 0
        ri = len(self.__event_list__) - 1
        # index = 0
        mark = False
        try:
            while ri >= le:
                mid = (le + ri) // 2
                if event['dest'] == self.__event_list__[mid]['dest']:
                    mark = True
                    while event['dest'] == self.__event_list__[mid]['dest']:
                        mid += 1
                        if mid >= len(self.__event_list__):
                            break
                    # index = mid
                    break
                elif event['dest'] > self.__event_list__[mid]['dest']:
                    le = mid + 1
                    # index = mid + 1
                else:
                    ri = mid - 1
                    # index = mid - 1
            if le != ri+1 and mark == False:
                raise Exception("二分查找点 佐和尤佳衣应该是一致的")
            if mark:
                self.__event_list__.insert(mid, event)
            else:
                self.__event_list__.insert(le, event)
        except IndexError as e:
            print(e)


    __event_list__ = []
    '''
     {
        'target':target,
        'src':src,
        'data':data,
        'info_type':info_type,
        'dest':SimEnv.clock+0.001,
        'log':"",
    }
    '''

    def process_event(self, event):
        log_info = {}
        src_obj = self.obj_list[event['src']]
        target_obj = self.obj_list[event['target']]
        if self.logger:
            log_info['clock - dest'] = '{}-{}'.format(self.clock, event['dest'])
            log_info['log'] = event['log']
            log_info['event_type'] =str(event['data']['event_type'])
            log_info['src--target'] = "{}--{}".format("{}:{}".format(src_obj.__class__.__name__, src_obj.id),
                                                      "{}:{}".format(src_obj.__class__.__name__, src_obj.id))
            # log_info['target'] =
        # print(event)
        if (event['dest'] < self.clock):
            if self.logger:
                log_info['error'] = "迟到事件"

                self.logger.error(json.dumps(log_info, ensure_ascii=False))
                pass
            raise Exception('clock 前期事件')
        if (event['info_type'] == Parameters.SimEventType.SEND):
            if self.logger:
                self.logger.info(log_info)
            self.clock = event['dest']
            target_obj.process_event(event)
            return
        if (event['info_type'] == Parameters.SimEventType.BROADCAST):
            if self.logger:
                self.logger.info(log_info)
            self.clock = event['dest']
            for obj in self.obj_list:
                try:
                    obj.process_event(event)
                except AttributeError as e:
                    print(e)
            return
            pass

        pass

    def run(self):
        if self.state == Parameters.SimSTATEType.INIT:
            for obj in self.obj_list:
                obj.init()

        self.state = Parameters.SimSTATEType.RUN
        while self.state == Parameters.SimSTATEType.RUN:
            '''
            事件处理
            '''
            if len(self.__event_list__) > 0:
                event = self.__event_list__.pop(0)
                self.process_event(event)
            else:
                break
            pass




    @property
    def clock(self):
        return self.__clock__

    @clock.setter
    def clock(self, value):
        self.__clock__ = value
        return self.__clock__



if __name__ == '__main__':
    # asd = T()
    # asd.o()
    # T.id = 2
    # asd.o()
    # asd = SimEnv()
    # print(asd.asd)
    # asd.asd = 20
    # print(asd.asd)
    # print(SimEnv().asd)
    pass
