from Base.Parameters import Parameters
import numpy as np

class Machine(object):
    _id = -1
    _mips = -1
    # _numberOfPes = -1
    # _ram = -1
    _bw = -1



    state = None
    task_list = None
    over_task_list = None

    def __init__(self, vmconfigure, parent, share_model=None):
        if share_model == None:
            share_model = Parameters.ShareModel
        self._mips = vmconfigure['mips']
        # self._numberOfPes = vmconfigure['numberOfPes']
        # self._ram = vmconfigure['ram']
        self._bw = vmconfigure['bw']
        self.task_list = []
        self.over_task_list = []

        self.c_mips = 0
        # self.c_numberOfPes = self._numberOfPes
        # self.c_ram = self._ram
        self.c_bw = 0
        self.parent = parent
        self.env = parent.env
        self.state = Parameters.VMState.STOP

        self.task = None
        self.recode_list = []
        # self.vm_state = Parameters.VMState.STOP
        self.share_model = share_model

        self._price = None
        pass

    def recodeState(self):
        item = {
            'time':self.env.clock
        }
        item.update(self.a_source)
        self.recode_list.append(item)

    def matchTask(self, task):
        mark = True
        if self.share_model == Parameters.VMState.TIMESHARE:
            for s in Parameters.BaseSource:
                if task.need_source[s] > self.a_source[s]:
                    return False
        elif self.share_model == Parameters.VMState.SPACESHARE:
            for s in Parameters.BaseSource:
                if task.need_source[s] > self.source[s]:
                    return False
        # if self.share_model == Parameters.VMState.SPACESHARE:
        #     mark = mark and (self.state == Parameters.VMState.STOP)
        return mark

    def executeTask(self, task):
        if self.share_model == Parameters.VMState.TIMESHARE:
            if self.matchTask(task) and (self.state == Parameters.VMState.STOP):
                # self.task = task
                self.task_list.append(task)
                self.c_mips += task.mips
                self.c_bw += task.bw
                task.start_time = self.env.clock
                self.state = Parameters.VMState.RUN
                return task.duration
            raise Exception('超出资源需求的任务')
            return -1
        elif self.share_model == Parameters.VMState.SPACESHARE:
            if self.matchTask(task):
                if self.state == Parameters.VMState.STOP:
                    # self.task = task
                    self.task_list.append(task)
                    self.c_mips += task.mips
                    self.c_bw += task.bw
                    task.start_time = self.env.clock
                    dur = 0
                    dur += task.mips_length/self.mips
                    dur += task.bw_length/self.bw
                    self.state = Parameters.VMState.RUN
                    return dur
                elif self.state == Parameters.VMState.RUN:
                    self.task_list.append(task)
                    return -1
            raise Exception('超出资源需求的任务')
            return -1

    def continueExeTask(self):
        if self.share_model == Parameters.VMState.TIMESHARE:
            raise Exception('未加入TimeShare')
        elif self.share_model == Parameters.VMState.SPACESHARE:
            if self.state == Parameters.VMState.STOP:
                task = self.task_list[0]
                self.c_mips += task.mips
                self.c_bw += task.bw
                task.start_time = self.env.clock
                dur = 0
                dur += task.mips_length / self.mips
                dur += task.bw_length / self.bw
                self.state = Parameters.VMState.RUN
                return dur
            else:
                raise Exception('错误调用，还有任务在执行')

    def tryExcuteTask(self, task):
        if self.share_model == Parameters.VMState.TIMESHARE:
            if self.matchTask(task):
                return task.duration
            raise Exception('超出资源需求的任务')
            return False
        elif self.share_model == Parameters.VMState.SPACESHARE:
            if self.matchTask(task):
                dur = 0
                dur += task.mips_length/self.mips
                dur += task.bw_length/self.bw
                return dur
            raise Exception('超出资源需求的任务')
            return False

    def finishTask(self, task):
        if task in self.task_list:
            self.task_list.remove(task)
            self.c_mips -= task.mips
            self.c_bw -= task.bw
            task.finish_time = self.env.clock
            task.mc_price = self.price
            self.over_task_list.append(task)
            self.state = Parameters.VMState.STOP
            self.task = None
            return True
        if task in self.over_task_list:
            raise Exception('完成2次的任务？')
        raise Exception('完成不存在的任务？')
        return False

    def taskExpend(self, task):
        return self.tryExcuteTask(task) * self.price

    @property
    def price(self):
        if self._price == None:
            x_mips = self.mips
            x_bw = self.bw
            price = None
            if x_bw < Parameters.stair_bw[0]:
                price = x_bw * Parameters.bw_price[0] + x_mips * Parameters.mips_price
            else:
                price = Parameters.stair_bw[0] * Parameters.bw_price[0] + (x_bw - Parameters.stair_bw[0]) * Parameters.bw_price[1] + x_mips * Parameters.mips_price
            self._price = price
        return self._price

    @property
    def busy_expend(self):
        return self.price * self.busy_time

    @property
    def task_list_request(self):
        total_request ={}
        for s in Parameters.BaseSourceLength:
            total_request[s] = []
        for task in self.task_list:
            for s in Parameters.BaseSourceLength:
                total_request[s].append(task.need_source[s])
        if len(self.task_list) == 0:
            for s in Parameters.BaseSourceLength:
                total_request[s] = [0]
                raise Exception('空空')
        return total_request

    @property
    def busy_time(self):
        if len(self.task_list) == 0:
            return 0
        pass_time = self.env.clock - self.task_list[0].start_time
        busy_time = -pass_time
        all_request = self.task_list_request
        for s,sl in zip(Parameters.BaseSource,Parameters.BaseSourceLength):
            busy_time += (np.array(all_request[sl]).sum()/self.source[s])
        return busy_time


    @property
    def is_run(self):
        return self.state == Parameters.VMState.RUN

    @property
    def a_source(self):
        return {
            'mips': self.a_mips,
            'bw': self.a_bw,
        }

    @property
    def source(self):
        return {
            'mips': self.mips,
            'bw': self.bw,
            'price': self.price,
        }

    @property
    def a_mips(self):
        return self._mips - self.c_mips

    @property
    def a_bw(self):
        return self._bw - self.c_bw

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value
        return value

    @property
    def mips(self):
        return self._mips

    # @property
    # def number_pes(self):
    #     return self._numberOfPes
    #
    # @property
    # def ram(self):
    #     return self._ram

    @property
    def bw(self):
        return self._bw
