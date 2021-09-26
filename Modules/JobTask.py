
class TaskFile(object):
    def __init__(self, name, file_length, type):
        self.name = name
        self.file_length = file_length
        self.type =type

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.name == other.name
                and self.file_length == other.file_length )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def file_length(self):
        return self._file_length

    @file_length.setter
    def file_length(self,value):
        self._file_length=value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value





class Task(object):
    parent_list = None
    child_list = None
    file_list = None
    # task_finish_time = None
    task_id = None
    task_name = ''
    task_length = None
    runtime = None

    load_time = None
    start_time = None
    finish_time = None

    mips = None
    bw = None
    duration = None
    mips_length = None
    bw_length = None
    # vm_id = None
    # number_pes = None
    def __init__(self):
        self.parent_list = []
        self.child_list = []
        self.file_list = []
        # self.task_finish_time = None
        self.task_id = None
        self.task_name = ''
        self.task_length = None
        self.runtime = None
        self.load_time = None
        self.start_time = None
        self.finish_time = None
        # self.vm_id = None
        # self.number_pes = 1
        self.job = None

        self.mips = None
        self.bw = None
        self.duration = None
        self.mips_length = None
        self.bw_length = None
        self.mc_price = None

    def check_parent_over(self):
        for parent_key in self.parent_list:
            if self.job.task_map[parent_key].finish_time == None:
                return False
        return True

    def ex_source(self, clock):
        ex_time = clock - self.load_time
        ex_source ={
            'ex_mips': 0,
            'ex_bw': 0,
        }
        if ex_time > 0:
            mips_time = self.mips_length / self.mips
            bw_time = self.bw_length / self.bw
            remain_time = self.duration - ex_time
            mips_time = remain_time * mips_time / (mips_time+bw_time)
            bw_time = remain_time * bw_time / (mips_time+bw_time)
            ex_mips = int(self.mips_length/mips_time) + 1 - self.mips
            ex_bw = int(self.bw_length/bw_time) + 1 - self.bw
            ex_source['ex_mips'] = ex_mips
            ex_source['ex_bw'] = ex_bw
        return ex_source

    def ex_need_source(self, clock):
        ex_source = self.ex_source(clock)
        return {
            'mips': self.mips + ex_source['ex_mips'],
            'bw': self.bw + ex_source['ex_bw'],
        }

    @property
    def expend(self):
        return self.mc_price * (self.finish_time - self.start_time)

    @property
    def need_source(self):
        return {
            'mips': self.mips,
            'bw': self.bw,
            'mips_length': self.mips_length,
            'bw_length': self.bw_length,
        }

    @property
    def out_file_length(self):
        all_size = 0
        for f in self.file_list:
            if f.type == 'output':
                all_size += f.file_length
        return all_size
        pass

    @property
    def out_file_num(self):
        all_num = 0
        for f in self.file_list:
            if f.type == 'output':
                all_num += 1
        return all_num

class Job(object):
    task_map = None
    def __init__(self):
        self.task_map = {} # id - task
        self.wait_task_list = []
        self.available_task_list = []
        self.ran_task_list = []

    def arrange(self):
        for task in self.task_map.values():
            if task in self.ran_task_list:
                continue
            if task.check_parent_over():
                self.available_task_list.append(task)
            else:
                self.wait_task_list.append(task)

    def updata_available(self):
        for task in self.wait_task_list:
            if task.check_parent_over():
                self.available_task_list.append(task)
                self.wait_task_list.remove(task)

import xml.etree.ElementTree as ET
from Base.Parameters import Parameters
class Dax2Job(object):
    @staticmethod
    def pares(xml_tree, job_obj: Job):
        root = xml_tree.getroot()
        for task in root.iter('job'):
            tmp_task = Task()
            tmp_task.runtime = float(task.attrib['runtime'])
            tmp_task.task_id = task.attrib['id']
            tmp_task.task_length = tmp_task.runtime * Parameters.RuntimeScale
            for uses in task.iter('uses'):
                tmp_file = TaskFile(name=uses.attrib['file'],
                                    file_length=int(uses.attrib['size']) / Parameters.FileLengthScale,
                                    type=uses.attrib['link'])
                tmp_task.file_list.append(tmp_file)
            job_obj.task_map[tmp_task.task_id] = tmp_task
            tmp_task.job = job_obj

        for ship in root.iter('child'):
            tmp_child_id = ship.attrib['ref']
            for parent in ship.iter('parent'):
                tmp_parent_id = parent.attrib['ref']
                job_obj.task_map[tmp_child_id].parent_list.append(tmp_parent_id)
                job_obj.task_map[tmp_parent_id].child_list.append(tmp_child_id)
        pass


    pass

import os.path

if __name__ == '__main__':
    source_path = 'E:/Storage/Code/DjG/Assisant/Source/RL'
    file_path = 'Cloud/dax/HEFT_paper.xml'
    xml_tree = ET.parse(os.path.join(source_path,file_path))
    job = Job()
    Dax2Job.pares(xml_tree, job)

    asd = TaskFile()
    print(asd.name)
