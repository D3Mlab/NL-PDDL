import re

class ALFWorldTextObsParser:

    def __init__(self, obj_pattern=r'\b\w+\s\d+\b'):
        self.obj_pattern = obj_pattern

    def extract_objects_text(self, obs, location):

        obs = obs.lower().strip()
        obj_pattern = self.obj_pattern
        obj_list = []

        if obs.startswith('you are in the middle of a room') or \
           obs.startswith('you arrive at loc') or\
           obs.startswith('you open') or \
           'is empty' in obs or \
           obs.strip() == "":
            # raw_objs = re.findall(obj_pattern, obs)
            # objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            return obj_list
        elif 'looking quickly around you' in obs:
            obs = ','.join(obs.split(',')[1:])
            raw_objs = re.findall(obj_pattern, obs)
            objs = [("_".join(i.split(' ')),'receptacle', None, "_".join(i.split(' '))) for i in raw_objs if 'loc ' not in i]
            obj_list = obj_list + objs
        elif "is open" in obs:
            raw_objs = re.findall(obj_pattern, obs)
            objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            obj_list.append((objs[0], 'receptacle', 'open', objs[0]))
        elif "is closed" in obs:
            raw_objs = re.findall(obj_pattern, obs)
            objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            obj_list.append((objs[0], 'receptacle', 'close', objs[0]))
        elif obs.startswith('in it'):
            obs = ','.join(obs.split(',')[1:])
            raw_objs = re.findall(obj_pattern, obs)
            objs = [("_".join(i.split(' ')),'object', 'inside', location) for i in raw_objs if 'loc ' not in i]
            obj_list = obj_list + objs
        elif obs.startswith('on the') or obs.startswith('on it'):
            obs = ','.join(obs.split(',')[1:])
            raw_objs = re.findall(obj_pattern, obs)
            objs = [("_".join(i.split(' ')),'object', 'outside', location) for i in raw_objs if 'loc ' not in i]
            obj_list = obj_list + objs
        else:
            raise NotImplementedError(str(obs) + ' is unseen, check text extraction')

        return obj_list
    
    def parse_text_observation(self, obs_str):
        obj_pattern = self.obj_pattern
        all_objects = []
        location = None

        sentences = obs_str.split('.')

        obs_str = obs_str.lower()
        
        if  obs_str.startswith('you are in the middle of a room'):
            location = None
        elif obs_str.startswith('you arrive at') or obs_str.startswith('you open') :
            raw_objs = re.findall(obj_pattern, obs_str)
            objs = ["_".join(i.split(' ')) for i in raw_objs if 'loc ' not in i]
            location = objs[0]

        for s in sentences:
            objs = self.extract_objects_text(s, location)
            all_objects = all_objects + objs
        return all_objects

    