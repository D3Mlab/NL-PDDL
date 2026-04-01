
class ALFWorldTextAction:
    def __init__(self, action_type,  obj, recep=None):
        self.action_type = action_type
        self.obj = obj
        self.recep = recep
        self.successful = None
    

class EnvInteractAction(ALFWorldTextAction):
    def __init__(self, action_type,  obj, recep=None):
        super().__init__(action_type, obj, recep)
        
    def gen_action_str_list(self, agent_location):
        action_str_list = []
        if self.action_type == 'take':
            action_str = "take {} from {}".format(self.obj.thor_name, self.recep.thor_name)
            action_str_list.append(action_str)
            if agent_location != self.obj.location:
                goto_str = "go to {}".format(self.recep.thor_location)
                action_str_list.insert(0, goto_str)
        elif self.action_type == 'put':
            action_str = "put {} in/on {}".format(self.obj.thor_name, self.recep.thor_name)
            action_str_list.append(action_str)
            
            if agent_location != self.recep.location:
                goto_str = "go to {}".format(self.recep.thor_location)
                action_str_list.insert(0, goto_str)

        elif self.action_type == 'heat':
            if agent_location != self.recep.location:
                goto_str = "go to {}".format(self.recep.thor_location)
                action_str_list.append(goto_str)
            action_str = "heat {} with {}".format(self.obj.thor_name, self.recep.thor_name)
            action_str_list.append(action_str)
        elif self.action_type == 'cool':
            if agent_location != self.recep.location:
                goto_str = "go to {}".format(self.recep.thor_location)
                action_str_list.append(goto_str)
            action_str = "cool {} with {}".format(self.obj.thor_name, self.recep.thor_name)
            action_str_list.append(action_str)
        elif self.action_type == 'clean':
            if agent_location != self.recep.location:
                goto_str = "go to {}".format(self.recep.thor_location)
                action_str_list.append(goto_str)
            action_str = "clean {} with {}".format(self.obj.thor_name, self.recep.thor_name)
            action_str_list.append(action_str)
        elif self.action_type == 'light':
            if agent_location != self.recep.location:
                goto_str = "go to {}".format(self.recep.thor_location)
                action_str_list.append(goto_str)
            action_str = "use {}".format(self.recep.thor_name)
            action_str_list.append(action_str)

        else:
            raise NotImplementedError(self.action_type, " not implemented in env action!")


        return action_str_list
    
    
    def __eq__(self, other):
        return self.action_type == other.action_type and self.obj == other.obj and self.recep == other.recep

    def __hash__(self):
        return hash((self.action_type, self.obj, self.recep))
    
    def __repr__(self):
        action_str = None
        if self.action_type == 'take':
            action_str = "take {} from {}".format(self.obj.thor_name, self.recep.thor_name)
        elif self.action_type == 'put':
            action_str = "put {} in/on {}".format(self.obj.thor_name, self.recep.thor_name)
        elif self.action_type == 'heat':
            action_str = "heat {} with {}".format(self.obj.thor_name, self.recep.thor_name)
        elif self.action_type == 'cool':
            action_str = "cool {} with {}".format(self.obj.thor_name, self.recep.thor_name)
        elif self.action_type == 'clean':
            action_str = "clean {} with {}".format(self.obj.thor_name, self.recep.thor_name)
        elif self.action_type == 'light':
            action_str = "toggle {}".format(self.recep.thor_name)
        else:
            raise NotImplementedError(self.action_type, " not implemented in env action!")
        return action_str

class EnvExploreAction(ALFWorldTextAction):
    """Explore actions: either moving to or opening an object."""
    SUPPORTED = {'go to', 'open'}

    def __init__(self, action_type, obj, recep=None):
        if action_type not in self.SUPPORTED:
            raise NotImplementedError(f"{action_type} not supported")
        super().__init__(action_type, obj, recep)

    def gen_action_str_list(self, agent_location):
        """
        Return a list of one or two commands:
          - always a 'go to' if not already there,
          - then, if this is an 'open', the 'open' command.
        Also mutates obj_state on open.
        """
        cmds = []
        # 1) maybe move
        goto_cmd = self._goto_command(agent_location)
        if goto_cmd:
            cmds.append(goto_cmd)

        # 2) maybe open
        if self.action_type == 'open':
            cmds.append(self._base_command())
            self.obj.obj_state = 'open'

        return cmds

    def _goto_command(self, agent_location):
        """Return a 'go to X' command if needed, else None."""
        if agent_location != self.obj.location:
            return f"go to {self.obj.thor_location}"
        return None

    def _base_command(self):
        """The single-step command for this action_type."""
        if self.action_type == 'go to':
            return f"go to {self.obj.thor_location}"
        # must be 'open' here
        return f"open {self.obj.thor_name}"

    def __eq__(self, other):
        return (
            isinstance(other, EnvExploreAction) and
            self.action_type == other.action_type and
            self.obj == other.obj
        )

    def __hash__(self):
        return hash((self.action_type, self.obj))

    def __repr__(self):
        # just show the base command, same as the single-step action
        return self._base_command()

# class EnvExploreAction(ALFWorldTextAction):
#     def __init__(self, action_type,  obj, recep=None):
#         super().__init__(action_type, obj, recep)

    
#     def gen_action_str_list(self, agent_location):
#         action_str_list = []
#         if self.action_type == 'go to':
#             if agent_location != self.obj.location:
#                 goto_str = "go to {}".format(self.obj.thor_location)
#                 action_str_list.append(goto_str)
#         elif self.action_type == 'open':
#             if agent_location != self.obj.location:
#                 goto_str = "go to {}".format(self.obj.thor_location)
#                 action_str_list.append(goto_str)
#             action_str = "open {}".format(self.obj.thor_name)
#             action_str_list.append(action_str)
#             self.obj.obj_state = 'open'
#         else:
#             raise NotImplementedError(self.action_type, " not implemented in explore action!")
#         return action_str_list

#     def __eq__(self, other):
#         return self.action_type == other.action_type and self.obj == other.obj
    
#     def __hash__(self):
#         return hash((self.action_type, self.obj))
    
#     def __repr__(self):
#         action_str = None
#         if self.action_type == 'go to':
#             action_str = "go to {}".format(self.obj.thor_location)
#         elif self.action_type == 'open':
#             action_str = "open {}".format(self.obj.thor_name)
#         else:
#             raise NotImplementedError(self.action_type, " not implemented in explore action!")
#         return action_str