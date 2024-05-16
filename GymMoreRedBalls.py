import random
from minigrid.envs.babyai import goto
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel, BabyAIMissionSpace
from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.verifier import (
    Instr,
    ActionInstr,
    AfterInstr,
    AndInstr,
    BeforeInstr,
    PutNextInstr,
    SeqInstr,
)
from minigrid.core.roomgrid import RoomGrid
from minigrid.wrappers import FullyObsWrapper
from abc import ABC, abstractmethod
import numpy as np


class SeqInOrderInstr(Instr, ABC):
    """
    Base class for sequencing instructions (before, after, and)
    """

    def __init__(self, instr_s, strict=False):
        for instr in instr_s:
            assert isinstance(instr, ActionInstr) or isinstance(instr, AndInstr)
        self.instr_s = instr_s
        self.strict = strict
        self.reward = 0
class InOrderInstr(SeqInOrderInstr):
    """
    Sequence two instructions in order:
    eg: pickup red ball 1, pickup red ball 2, pickup red ball 3
    """
    def surface(self, env):
        text = ""

        for i, instr in enumerate(self.instr_s):
            if i == len(self.instr_s) - 1:
                text += instr.surface(env)
            else:
                text += instr.surface(env) + ", then "

        return text

    def reset_verifier(self, env):
        super().reset_verifier(env)
        for instr in self.instr_s:
            instr.reset_verifier(env)
        self.s_done = [False for _ in self.instr_s]

    def verify(self, action, i=0):
        # if all instructions are sequentially done, return "success". if not, "failure"

        if self.s_done[i] == "success":
            if i == len(self.instr_s) - 1:
                return "success"
            else:
                return self.verify(action, i+1)
        else:
            self.s_done[i] = self.instr_s[i].verify(action)
            if self.s_done[i] == "failure":

                return "failure"

            if self.s_done[i] == "success":

                #self.reward += 0.3
                #print("mid_success")
                #print(self.reward)
                return self.verify(action)


        return "continue"

        # if self.a_done == "success":
        #     self.b_done = self.instr_b.verify(action)
        #
        #     if self.b_done == "failure":
        #         return "failure"
        #
        #     if self.b_done == "success":
        #         return "success"
        # else:
        #     self.a_done = self.instr_a.verify(action)
        #     if self.a_done == "failure":
        #         return "failure"
        #
        #     if self.a_done == "success":
        #         return self.verify(action)
        #
        #     # In strict mode, completing b first means failure
        #     if self.strict:
        #         if self.instr_b.verify(action) == "success":
        #             return "failure"
        #
        # return "continue"

class GymMoreRedBalls(RoomGridLevel):
    """
    ## Description

    Go to the red ball. No distractors present.
    [ADDED] but you should go to the closest red ball.

    ## Mission Space

    "go to the red ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |
    """

    def __init__(self, room_size=10,
        num_rows=3,
        num_cols=3,
        num_dists=0, num_objs=3,
        action_kinds=["goto"],
        instr_kinds=["inoder_seq"], # or ["inorder_seq"],
        **kwargs,
        ):
        self.num_dists = num_dists
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds
        self.locked_room = None

        self.num_dists = num_dists
        self.num_objs = num_objs
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)


    def gen_mission(self):
        self.place_agent()

        colors = ['red', 'green', 'blue']

        self.objs = []
        for i, c in enumerate(colors):
            obj, _ = self.add_object(0, 0, "ball", c)
            self.objs.append(obj)
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        # TODO 이 부분을 inorder 로 해야함
        self.instrs = self.gen_instr_inorder()

    def gen_instr_inorder(self):

        instr_red = GoToInstr(ObjDesc(self.objs[0].type, self.objs[0].color))
        instr_green = GoToInstr(ObjDesc(self.objs[1].type, self.objs[1].color))
        instr_blue = GoToInstr(ObjDesc(self.objs[2].type, self.objs[2].color))
        #빨,초,파 순서
        instr_ = InOrderInstr([instr_red, instr_green, instr_blue])
        #instr_ = InOrderInstr([instr_red, instr_blue, instr_green])
        instr_list = [instr_red,instr_green,instr_blue]
        #random.shuffle(instr_list)  # 무작위로 지시사항을 섞음

        #instr_ = InOrderInstr(instr_list)
        return instr_
    def validate_instrs(self, instr):
        """
        Perform some validation on the generated instructions
        """
        # Gather the colors of locked doors
        colors_of_locked_doors = []
        if hasattr(self, "unblocking") and self.unblocking:
            for i in range(self.num_cols):
                for j in range(self.num_rows):
                    room = self.get_room(i, j)
                    for door in room.doors:
                        if door and door.is_locked:
                            colors_of_locked_doors.append(door.color)

        if isinstance(instr, PutNextInstr):
            # Resolve the objects referenced by the instruction
            instr.reset_verifier(self)

            # Check that the objects are not already next to each other
            if set(instr.desc_move.obj_set).intersection(set(instr.desc_fixed.obj_set)):
                raise RejectSampling(
                    "there are objects that match both lhs and rhs of PutNext"
                )
            if instr.objs_next():
                raise RejectSampling("objs already next to each other")

            # Check that we are not asking to move an object next to itself
            move = instr.desc_move
            fixed = instr.desc_fixed
            if len(move.obj_set) == 1 and len(fixed.obj_set) == 1:
                if move.obj_set[0] is fixed.obj_set[0]:
                    raise RejectSampling("cannot move an object next to itself")

        if isinstance(instr, ActionInstr):
            if not hasattr(self, "unblocking") or not self.unblocking:
                return
            # TODO: either relax this a bit or make the bot handle this super corner-y scenarios
            # Check that the instruction doesn't involve a key that matches the color of a locked door
            potential_objects = ("desc", "desc_move", "desc_fixed")
            for attr in potential_objects:
                if hasattr(instr, attr):
                    obj = getattr(instr, attr)
                    if obj.type == "key" and obj.color in colors_of_locked_doors:
                        raise RejectSampling(
                            "cannot do anything with/to a key that can be used to open a door"
                        )
            return

        if isinstance(instr, SeqInstr):
            self.validate_instrs(instr.instr_a)
            self.validate_instrs(instr.instr_b)
            return

        if isinstance(instr, InOrderInstr):
            for inst in instr.instr_s:
                self.validate_instrs(inst)
            return

        assert False, "unhandled instruction type"
    def num_navs_needed(self, instr) -> int:
        """
        Compute the maximum number of navigations needed to perform
        a simple or complex instruction
        """

        if isinstance(instr, PutNextInstr):
            return 2

        elif isinstance(instr, ActionInstr):
            return 1

        elif isinstance(instr, SeqInstr):
            na = self.num_navs_needed(instr.instr_a)
            nb = self.num_navs_needed(instr.instr_b)
            return na + nb

        elif isinstance(instr, InOrderInstr):
            n = 0
            for inst in instr.instr_s:
                n += self.num_navs_needed(inst)
            return n

        else:
            raise NotImplementedError(
                "instr needs to be an instance of PutNextInstr, ActionInstr, or SeqInstr"
            )

    def step(self, action):
        obs_all, reward, terminated, truncated, info = super().step(action)
        count = 0
        count +=1
        obs = obs_all['image'][:,:,0]

        # If we drop an object, we need to update its position in the environment
        if action == self.actions.drop:
            self.update_objs_poss()

        # If we've successfully completed the mission
        #status,temp_reward = self.instrs.verify(action)
        #print(temp_reward, reward)
        #reward += temp_reward
        #print(status)
        status = self.instrs.verify(action)
        if status == "success":
            terminated = True
            reward = self._reward()

        elif status == "failure":
            terminated = True
            reward = 0


        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    seed = 123
    env = GymMoreRedBalls(room_size=10,render_mode='human')
    env.reset(seed=123)
    env.render()
    """F
    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused    
    """
    # when room_size = 10, and seed=123
    #일단 주워보자
    Oracle_action = [2, 2, 1, 0, 0,
                         2, 2, 2, 2, 2,
                         1, 2, 0, 1,
                     2, 2, 2, 1, 2, 2, 2, 2]


#줍는 경우
#    [2, 2, 1, 3, 0, 0,
#     2, 2, 2, 2, 2,
#     1, 2, 0, 3, 1, 2, 2,
#     2, 1, 2, 2, 2, 2, 3]








    Wrong_action = [2,2,2,2,2,
                    1,1,2,2,2,0,0,2,0,2,2,2,2,2,2,2,2,]

    seed = 123
    print(env.mission)
    for i in range(len(Oracle_action)):

        action = Oracle_action[i]
            #action = Wrong_action[i]

        print("action:", action) # 어떤 행동을 하는지
        obs, reward, done, truncated, info = env.step(action)
        print("obs:", obs) # 관측한 observation 프린트
        print("env.instr.s_done:", env.instrs.s_done)
        # 순서대로 빨강, 초록, 파랑 공을 방문했는지 여부.
        # 순서대로 방문하지 않으면 success가 되지 않음.
        # 최종 reward 는 맨 마지막에 도달해야지만 얻기 때문에 이런 정보를 활용해서 reward를 추가로 줄 수도 있음.

        print(f"step={i}, action={action}, reward={reward}")
        env.render()
        if done:
            break