"""
Levels described in the ICLR 2019 submission.
"""

import gym
from .verifier import *
from .levelgen import *


class Level_GoToRedBallGrey(RoomGridLevel):
    """
    Go to the red ball, single room, with obstacles.
    The obstacles/distractors are all grey boxes, to eliminate
    perceptual complexity. No unblocking required.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')

        for i in range(self.num_dists):
            self.add_object(0, 0, 'box', 'grey')

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBall(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBallNoDists(Level_GoToRedBall):
    """
    Go to the red ball. No distractors present.
    """

    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=0, seed=seed)


class Level_GoToObj(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=1)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToObjS4(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=4, seed=seed)


class Level_GoToObjS6(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=6, seed=seed)


class Level_GoToLocal(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, num_dists=8, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocalAdaptation(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, num_dists=8, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        agent_start_pos = self.place_agent()
        # add goal obj here 
        objs = self.add_distractors(num_distractors=1, all_unique=False)
        obj = self._rand_elem(objs)

        goal_obj_pos = obj.cur_pos
        dx = goal_obj_pos[0] - agent_start_pos[0]
        dy = goal_obj_pos[1] - agent_start_pos[1]
        #print(f"goal_obj_pos {goal_obj_pos}, agent_start_pos {agent_start_pos}, ({dx},{dy})")
        if (abs(dx) >= 2):
            new_obj_dx = random.randint(dx+1, -1) if dx < 0 else random.randint(1, dx-1)
            #print(f"new obj ({new_obj_dx+agent_start_pos[0]},{agent_start_pos[1]})")
            self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=agent_start_pos[1], type='lava')
        if (abs(dy) >= 2):
            new_obj_dy = random.randint(dy+1, -1) if dy < 0 else random.randint(1, dy-1)
            #print(f"new obj ({agent_start_pos[0]},{new_obj_dy+agent_start_pos[1]})")
            self.user_defined_add_distractors(i=agent_start_pos[0],j=new_obj_dy+agent_start_pos[1], type='lava')
        #print(f"num objs {objs}, agent_start_pos {agent_start_pos}, obj pos {obj.cur_pos}")
        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocalAdaptationMedian(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=8, num_dists=8, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        agent_start_pos = self.place_agent()
        # add goal obj here 
        objs = self.add_distractors(num_distractors=1, all_unique=False)
        obj = self._rand_elem(objs)

        goal_obj_pos = obj.cur_pos
        dx = goal_obj_pos[0] - agent_start_pos[0]
        dy = goal_obj_pos[1] - agent_start_pos[1]
        #print(f"goal_obj_pos {goal_obj_pos}, agent_start_pos {agent_start_pos}, ({dx},{dy})")
        if (abs(dx) >= 2):
            new_obj_dx = random.randint(dx+1, -1) if dx < 0 else random.randint(1, dx-1)
            #print(f"new obj ({new_obj_dx+agent_start_pos[0]},{agent_start_pos[1]})")
            self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=agent_start_pos[1], type='lava')
            if (agent_start_pos[1]-1 >= 1):
                self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=agent_start_pos[1]-1, type='lava')
            #if (agent_start_pos[1]-2 >= 1):
            #    self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=agent_start_pos[1]-2, type='lava')
            if (agent_start_pos[1]+1 <= 6):
                self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=agent_start_pos[1]+1, type='lava')
            #if (agent_start_pos[1]+2 <= 6):
            #    self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=agent_start_pos[1]+2, type='lava')
        if (abs(dy) >= 2):
            new_obj_dy = random.randint(dy+1, -1) if dy < 0 else random.randint(1, dy-1)
            #print(f"new obj ({agent_start_pos[0]},{new_obj_dy+agent_start_pos[1]})")
            self.user_defined_add_distractors(i=agent_start_pos[0],j=new_obj_dy+agent_start_pos[1], type='lava')
            if (agent_start_pos[0]-1 >= 1):
                self.user_defined_add_distractors(i=agent_start_pos[0]-1,j=new_obj_dy+agent_start_pos[1], type='lava')
            #if (agent_start_pos[0]-2 >= 1):
            #    self.user_defined_add_distractors(i=agent_start_pos[0]-2,j=new_obj_dy+agent_start_pos[1], type='lava')
            if (agent_start_pos[0]+1 <= 6):
                self.user_defined_add_distractors(i=agent_start_pos[0]+1,j=new_obj_dy+agent_start_pos[1], type='lava')
            #if (agent_start_pos[0]+2 <= 6):
            #    self.user_defined_add_distractors(i=agent_start_pos[0]+2,j=new_obj_dy+agent_start_pos[1], type='lava')
        #print(f"num objs {objs}, agent_start_pos {agent_start_pos}, obj pos {obj.cur_pos}")
        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocalComposition(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """

    def __init__(self, room_size=10, num_dists=8, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        # add goal obj here 
        agent_start_pos = self.place_agent()
        agent_x, agent_y = agent_start_pos[0], agent_start_pos[1]
        goal_x = agent_x + 3 if agent_x <= 5 else agent_x - 3
        goal_y = agent_y + 3 if agent_y <= 5 else agent_y - 3

        goal_obj = self.user_defined_add_distractors(i=goal_x,j=goal_y, type='key')
        self.instrs = GoToInstr(ObjDesc(goal_obj.type, goal_obj.color))

        goal_obj_pos = goal_obj.cur_pos
        self.goal_obj_pos = goal_obj_pos
        dx = goal_obj_pos[0] - agent_start_pos[0]
        dy = goal_obj_pos[1] - agent_start_pos[1]
        assert (abs(dx) >= 3 and abs(dy) >= 3)
        #print(f"agent_start_pos {agent_start_pos}, goal_obj_pos {goal_obj_pos}")
        #print(f"goal_obj_pos {goal_obj_pos}, agent_start_pos {agent_start_pos}, ({dx},{dy})")
        new_obj_dx = random.randint(dx+1, -1) if dx < 0 else random.randint(1, dx-1)
        new_obj_dy = random.randint(dy+1, -1) if dy < 0 else random.randint(1, dy-1)
        #print(f"new obj ({new_obj_dx+agent_start_pos[0]},{agent_start_pos[1]})")
        self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=agent_start_pos[1], type='lava')
        self.user_defined_add_distractors(i=agent_start_pos[0],j=new_obj_dy+agent_start_pos[1], type='lava')
        #print(f"new obj ({agent_start_pos[0]},{new_obj_dy+agent_start_pos[1]})")
        
        self.user_defined_add_distractors(i=new_obj_dx+agent_start_pos[0],j=goal_obj_pos[1], type='lava')
        self.user_defined_add_distractors(i=goal_obj_pos[0],j=new_obj_dy+agent_start_pos[1], type='lava')
        #print(f"num objs {objs}, agent_start_pos {agent_start_pos}, obj pos {obj.cur_pos}")
        self.check_objs_reachable()
        self.instrs = GoToInstr(ObjDesc(goal_obj.type, goal_obj.color))

    def get_goal_pos(self):
        return list(self.goal_obj_pos)

class Level_GoToLocalS5N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_dists=2, seed=seed)


class Level_GoToLocalS6N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=2, seed=seed)


class Level_GoToLocalS6N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=3, seed=seed)


class Level_GoToLocalS6N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=4, seed=seed)


class Level_GoToLocalS7N1(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=1, seed=seed)

class Level_GoToLocalS7N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=2, seed=seed)

class Level_GoToLocalS7N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=4, seed=seed)


class Level_GoToLocalS7N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=5, seed=seed)

class Level_GoToLocalS8N1(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=1, seed=seed)

class Level_GoToLocalS8N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=2, seed=seed)


class Level_GoToLocalS8N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=3, seed=seed)


class Level_GoToLocalS8N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=4, seed=seed)


class Level_GoToLocalS8N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=5, seed=seed)


class Level_GoToLocalS8N6(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=6, seed=seed)


class Level_GoToLocalS8N7(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=7, seed=seed)

class Level_GoToLocalS8N15(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=15, seed=seed)


class Level_GoToLocalS10N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=10, num_dists=3, seed=seed)

class Level_GoToLocalS10N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=10, num_dists=5, seed=seed)

class Level_GoToLocalS10N25(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=10, num_dists=25, seed=seed)

class Level_PutNextLocal(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """

    def __init__(self, room_size=8, num_objs=8, seed=None):
        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PutNextLocalS5N3(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_objs=3, seed=seed)


class Level_PutNextLocalS6N4(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_objs=4, seed=seed)


class Level_GoTo(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=False,
        seed=None
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class Level_GoToObjMazeS4AdaptationBoss(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, doors_open=True, seed=seed)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        lavas = self.add_distractors(num_distractors=4, all_unique=False, all_lava=True)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()



class Level_GoToOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(doors_open=True, seed=seed)


class Level_GoToObjMaze(Level_GoTo):
    """
    Go to an object, the object may be in another room. No distractors.
    """

    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=False, seed=seed)


class Level_GoToObjMazeOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=True, seed=seed)


class Level_GoToObjMazeS4R2(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, num_rows=2, num_cols=2, doors_open=True, seed=seed)

class Level_GoToObjMazeS4R2Obs(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=3, room_size=4, num_rows=2, num_cols=2, doors_open=True, seed=seed)


class Level_GoToObjMazeS4R2Close(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, num_rows=2, num_cols=2, doors_open=False, seed=seed)

class Level_GoToObjMazeS4(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, doors_open=True, seed=seed)

class Level_GoToObjMazeS4Obs(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=5, room_size=4, doors_open=True, seed=seed)

class Level_GoToObjMazeS4Close(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, doors_open=False, seed=seed)

class Level_GoToObjMazeS5(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=5, doors_open=True, seed=seed)


class Level_GoToObjMazeS6(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=6, doors_open=True, seed=seed)


class Level_GoToObjMazeS7(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=7, doors_open=True, seed=seed)


class Level_GoToImpUnlock(RoomGridLevel):
    """
    Go to an object, which may be in a locked room.
    Competencies: Maze, GoTo, ImpUnlock
    No unblocking.
    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_rows)
        jd = self._rand_int(0, self.num_cols)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i is not id or j is not jd:
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=2,
                        all_unique=False
                    )

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.start_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        # Add a single object to the locked room
        # The instruction requires going to an object matching that description
        obj, = self.add_distractors(id, jd, num_distractors=1, all_unique=False)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_Pickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnblockPickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room. The path may
    be blocked by one or more obstructors.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=20, all_unique=False)

        # Ensure that at least one object is not reachable without unblocking
        # Note: the selected object will still be reachable most of the time
        if self.check_objs_reachable(raise_exc=False):
            raise RejectSampling('all objects reachable')

        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_Open(RoomGridLevel):
    """
    Open a door, which may be in another room
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        # Collect a list of all the doors in the environment
        doors = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)

        door = self._rand_elem(doors)
        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_Unlock(RoomGridLevel):
    """
    Unlock a door.

    Competencies: Maze, Open, Unlock. No unblocking.
    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_rows)
        jd = self._rand_int(0, self.num_cols)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_rows)
            jk = self._rand_int(0, self.num_cols)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, 'key', door.color)
            break

        # With 50% probability, ensure that the locked door is the only
        # door of that color
        if self._rand_bool():
            colors = list(filter(lambda c: c is not door.color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if i is not id or j is not jd:
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=3,
                        all_unique=False
                    )

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.start_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_PutNext(RoomGridLevel):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)
        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )

'''
class Level_PickupS4R2(Level_GoTo):

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        
        super().__init__(
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False
        )
        
        super().__init__(action_kinds=['pickup'],instr_kinds=['action'],num_dists=1, room_size=4, num_rows=2, num_cols=2, doors_open=True, seed=seed)
'''


class Level_PickupLoc(LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False
        )

class Level_PickupLocLarge(LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            room_size=10,
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False
        )


class Level_PickupLocMaze(LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=['pickup'],
            instr_kinds=['action'],
            room_size=4,
            num_rows=3,
            num_cols=3,
            num_dists=1,
            locked_room_prob=0,
            locations=True,
            unblocking=False
        )

class Level_GoToSeq(LevelGen):
    """
    Sequencing of go-to-object commands.

    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None
    ):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto'],
            locked_room_prob=0,
            locations=False,
            unblocking=False
        )


class Level_GoToSeqS5R2(Level_GoToSeq):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_rows=2, num_cols=2, num_dists=4, seed=seed)


class Level_Synth(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            instr_kinds=['action'],
            locations=False,
            unblocking=True,
            implicit_unlock=False
        )


class Level_SynthS5R2(Level_Synth):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=2,
            num_cols=2,
            num_dists=7,
            seed=seed
        )


class Level_SynthLoc(LevelGen):
    """
    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            instr_kinds=['action'],
            locations=True,
            unblocking=True,
            implicit_unlock=False
        )


class Level_SynthSeq(LevelGen):
    """
    Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.
    No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq
    """

    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            locations=True,
            unblocking=True,
            implicit_unlock=False
        )


class Level_MiniBossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25
        )


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )


class Level_BossLevelNoUnlock(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            locked_room_prob=0,
            implicit_unlock=False
        )


# Register the levels in this file
register_levels(__name__, globals())
