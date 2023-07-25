# Copyright 2023 Technion project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main
import up_multi_goal_split
from up_multi_goal_split.multi_goal_split import *
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.engines import PlanGenerationResultStatus
from collections import namedtuple
import random
import os



def test_wcd():
    problem = Problem()
    
    loc = UserType("loc")

    # Environment     
    connected = Fluent('connected', BoolType(), l1=loc, l2=loc)        
    problem.add_fluent(connected, default_initial_value=False)

    free = Fluent('free', BoolType(), l=loc)
    problem.add_fluent(free, default_initial_value=True)

    nw, ne, sw, se = Object("nw", loc), Object("ne", loc), Object("sw", loc), Object("se", loc)        
    problem.add_objects([nw, ne, sw, se])
    problem.set_initial_value(connected(nw, ne), True)
    problem.set_initial_value(connected(nw, sw), True)
    problem.set_initial_value(connected(ne, nw), True)
    problem.set_initial_value(connected(ne, se), True)
    problem.set_initial_value(connected(sw, se), True)
    problem.set_initial_value(connected(sw, nw), True)
    problem.set_initial_value(connected(se, sw), True)
    problem.set_initial_value(connected(se, ne), True)


    at = Fluent('at', BoolType(), l1=loc)
    problem.add_fluent(at, default_initial_value=False)

    move = InstantaneousAction('move', l1=loc, l2=loc)
    l1 = move.parameter('l1')
    l2 = move.parameter('l2')
    move.add_precondition(at(l1))
    move.add_precondition(free(l2))
    move.add_precondition(connected(l1,l2))
    move.add_effect(at(l2),True)
    move.add_effect(free(l2), False)
    move.add_effect(at(l1), False)
    move.add_effect(free(l1), True)    
    problem.add_action(move)


    problem.set_initial_value(at(nw), True)
    problem.set_initial_value(free(nw), False)
    
    
    mgs = MultiGoalSplit(MultiGoalSplitType.WCD, goals = [[at(sw)], [at(ne)]])
    res = mgs.compile(problem)
    print(res.problem)

    planner = OneshotPlanner(name="fast-downward")
    sol = planner.solve(res.problem)
    print(sol)


test_wcd()
