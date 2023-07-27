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

def create_problem(grid_size = 5):
    problem = Problem()
    
    loc = UserType("loc")

    # Environment     
    connected = Fluent('connected', BoolType(), l1=loc, l2=loc)        
    problem.add_fluent(connected, default_initial_value=False)

    free = Fluent('free', BoolType(), l=loc)
    problem.add_fluent(free, default_initial_value=True)

    
    cells = [
        [Object("l_" + str(i) + "_" + str(j), loc) for i in range(grid_size)]
        for j in range(grid_size)
    ]
    for row in cells:
        problem.add_objects(row)

    for i in range(grid_size):
        for j in range(grid_size):
            if i + 1 < grid_size:
                problem.set_initial_value(connected(cells[i][j], cells[i+1][j]), True)
            if i - 1 >= 0:
                problem.set_initial_value(connected(cells[i][j], cells[i-1][j]), True)
            if j + 1 < grid_size:
                problem.set_initial_value(connected(cells[i][j], cells[i][j+1]), True)
            if j - 1 >= 0:
                problem.set_initial_value(connected(cells[i][j], cells[i][j-1]), True)
    

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


    problem.set_initial_value(at(cells[0][0]), True)
    problem.set_initial_value(free(cells[0][0]), False)

    goals = [
        [at(cells[grid_size - 1][0])], 
        [at(cells[0][grid_size - 1])],
        [at(cells[grid_size - 1][grid_size - 1])],
        [at(cells[0][grid_size - 2])],
        [at(cells[1][grid_size - 1])]        
        ]
    problem.add_goal(goals[0][0])
    return problem, goals

def test_wcd():
    problem, goals = create_problem()
    
    mgs = MultiGoalSplit(MultiGoalSplitType.WCD, goals = goals)            

    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="fast-downward-opt")

    print("WCD", sol)

def test_wcd_without_achieve_goals_sequentially():
    problem, goals = create_problem()
    
    mgs = MultiGoalSplit(MultiGoalSplitType.WCD, goals = goals)

    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="fast-downward-opt")
    mgs.achieve_goals_sequentially = False               

    print("WCD", sol)    

def test_centroid_without_achieve_goals_sequentially():
    problem, goals = create_problem()
    
    mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)
    mgs.achieve_goals_sequentially = False   

    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="fast-downward-opt")
    
    print("CENTROID", sol)    

def test_centroid():
    problem, goals = create_problem()
    
    mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)            

    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="fast-downward-opt")

    print("CENTROID", sol)    


def test_mincover_budget():
    problem, goals = create_problem(grid_size = 3)
    
    mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)
    mgs.budget_from_split = 1
    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="tamer")
    print("MIN-COVER-1", sol)

    mgs.budget_from_split = 2
    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="tamer")
    print("MIN-COVER-2", sol)

def test_mincover_turns():
    problem, goals = create_problem()
    
    mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)
    mgs.take_turns_after_split = True
    mgs.achieve_goals_sequentially = False

    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="fast-downward-opt")
    
    print("MIN-COVER-TURNS", sol)

def test_deception_point():
    problem, goals = create_problem()

    mgs = deception_point_compilation(goals)

    sol = mgs.solve_and_get_plan_until_split(problem, planner_name="fast-downward-opt")
    
    print("DECEPTION-POINT", sol)


test_wcd()
test_wcd_without_achieve_goals_sequentially()

test_deception_point()

test_centroid()
test_centroid_without_achieve_goals_sequentially()

test_mincover_budget()
test_mincover_turns()