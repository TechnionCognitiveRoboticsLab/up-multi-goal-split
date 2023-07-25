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



class TestProblem(TestCase):
    def setUp(self):
        TestCase.setUp(self)        

    def create_problem(self, grid_size = 5):
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

    def test_wcd(self):
        problem, goals = self.create_problem()
        
        mgs = MultiGoalSplit(MultiGoalSplitType.WCD, goals = goals)            
        res = mgs.compile(problem)
        #print(res.problem)

        planner = OneshotPlanner(name="fast-downward-opt")
        sol = planner.solve(res.problem)
        print("WCD", sol)

    def test_wcd_without_achieve_goals_sequentially(self):
        problem, goals = self.create_problem()
        
        mgs = MultiGoalSplit(MultiGoalSplitType.WCD, goals = goals)
        mgs.achieve_goals_sequentially = False               
        res = mgs.compile(problem)
        #print(res.problem)

        planner = OneshotPlanner(name="fast-downward-opt")
        sol = planner.solve(res.problem)
        print("WCD", sol)    

    def test_centroid_without_achieve_goals_sequentially(self):
        problem, goals = self.create_problem()
        
        mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)
        mgs.achieve_goals_sequentially = False   
        res = mgs.compile(problem)
        #print(res.problem)

        planner = OneshotPlanner(name="fast-downward-opt")
        sol = planner.solve(res.problem)
        print("CENTROID", sol)    

    def test_centroid(self):
        problem, goals = self.create_problem()
        
        mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)            
        res = mgs.compile(problem)
        #print(res.problem)

        planner = OneshotPlanner(name="fast-downward-opt")
        sol = planner.solve(res.problem)
        print("CENTROID", sol)    


    def test_mincover_budget(self):
        problem, goals = self.create_problem(grid_size = 3)
        
        mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)
        mgs.budget_from_split = 1
        res = mgs.compile(problem)
        #print(res.problem)

        planner = OneshotPlanner(name="tamer")
        sol = planner.solve(res.problem)
        print("MIN-COVER-1", sol)
        self.assertIn(sol.status, [PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY, PlanGenerationResultStatus.UNSOLVABLE_PROVEN])

        mgs.budget_from_split = 2
        res = mgs.compile(problem)        

        planner = OneshotPlanner(name="tamer")
        sol = planner.solve(res.problem)
        print("MIN-COVER-2", sol)
        self.assertIn(sol.status, [PlanGenerationResultStatus.SOLVED_SATISFICING, PlanGenerationResultStatus.SOLVED_OPTIMALLY])

    def test_mincover_turns(self):
        problem, goals = self.create_problem()
        
        mgs = MultiGoalSplit(MultiGoalSplitType.CENTROID, goals = goals)
        mgs.take_turns_after_split = True
        mgs.achieve_goals_sequentially = False
        #mgs.cost_together = MultiGoalSplitCostFunctions.zero_cost
        res = mgs.compile(problem)
        print(res.problem)

        planner = OneshotPlanner(name="fast-downward-opt")
        sol = planner.solve(res.problem)
        print("MIN-COVER-TURNS", sol)

            
