# Copyright 2023 Technion
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
#
"""This module defines the robustness verification compiler classes"""

import unified_planning as up
from unified_planning.engines.mixins.compiler import CompilationKind, CompilerMixin
from unified_planning.engines.results import CompilerResult
from unified_planning.model.problem_kind import ProblemKind
import unified_planning.exceptions
from typing import List, Dict, Union, Optional
import unified_planning.engines as engines
from unified_planning.engines import Credits
from enum import Enum, auto
from unified_planning.model import Problem, InstantaneousAction, Action
from unified_planning.model import *
from unified_planning.shortcuts import *
from unified_planning.model.walkers.identitydag import IdentityDagWalker
from unified_planning.engines.compilers.utils import replace_action, get_fresh_name
from functools import partial

credits = Credits('Multi Goal Split',
                  'Technion Cognitive Robotics Lab (cf. https://github.com/TechnionCognitiveRoboticsLab)',
                  'karpase@technion.ac.il',
                  'https://https://cogrob.net.technion.ac.il/',
                  'Apache License, Version 2.0',
                  'Multi-goal split compilation for WCD, centroids, etc.',
                  'Multi-goal split compilation for WCD, centroids, etc.')


class MultiGoalSplitType(Enum):
    WCD = auto()
    CENTROID = auto()


class FluentMapSubstituter(IdentityDagWalker):
    """Performs substitution according to the given FluentMap"""

    def __init__(self, problem : Problem ,env: "unified_planning.environment.Environment"):
        IdentityDagWalker.__init__(self, env, True)
        self.problem = problem
        self.env = env
        self.manager = env.expression_manager
        self.type_checker = env.type_checker

    def _get_key(self, expression, **kwargs):
        return expression

    def substitute(self, expression: FNode, fmap, goal_id) -> FNode:
        """
        Performs substitution into the given expression, according to the given FluentMap
        """
        return self.walk(expression, fmap = fmap, goal_id = goal_id)

    def walk_fluent_exp(self, expression: FNode, args: List[FNode], **kwargs) -> FNode:        
        fluent = expression.fluent()

        goal_id = kwargs["goal_id"]
        assert goal_id is not None
        fmap = kwargs["fmap"]
        assert fmap is not None

        agent_fluent = fmap[fluent][goal_id]

        args = expression.args
        
        return FluentExp(agent_fluent, args)

class MultiGoalSplitCostFunctions:
    epsilon = 0.001

    def zero_cost(original_cost):
        return 0
    
    def epsilon_cost(original_cost):
        return original_cost
    
    def epsilon_discount(original_cost):
        return original_cost * (1.0 / MultiGoalSplitCostFunctions.epsilon) - 1

    def same_cost(original_cost):
        return original_cost * (1.0 / MultiGoalSplitCostFunctions.epsilon)
    


class MultiGoalSplit(engines.engine.Engine, CompilerMixin):
    '''MultiGoal Split (abstract) class:
    this class requires a problem with multiple goals, and creates a classical planning problem in which the agents can move together and then split and move alone.'''
    def __init__(self, 
                 type : MultiGoalSplitType, 
                 goals: List[List["up.model.fnode.FNode"]]):
        engines.engine.Engine.__init__(self)
        CompilerMixin.__init__(self, CompilationKind.MULTI_GOAL_SPLIT)
        self.compilation_type = type
        self.goals = goals # TODO: get this from the multi-goal problem class when we have it        
        self._achieve_goals_sequentially = True
        self._take_turns_after_split = False
        self._budget_from_split = None

        if self.compilation_type == MultiGoalSplitType.CENTROID:
            self._cost_together = MultiGoalSplitCostFunctions.epsilon_cost
        else:
            self._cost_together = MultiGoalSplitCostFunctions.epsilon_discount
        self._cost_split = MultiGoalSplitCostFunctions.same_cost
        
        
    @staticmethod
    def get_credits(**kwargs) -> Optional['Credits']:
        return credits

    @property
    def name(self):
        return "mgs"
    
    @property
    def cost_together(self):
        return self._cost_together
    
    @cost_together.setter
    def cost_together(self, val):
        self._cost_together = val

    @property
    def cost_split(self):
        return self._cost_split
    
    @cost_split.setter
    def cost_split(self, val):
        self._cost_split = val

    @property
    def achieve_goals_sequentially(self):
        return self._achieve_goals_sequentially
    
    @achieve_goals_sequentially.setter
    def achieve_goals_sequentially(self, val):
        self._achieve_goals_sequentially = val

    @property
    def take_turns_after_split(self):
        return self._take_turns_after_split
    
    @take_turns_after_split.setter
    def take_turns_after_split(self, val):
        self._take_turns_after_split = val        
    
    @property
    def budget_from_split(self):
        return self._budget_from_split
    
    @budget_from_split.setter
    def budget_from_split(self, val):
        self._budget_from_split = val
        
    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.MULTI_GOAL_SPLIT

    @staticmethod
    def resulting_problem_kind(
        problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        new_kind = ProblemKind(problem_kind.features)    
        return new_kind
    
    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind = unified_planning.model.problem_kind.classical_kind.union(unified_planning.model.problem_kind.actions_cost_kind)
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= MultiGoalSplit.supported_kind()
    
    def _compile(self, problem: "up.model.AbstractProblem", compilation_kind: "up.engines.CompilationKind") -> CompilerResult:
        if self.take_turns_after_split and self.achieve_goals_sequentially:
            raise unified_planning.exceptions.UPUsageError("Unsupported combinations: take_turns_after_split and achieve_goals_sequentially")
        if self.take_turns_after_split and self.budget_from_split is not None:
            raise unified_planning.exceptions.UPUsageError("Unsupported combinations: take_turns_after_split and budget_from_split != None")
        
        new_to_old: Dict[Action, Action] = {}
        
        new_problem = Problem(f'{self.name}_{problem.name}')

        # Add types
        for type in problem.user_types:
            new_problem._add_user_type(type)

        # Add objects 
        new_problem.add_objects(problem.all_objects)

        # Add fluents
        fluent_map = {}
        for f in problem.fluents:
            fluent_map[f] = {}
            for i, g in enumerate(self.goals):
                goal_name = "g" + str(i)
                fluent_name = goal_name + "__" + f.name
                g_fluent = Fluent(fluent_name, f.type, f.signature)
                new_problem.add_fluent(g_fluent, default_initial_value=problem.fluents_defaults[f])
                fluent_map[f][i] = g_fluent

        # Add split/unplit fluents
        split_fluent = Fluent("split")
        unsplit_fluent = Fluent("unsplit")
        new_problem.add_fluent(split_fluent, default_initial_value = False)
        new_problem.add_fluent(unsplit_fluent, default_initial_value = True)
        
        # Add done_i fluents
        done_map = {}
        if self.achieve_goals_sequentially:
            for i, g in enumerate(self.goals):
                done_map[i] = Fluent("done__" + str(i))
                new_problem.add_fluent(done_map[i], default_initial_value=False)

        # Add turn_i fluents
        turn_map = {}
        if self.take_turns_after_split:
            for i, g in enumerate(self.goals):
                turn_map[i] = Fluent("turn__" + str(i))                
                default_initial_value = (i == 0)
                new_problem.add_fluent(turn_map[i], default_initial_value=default_initial_value)

        # Add B_i fluents
        budget_map = {}
        if self.budget_from_split is not None:
            for i, g in enumerate(self.goals):
                budget_map[i] = Fluent("budget__" + str(i), RealType(0, self.budget_from_split))
                new_problem.add_fluent(budget_map[i], default_initial_value=self.budget_from_split)



        fsub = FluentMapSubstituter(problem, new_problem.environment)

        # Initial state
        eiv = problem.explicit_initial_values     
        for fluent in eiv:
            for i, g in enumerate(self.goals):
                gfluent = fsub.substitute(fluent, fluent_map, i)
                new_problem.set_initial_value(gfluent, eiv[fluent])

        #Goals
        for i,g in enumerate(self.goals):
            for goal_fact in g:
                new_goal_fact = fsub.substitute(goal_fact, fluent_map, i)
                new_problem.add_goal(new_goal_fact)

        # Actions

        # Split action
        split_action = InstantaneousAction("do_split")
        split_action.add_precondition(unsplit_fluent)
        split_action.add_effect(unsplit_fluent, False)
        split_action.add_effect(split_fluent, True)
        new_problem.add_action(split_action)

        new_action_costs_dict = {split_action : 0}

        # Done Actions
        if self.achieve_goals_sequentially:
            for i,g in enumerate(self.goals):
                done_action = InstantaneousAction("end__" + str(i))
                for goal_fact in g:
                    done_action.add_precondition(fsub.substitute(goal_fact, fluent_map, i))
                done_action.add_effect(done_map[i], True)
                new_action_costs_dict[done_action] = 0
                new_problem.add_action(done_action)

        # Noop actions
        if self.take_turns_after_split:
            for i,g in enumerate(self.goals):
                noop_action = InstantaneousAction("noop__" + str(i))
                for goal_fact in g:
                    noop_action.add_precondition(fsub.substitute(goal_fact, fluent_map, i))
                noop_action.add_precondition(split_fluent)
                noop_action.add_precondition(turn_map[i])
                noop_action.add_effect(turn_map[i], False)
                noop_action.add_effect(turn_map[(i + 1) % len(self.goals)], True)
                if i == 0:
                    new_cost = MultiGoalSplitCostFunctions.same_cost(1)
                else:
                    new_cost = 0
                new_problem.add_action(noop_action)
                new_action_costs_dict[noop_action] = new_cost

        # Regular Actions
        for action in problem.actions:
            original_action_cost = 1
            for qm in problem.quality_metrics:
                if qm.is_minimize_action_costs():
                    original_action_cost = qm.get_action_cost(action)
                    if original_action_cost != 1 and self.take_turns_after_split:
                        raise unified_planning.exceptions.UPUsageError("take_turns_after_split can not handle non-unit action costs")
                elif qm.is_minimize_sequential_plan_length():
                    original_action_cost = 1
                else:
                    raise unified_planning.exceptions.UPUsageError("can not handle metric ", qm)

            d = {}
            for p in action.parameters:
                d[p.name] = p.type

            together_action = InstantaneousAction(
                    "__".join(["together", action.name]), _parameters=d)
            together_action.add_precondition(unsplit_fluent)
            for fact in action.preconditions:
                for i,g in enumerate(self.goals):
                    together_action.add_precondition(fsub.substitute(fact, fluent_map, i))                
            for effect in action.effects:
                for i,g in enumerate(self.goals):
                    together_action.add_effect(fsub.substitute(effect.fluent, fluent_map, i), effect.value)
            new_problem.add_action(together_action)
            new_to_old[together_action] = action

            new_cost = self._cost_together(original_action_cost * len(self.goals))
            new_action_costs_dict[together_action] = new_cost

            for i,g in enumerate(self.goals):
                goal_name = "g" + str(i)
                new_action = InstantaneousAction(
                    "__".join([goal_name, action.name]), _parameters=d)
                new_action.add_precondition(split_fluent)

                if self.achieve_goals_sequentially and i > 0:
                    new_action.add_precondition(done_map[i-1])
                if self.budget_from_split is not None:
                    new_action.add_precondition(GE(budget_map[i], original_action_cost))
                    new_action.add_decrease_effect(budget_map[i], original_action_cost)
                if self.take_turns_after_split:
                    new_action.add_precondition(turn_map[i])
                    new_action.add_effect(turn_map[i], False)
                    new_action.add_effect(turn_map[(i + 1) % len(self.goals)], True)

                for fact in action.preconditions:
                    new_action.add_precondition(fsub.substitute(fact, fluent_map, i))                
                for effect in action.effects:
                    new_action.add_effect(fsub.substitute(effect.fluent, fluent_map, i), effect.value)
                new_problem.add_action(new_action)
                new_to_old[new_action] = action

                if self.take_turns_after_split:
                    if i == 0:
                        new_cost = self._cost_split(original_action_cost)
                    else:
                        new_cost = 0
                else:        
                    new_cost = self._cost_split(original_action_cost)
                new_action_costs_dict[new_action] = new_cost


        new_problem.add_quality_metric(MinimizeActionCosts(new_action_costs_dict))

        
        return CompilerResult(
            new_problem, partial(replace_action, map=new_to_old), self.name
        )  
